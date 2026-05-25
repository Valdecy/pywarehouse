"""Self-contained HTML web viewer for pyWarehouse.

Produces a single ``.html`` file containing a custom SVG/JS visualization
of a warehouse layout and (optionally) a solved route. The viewer offers
three modes — Layout, Route, Animation — with a scrubbable timeline,
playback controls, a telemetry rail, the full pick sequence, keyboard
shortcuts, and fullscreen support.

The file is fully self-contained except for the Google Fonts ``<link>``
in ``<head>``; if offline, the CSS font stack falls back to common
mono / sans system faces so the layout still works.
"""

from __future__ import annotations

import html as _html
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .graph import edge_weight, undirected_key
from .models import Graph, Point, Route, Segment


# ---------------------------------------------------------------------------
# Path interpolation (shared with Plotly animation logic, copied to keep web.py
# usable without importing visualization.py and without creating a cycle)
# ---------------------------------------------------------------------------

def _interpolate_path(
    path: Sequence[Point],
    target_step: float = 1.0,
    max_frames: int = 400,
) -> List[Point]:
    """Subdivide axis-aligned route edges into animation frames.

    The returned smooth path always tries to preserve the original graph
    vertices. This matters for sequence seeking: clicking a pick should seek to
    the frame where the picker actually reaches that product, not to a nearby
    approximated frame before the product.
    """
    if not path:
        return []
    if len(path) == 1:
        return [tuple(path[0])]  # type: ignore[list-item]

    target_step = max(float(target_step), 1e-9)
    max_frames = max(int(max_frames), 2)

    out: List[Point] = [tuple(path[0])]  # type: ignore[list-item]
    protected = {0}

    for i in range(1, len(path)):
        u = path[i - 1]
        v = path[i]
        dx = v[0] - u[0]
        dy = v[1] - u[1]
        edge_len = abs(dx) + abs(dy)
        if edge_len <= target_step + 1e-9:
            out.append(tuple(v))  # type: ignore[arg-type]
            protected.add(len(out) - 1)
            continue
        n_steps = max(1, int(round(edge_len / target_step)))
        for s in range(1, n_steps + 1):
            t = s / n_steps
            out.append((u[0] + t * dx, u[1] + t * dy))
        protected.add(len(out) - 1)

    # Uniform downsampling used to be allowed to drop graph vertices. Dropping a
    # vertex can make the trail draw diagonally across a corner and can make a
    # sequence click stop before a pick. Keep protected indices whenever possible.
    if len(out) > max_frames:
        if len(protected) >= max_frames:
            protected_sorted = sorted(protected)
            step = (len(protected_sorted) - 1) / (max_frames - 1)
            keep = {protected_sorted[min(int(round(i * step)), len(protected_sorted) - 1)] for i in range(max_frames)}
        else:
            keep = set(protected)
            needed = max_frames - len(keep)
            step = (len(out) - 1) / max(needed + 1, 1)
            for i in range(1, needed + 1):
                keep.add(min(int(round(i * step)), len(out) - 1))
        out = [out[i] for i in sorted(keep)]

    return out


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def _all_edges(G: Graph) -> List[Tuple[Point, Point]]:
    edges: List[Tuple[Point, Point]] = []
    seen = set()
    for u, nbrs in G.items():
        for v in nbrs:
            k = undirected_key(u, v)
            if k not in seen:
                seen.add(k)
                edges.append(k)
    return edges


def _label_for(node_attrs: Dict[Point, Dict[str, Any]], p: Point) -> str:
    a = node_attrs.get(p, {})
    if a.get("type") == "terminal":
        kinds = a.get("kinds", [])
        if ("start" in kinds and "finish" in kinds) or a.get("label") == "Start/Finish":
            return "S/F"
        if "start" in kinds:
            return "S"
        if "finish" in kinds:
            return "F"
        return str(a.get("id", ""))
    return str(a.get("id", ""))


def make_item_labels(
    node_attrs: Dict[Point, Dict[str, Any]],
    mode: str = "id",
    prefix: str = "P",
    pad: int = 3,
) -> Dict[Point, str]:
    """Create display labels for product and start/finish terminal nodes.

    Parameters
    ----------
    node_attrs:
        Node-attribute dictionary returned by ``WarehouseLayout.build_graph``.
    mode:
        ``"id"`` keeps the terminal ids, ``"p001"``/``"p"`` creates
        sequential labels such as ``P001``, and ``"letters"`` creates
        ``A``, ``B``, ``C``... labels.
    prefix, pad:
        Prefix and zero-padding used by the sequential ``p001`` mode.
    """
    picks = sorted(
        [
            p for p, a in node_attrs.items()
            if a.get("type") == "terminal" and "pick" in a.get("kinds", [])
        ],
        key=lambda q: (q[0], q[1]),
    )
    labels: Dict[Point, str] = {}
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, p in enumerate(picks, start=1):
        attrs = node_attrs[p]
        m = mode.lower()
        if m == "id":
            labels[p] = str(attrs.get("id", f"{prefix}{i:0{pad}d}"))
        elif m in {"p001", "p"}:
            labels[p] = f"{prefix}{i:0{pad}d}"
        elif m == "letters":
            if i <= len(alphabet):
                labels[p] = alphabet[i - 1]
            else:
                labels[p] = f"{alphabet[(i - 1) % len(alphabet)]}{(i - 1) // len(alphabet)}"
        else:
            labels[p] = str(attrs.get("id", f"{prefix}{i:0{pad}d}"))

    for p, a in node_attrs.items():
        if a.get("type") == "terminal" and "pick" not in a.get("kinds", []):
            kinds = a.get("kinds", [])
            if ("start" in kinds and "finish" in kinds) or a.get("label") == "Start/Finish":
                labels[p] = "S/F"
            elif "start" in kinds:
                labels[p] = "S"
            elif "finish" in kinds:
                labels[p] = "F"
            else:
                labels[p] = str(a.get("id", a.get("label", "")))
    return labels


def _manhattan_length(path: Sequence[Point]) -> float:
    return float(
        sum(abs(path[i][0] - path[i - 1][0]) + abs(path[i][1] - path[i - 1][1]) for i in range(1, len(path)))
    )


def _edge_len_or_manhattan(G: Graph, u: Point, v: Point) -> float:
    """Return graph edge length, falling back to Manhattan for synthetic paths."""
    try:
        return float(edge_weight(G, u, v))
    except Exception:
        return float(abs(v[0] - u[0]) + abs(v[1] - u[1]))


def _cumulative_lengths(path: Sequence[Point], G: Optional[Graph] = None) -> List[float]:
    """Cumulative route length at every point in a path."""
    if not path:
        return []
    out = [0.0]
    total = 0.0
    for i in range(1, len(path)):
        u = tuple(path[i - 1])  # type: ignore[arg-type]
        v = tuple(path[i])      # type: ignore[arg-type]
        total += _edge_len_or_manhattan(G, u, v) if G is not None else abs(v[0] - u[0]) + abs(v[1] - u[1])
        out.append(float(total))
    return out


def _terminal_points_by_id(node_attrs: Dict[Point, Dict[str, Any]]) -> Dict[str, Point]:
    mapping: Dict[str, Point] = {}
    for p, attrs in node_attrs.items():
        if attrs.get("type") != "terminal":
            continue
        raw_id = str(attrs.get("id") or "")
        for tid in raw_id.split("/"):
            if tid:
                mapping[tid] = p
        label = str(attrs.get("label") or "")
        if label and raw_id:
            mapping.setdefault(label, p)
    return mapping


def _sequence_node_indices(
    terminal_sequence: Sequence[str],
    node_path: Sequence[Point],
    node_attrs: Dict[Point, Dict[str, Any]],
) -> List[int]:
    """Map each requested terminal id to the node_path index reached in order."""
    if not terminal_sequence or not node_path:
        return []
    point_by_id = _terminal_points_by_id(node_attrs)
    out: List[int] = []
    cursor = 0
    last_idx = 0
    for pos, tid in enumerate(terminal_sequence):
        target = point_by_id.get(str(tid))
        found: Optional[int] = None
        if target is not None:
            # Same physical depot may be both START and FINISH. The last sequence
            # element should resolve to the final occurrence, not the first.
            if pos == len(terminal_sequence) - 1:
                scan = range(len(node_path) - 1, max(cursor - 1, -1), -1)
            else:
                scan = range(cursor, len(node_path))
            for i in scan:
                if tuple(node_path[i]) == target:  # type: ignore[arg-type]
                    found = i
                    break
        if found is None:
            found = last_idx
        out.append(found)
        last_idx = found
        cursor = min(found + 1, len(node_path) - 1)
    return out


def _distance_to_frame_indices(smooth_cum: Sequence[float], target_distances: Sequence[float]) -> List[int]:
    """Return first smooth frame whose cumulative distance reaches each target."""
    if not smooth_cum:
        return []
    frames: List[int] = []
    cursor = 0
    for d in target_distances:
        while cursor + 1 < len(smooth_cum) and smooth_cum[cursor] + 1e-9 < d:
            cursor += 1
        frames.append(cursor)
    return frames


def _path_from_segments(segments: Optional[Sequence[Segment]]) -> List[Point]:
    if not segments:
        return []
    out: List[Point] = []
    for seg in segments:
        nodes = list(getattr(seg, "nodes", []) or [])
        if not nodes:
            continue
        if out and out[-1] == nodes[0]:
            out.extend(nodes[1:])
        else:
            out.extend(nodes)
    return out


def _infer_terminal_sequence(
    node_attrs: Dict[Point, Dict[str, Any]],
    node_path: Sequence[Point],
) -> List[str]:
    seq: List[str] = []
    for p in node_path:
        attrs = node_attrs.get(tuple(p), {})  # type: ignore[arg-type]
        if attrs.get("type") != "terminal":
            continue
        tid = str(attrs.get("id") or _label_for(node_attrs, tuple(p)))  # type: ignore[arg-type]
        if not seq or seq[-1] != tid:
            seq.append(tid)
    return seq


def _build_payload(
    G: Graph,
    node_attrs: Dict[Point, Dict[str, Any]],
    route: Optional[Route],
    target_step: float,
    max_frames: int,
    *,
    label_mode: str = "id",
    node_path: Optional[Sequence[Point]] = None,
    segments: Optional[Sequence[Segment]] = None,
) -> Dict[str, Any]:
    """Convert graph + route into the JSON payload consumed by the viewer."""
    # Split edges by orientation for stylistic differentiation.
    v_edges: List[List[List[float]]] = []
    h_edges: List[List[List[float]]] = []
    other_edges: List[List[List[float]]] = []
    for u, v in _all_edges(G):
        if abs(u[0] - v[0]) < 1e-9:
            v_edges.append([[u[0], u[1]], [v[0], v[1]]])
        elif abs(u[1] - v[1]) < 1e-9:
            h_edges.append([[u[0], u[1]], [v[0], v[1]]])
        else:
            other_edges.append([[u[0], u[1]], [v[0], v[1]]])

    # Terminals and junctions.
    labels = make_item_labels(node_attrs, mode=label_mode)
    junctions: List[List[float]] = []
    picks: List[Dict[str, Any]] = []
    sf: List[Dict[str, Any]] = []
    for p, attrs in node_attrs.items():
        if attrs.get("type") == "steiner":
            junctions.append([p[0], p[1]])
            continue
        if attrs.get("type") != "terminal":
            continue
        kinds = attrs.get("kinds", [])
        raw_id = str(attrs.get("id", ""))
        aliases = [a for a in raw_id.split("/") if a]
        entry: Dict[str, Any] = {
            "x": float(p[0]),
            "y": float(p[1]),
            "id": raw_id,
            "label": labels.get(p, _label_for(node_attrs, p)),
            "kinds": list(kinds),
            "aliases": aliases or ([raw_id] if raw_id else []),
        }
        if "pick" in kinds:
            picks.append(entry)
        else:
            sf.append(entry)
    picks.sort(key=lambda e: e["id"])

    # Bounds from every known node.
    xs: List[float] = [p[0] for p in G.keys()]
    ys: List[float] = [p[1] for p in G.keys()]
    bounds = {
        "xmin": float(min(xs)),
        "xmax": float(max(xs)),
        "ymin": float(min(ys)),
        "ymax": float(max(ys)),
    }

    # Route + smooth path.
    route_node_path: List[Point] = []
    route_segments: Optional[Sequence[Segment]] = None
    strategy = ""
    total_distance = 0.0
    seq: List[str] = []
    waypoint_sequence: List[str] = []
    waypoint_details: List[Dict[str, Any]] = []

    if route is not None:
        route_node_path = list(getattr(route, "node_path", []) or [])
        route_segments = getattr(route, "segments", None)
        strategy = str(getattr(route, "strategy", "") or "")
        total_distance = float(getattr(route, "total_distance", 0.0) or 0.0)
        seq = [str(s) for s in (getattr(route, "terminal_sequence", []) or [])]
        waypoint_sequence = [str(s) for s in (getattr(route, "waypoint_sequence", []) or [])]
        waypoint_details = list(getattr(route, "waypoint_details", []) or [])
    else:
        if node_path is not None:
            route_node_path = list(node_path)
        elif segments is not None:
            route_node_path = _path_from_segments(segments)
        route_segments = segments
        total_distance = _manhattan_length(route_node_path)
        seq = _infer_terminal_sequence(node_attrs, route_node_path)
        waypoint_sequence = seq[:]
        waypoint_details = []

    route_data: Optional[Dict[str, Any]] = None
    if route_node_path:
        smooth = _interpolate_path(route_node_path, target_step, max_frames)
        pick_ids = {entry["id"] for entry in picks}
        for entry in picks:
            pick_ids.update(str(a) for a in entry.get("aliases", []))
        sf_ids = {entry["id"] for entry in sf}
        for entry in sf:
            sf_ids.update(str(a) for a in entry.get("aliases", []))
        n_picks = sum(1 for s in seq if s in pick_ids)

        route_edges = [
            [[route_node_path[i - 1][0], route_node_path[i - 1][1]],
             [route_node_path[i][0], route_node_path[i][1]]]
            for i in range(1, len(route_node_path))
        ]

        node_cum = _cumulative_lengths(route_node_path, G)
        # The HTML must display the distance of the route that it actually draws.
        # If a stale/mutated Route object carries a different total_distance,
        # prefer the graph length of route.node_path; otherwise the telemetry,
        # scrubber, and terminal printout can diverge.
        if node_cum:
            path_distance = float(node_cum[-1])
            if total_distance <= 0 or abs(path_distance - total_distance) > 1e-6:
                total_distance = path_distance
        elif route_segments and total_distance <= 0:
            total_distance = float(sum(float(getattr(seg, "length", 0.0) or 0.0) for seg in route_segments))
        if total_distance <= 0:
            total_distance = _manhattan_length(route_node_path)

        smooth_cum = _cumulative_lengths(smooth)
        if smooth_cum and total_distance > 0 and abs(smooth_cum[-1] - total_distance) > 1e-6:
            # Keep animation telemetry in the same scale as the graph route.
            scale = total_distance / smooth_cum[-1] if smooth_cum[-1] > 0 else 1.0
            smooth_cum = [d * scale for d in smooth_cum]

        seq_node_indices = _sequence_node_indices(seq, route_node_path, node_attrs)
        seq_distances = [node_cum[i] if 0 <= i < len(node_cum) else 0.0 for i in seq_node_indices]
        event_frames = _distance_to_frame_indices(smooth_cum, seq_distances)

        frame_pick_counts: List[int] = []
        pick_event_frames = [
            frame for frame, tid in zip(event_frames, seq)
            if tid in pick_ids
        ]
        for i in range(len(smooth)):
            frame_pick_counts.append(sum(1 for f in pick_event_frames if f <= i))

        route_data = {
            "strategy": strategy,
            "total_distance": float(total_distance),
            "n_picks": int(n_picks),
            "node_path": [[float(p[0]), float(p[1])] for p in route_node_path],
            "smooth_path": [[float(p[0]), float(p[1])] for p in smooth],
            "frame_distance": [float(d) for d in smooth_cum],
            "frame_picks": frame_pick_counts,
            "event_frames": event_frames,
            "event_distance": [float(d) for d in seq_distances],
            "edges": route_edges,
            "terminal_sequence": seq,
            "waypoint_sequence": waypoint_sequence or seq,
            "waypoint_details": waypoint_details,
        }

    return {
        "bounds": bounds,
        "edges": {"vertical": v_edges, "horizontal": h_edges, "other": other_edges},
        "junctions": junctions,
        "picks": picks,
        "sf": sf,
        "route": route_data,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def export_html(
    G: Graph,
    node_attrs: Dict[Point, Dict[str, Any]],
    route: Optional[Route] = None,
    output_path: str = "warehouse.html",
    *,
    title: str = "pyWarehouse",
    subtitle: str = "Warehouse Route Visualizer",
    label_mode: str = "id",
    initial_view: str = "layout",
    target_step: float = 1.0,
    max_frames: int = 400,
) -> str:
    """Render an interactive HTML viewer to ``output_path`` and return the path."""
    payload = _build_payload(
        G, node_attrs, route, target_step, max_frames, label_mode=label_mode
    )
    html_doc = _render_template(payload, title=title, subtitle=subtitle, initial_view=initial_view)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    return output_path


class HtmlViewer:
    """Small, Plotly-like wrapper around the self-contained web viewer.

    The wrapper intentionally exposes ``write_html(...)`` so existing code such
    as ``Plotter(...).draw_route(...).write_html('route.html')`` keeps working,
    but the generated output is now the custom SVG/JavaScript viewer from
    ``web.py`` rather than a Plotly figure.
    """

    def __init__(
        self,
        G: Graph,
        node_attrs: Dict[Point, Dict[str, Any]],
        route: Optional[Route] = None,
        *,
        node_path: Optional[Sequence[Point]] = None,
        segments: Optional[Sequence[Segment]] = None,
        title: str = "pyWarehouse",
        subtitle: str = "Warehouse Route Visualizer",
        label_mode: str = "id",
        initial_view: str = "layout",
        target_step: float = 1.0,
        max_frames: int = 400,
    ) -> None:
        self.G = G
        self.node_attrs = node_attrs
        self.route = route
        self.node_path = node_path
        self.segments = segments
        self.title = title
        self.subtitle = subtitle
        self.label_mode = label_mode
        self.initial_view = initial_view
        self.target_step = target_step
        self.max_frames = max_frames

    def to_html(self, *args: Any, **kwargs: Any) -> str:
        """Return the complete self-contained HTML document.

        Extra positional/keyword arguments are accepted for compatibility with
        Plotly's ``to_html``/``write_html`` call sites and are ignored.
        """
        payload = _build_payload(
            self.G,
            self.node_attrs,
            self.route,
            self.target_step,
            self.max_frames,
            label_mode=self.label_mode,
            node_path=self.node_path,
            segments=self.segments,
        )
        return _render_template(
            payload, title=self.title, subtitle=self.subtitle, initial_view=self.initial_view
        )

    def write_html(self, file: str = "warehouse.html", *args: Any, **kwargs: Any) -> str:
        """Write the viewer to an HTML file and return the file path.

        Plotly-specific kwargs such as ``include_plotlyjs`` are accepted and
        ignored, so old example code does not break.
        """
        with open(file, "w", encoding="utf-8") as f:
            f.write(self.to_html())
        return file

    save = write_html

    def show(self, file: str = "warehouse.html", *args: Any, **kwargs: Any) -> str:
        """Write and open the viewer in the default browser; return the path."""
        path = self.write_html(file)
        try:
            import os
            import webbrowser

            webbrowser.open("file://" + os.path.abspath(path))
        except Exception:
            pass
        return path

    def _repr_html_(self) -> str:
        return self.to_html()


class Plotter:
    """Public web-first plotter for warehouse layouts and routes.

    This class replaces the previous Plotly-backed default plotter while
    keeping the same high-level method names: ``draw_layout``, ``draw_route``,
    and ``animate_route``.
    """

    def __init__(self, G: Graph, node_attrs: Dict[Point, Dict[str, Any]]):
        self.G = G
        self.node_attrs = node_attrs

    def draw_layout(
        self,
        *,
        label_mode: str = "id",
        width: Optional[int] = None,
        height: Optional[int] = None,
        title: str = "Warehouse Layout",
        **kwargs: Any,
    ) -> HtmlViewer:
        return HtmlViewer(
            self.G,
            self.node_attrs,
            None,
            title=title,
            subtitle="Interactive Warehouse Layout",
            label_mode=label_mode,
            initial_view="layout",
        )

    def draw_route(
        self,
        route: Optional[Route] = None,
        *,
        node_path: Optional[Sequence[Point]] = None,
        segments: Optional[Sequence[Segment]] = None,
        label_mode: str = "id",
        width: Optional[int] = None,
        height: Optional[int] = None,
        title: str = "Warehouse Route",
        show_legend: bool = True,
        **kwargs: Any,
    ) -> HtmlViewer:
        return HtmlViewer(
            self.G,
            self.node_attrs,
            route,
            node_path=node_path,
            segments=segments,
            title=title,
            subtitle="Interactive Warehouse Route",
            label_mode=label_mode,
            initial_view=kwargs.pop("initial_view", "layout"),
        )

    def animate_route(
        self,
        route: Route,
        *,
        label_mode: str = "id",
        width: Optional[int] = None,
        height: Optional[int] = None,
        title: str = "Warehouse Route Animation",
        target_step: float = 1.0,
        max_frames: int = 400,
        frame_duration: int = 55,
        transition_duration: int = 40,
        **kwargs: Any,
    ) -> HtmlViewer:
        return HtmlViewer(
            self.G,
            self.node_attrs,
            route,
            title=title,
            subtitle="Interactive Picker Animation",
            label_mode=label_mode,
            initial_view=kwargs.pop("initial_view", "layout"),
            target_step=target_step,
            max_frames=max_frames,
        )


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

def _render_template(
    payload: Dict[str, Any], *, title: str, subtitle: str, initial_view: str = "layout"
) -> str:
    data_json = json.dumps(payload, separators=(",", ":"))
    if initial_view not in {"layout", "route", "animation"}:
        initial_view = "layout"
    return (
        _TEMPLATE
        .replace("__TITLE__", _html.escape(title))
        .replace("__SUBTITLE__", _html.escape(subtitle))
        .replace("__DATA_JSON__", data_json)
        .replace("__INITIAL_VIEW__", initial_view)
    )


# The full HTML / CSS / JS template lives at module level for readability.
# Brace literals would conflict with str.format, so substitution uses
# explicit double-underscore placeholders instead.
_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
<title>__TITLE__ &middot; __SUBTITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link rel="stylesheet"
      href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400;1,6..72,500&family=Inter+Tight:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" />
<style>
:root {
  /* ---- Palette: navy ink, honey gold ---- */
  --bg-deep:   #0a1018;
  --bg:        #0e1623;
  --bg-soft:   #131c2c;
  --bg-rail:   #0c121d;
  --panel:        rgba(14, 22, 35, 0.92);
  --panel-strong: rgba(19, 28, 44, 0.96);

  --hairline:        rgba(180, 200, 235, 0.08);
  --hairline-strong: rgba(180, 200, 235, 0.16);
  --hairline-bright: rgba(180, 200, 235, 0.28);

  --text:       #e6ecf5;
  --text-dim:   #aab4c7;
  --text-muted: #6e7d96;
  --text-faint: rgba(230, 236, 245, 0.32);

  /* The honey amber. Used as text fill and hairline border, never as a glow halo. */
  --amber:       #e9b870;
  --amber-deep:  #c8924a;
  --amber-soft:  rgba(233, 184, 112, 0.16);
  --amber-edge:  rgba(233, 184, 112, 0.38);

  /* Picker (the moving figure during animation) — soft pale, distinct from route */
  --picker:      #8ec1d6;
  --picker-soft: rgba(142, 193, 214, 0.18);

  /* Picks and start/finish — quiet, single-family */
  --pick:        #d4956b;
  --pick-soft:   rgba(212, 149, 107, 0.18);
  --sf:          #d6dde8;
  --sf-soft:     rgba(214, 221, 232, 0.16);

  --aisle: rgba(120, 138, 170, 0.32);
  --cross: rgba(120, 138, 170, 0.42);

  --font-serif: 'Newsreader', 'Source Serif 4', Georgia, 'Times New Roman', serif;
  --font-sans:  'Inter Tight', 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --font-mono:  'IBM Plex Mono', ui-monospace, 'SF Mono', Consolas, monospace;

  --topbar-h:   52px;
  --rail-w:     260px;
  --timeline-h: 78px;

  --ease: cubic-bezier(0.22, 0.61, 0.36, 1);
}

* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-sans);
  font-size: 13px;
  line-height: 1.45;
  overflow: hidden;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* Subtle grid — much softer than before. Decorative, not loud. */
body::before {
  content: "";
  position: fixed; inset: 0;
  background-image:
    linear-gradient(to right,  rgba(180, 200, 235, 0.018) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(180, 200, 235, 0.018) 1px, transparent 1px);
  background-size: 72px 72px;
  pointer-events: none;
  z-index: 0;
}

.app {
  position: relative;
  z-index: 1;
  height: 100vh;
  width: 100vw;
  display: grid;
  grid-template-rows: var(--topbar-h) 1fr var(--timeline-h);
  transition: grid-template-rows 320ms var(--ease);
}
.app.no-timeline { grid-template-rows: var(--topbar-h) 1fr 0px; }

/* =========================================================================
   TOPBAR — calm, hairline-driven, editorial brand
   ========================================================================= */
.topbar {
  display: flex; align-items: center;
  padding: 0 18px;
  border-bottom: 1px solid var(--hairline);
  background: var(--bg);
  gap: 16px;
}
.brand {
  display: flex; align-items: center; gap: 12px;
  min-width: calc(var(--rail-w) - 16px);
  padding-right: 18px;
  border-right: 1px solid var(--hairline);
  height: 100%;
}
/* The mark: a tiny crossed square. Hairline only — no fills, no glow. */
.brand-mark {
  width: 22px; height: 22px;
  border: 1px solid var(--amber-edge);
  border-radius: 3px;
  position: relative;
  background: transparent;
  flex-shrink: 0;
}
.brand-mark::before,
.brand-mark::after {
  content: ""; position: absolute;
  background: var(--amber);
  opacity: 0.70;
}
.brand-mark::before { inset: 4px 0; left: 50%; width: 1px; transform: translateX(-0.5px); }
.brand-mark::after  { inset: 0 4px; top: 50%; height: 1px; transform: translateY(-0.5px); }

.brand-text { display: flex; flex-direction: column; line-height: 1.05; min-width: 0; }
/* THE editorial moment: brand name in serif italic, honey-amber. */
.brand-name {
  font-family: var(--font-serif);
  font-style: italic;
  font-weight: 400;
  font-size: 19px;
  letter-spacing: -0.005em;
  color: var(--amber);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.brand-sub {
  margin-top: 3px;
  font-family: var(--font-mono);
  font-weight: 400;
  font-size: 9.5px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text-muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* Tabs: hairline pills with an amber underline on active — like the reference. */
.nav-tabs {
  display: flex;
  gap: 2px;
  height: 32px;
  align-items: stretch;
  padding: 0;
  background: transparent;
  border: 0;
  border-radius: 0;
}
.nav-tab {
  appearance: none; background: transparent;
  color: var(--text-dim);
  border: 0;
  cursor: pointer;
  font-family: var(--font-mono);
  font-weight: 500;
  font-size: 10.5px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  padding: 0 14px;
  position: relative;
  transition: color 160ms var(--ease);
}
.nav-tab::after {
  content: ""; position: absolute;
  left: 14px; right: 14px;
  bottom: 0; height: 1.5px;
  background: var(--amber);
  transform: scaleX(0);
  transform-origin: left center;
  transition: transform 220ms var(--ease);
}
.nav-tab:hover { color: var(--text); }
.nav-tab.active { color: var(--amber); }
.nav-tab.active::after { transform: scaleX(1); }

.nav-spacer { flex: 1; }

.status-pill {
  display: flex; align-items: center; gap: 8px;
  font-family: var(--font-mono);
  font-size: 9.5px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text-muted);
  padding: 5px 10px;
  border: 1px solid var(--hairline);
  border-radius: 999px;
  background: transparent;
}
.status-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--amber);
  opacity: 0.85;
}

.icon-btn {
  appearance: none; background: transparent;
  border: 1px solid var(--hairline);
  color: var(--text-dim);
  width: 30px; height: 30px;
  border-radius: 5px;
  display: grid; place-items: center;
  cursor: pointer;
  transition: color 160ms var(--ease), border-color 160ms var(--ease);
}
.icon-btn:hover {
  color: var(--amber);
  border-color: var(--amber-edge);
}
.icon-btn svg { width: 14px; height: 14px; }

/* =========================================================================
   WORKSPACE LAYOUT
   ========================================================================= */
.workspace {
  display: grid;
  grid-template-columns: var(--rail-w) 1fr;
  min-height: 0;
  overflow: hidden;
}

/* =========================================================================
   RAIL — editorial sidebar with hairline sections
   ========================================================================= */
.rail {
  border-right: 1px solid var(--hairline);
  background: var(--bg-rail);
  overflow-y: auto;
  overflow-x: hidden;
  padding: 18px 16px 22px;
  display: flex; flex-direction: column;
  gap: 22px;
  scrollbar-width: thin;
  scrollbar-color: rgba(180, 200, 235, 0.18) transparent;
}
.rail::-webkit-scrollbar { width: 6px; }
.rail::-webkit-scrollbar-thumb {
  background: rgba(180, 200, 235, 0.14);
  border-radius: 3px;
}

.rail-section { display: flex; flex-direction: column; gap: 10px; }
.rail-title {
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--text-muted);
  font-weight: 500;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--hairline);
}

/* Metric cards: hairline boxes, value in serif italic gold (the editorial signature). */
.metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.metric {
  position: relative;
  padding: 10px 12px 11px;
  border: 1px solid var(--hairline);
  border-radius: 4px;
  background: transparent;
}
.metric.full { grid-column: 1 / -1; }
.metric-label {
  font-family: var(--font-mono);
  font-size: 7px;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  color: var(--text-muted);
  font-weight: 400;
}
.metric-value {
  margin-top: 4px;
  font-family: var(--font-serif);
  font-style: italic;
  font-weight: 400;
  font-size: 12px;
  letter-spacing: -0.005em;
  color: var(--amber);
  font-variant-numeric: tabular-nums;
  line-height: 1.1;
}
.metric-unit {
  font-family: var(--font-mono);
  font-style: normal;
  font-size: 9px;
  color: var(--text-muted);
  margin-left: 5px;
  font-weight: 400;
  letter-spacing: 0.06em;
}

/* Legend: tiny mono, calm swatches with no glow. */
.legend {
  display: grid; grid-template-columns: 1fr 1fr; gap: 7px 12px;
  padding: 0; margin: 0;
}
.legend li {
  list-style: none;
  display: flex; align-items: center; gap: 8px;
  font-family: var(--font-mono);
  font-size: 9.5px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-dim);
}
.legend-sw {
  width: 10px; height: 10px;
  border-radius: 2px;
  flex-shrink: 0;
}
.legend-sw.dot  { border-radius: 50%; }
.legend-sw.line { height: 2px; border-radius: 1px; width: 14px; }

/* Sequence: hairline track, small dots, italic serif on current pick (the moment of focus). */
.sequence {
  list-style: none; padding: 0; margin: 0;
  display: flex; flex-direction: column;
  position: relative;
}
.sequence::before {
  content: "";
  position: absolute;
  top: 10px; bottom: 10px;
  left: 8px;
  width: 1px;
  background: var(--hairline-strong);
}
.seq-item {
  display: flex; align-items: center; gap: 10px;
  padding: 4px 0;
  cursor: pointer;
  position: relative;
}
.seq-dot {
  width: 16px; height: 16px;
  border-radius: 50%;
  display: grid; place-items: center;
  background: var(--bg-rail);
  border: 1px solid var(--hairline-strong);
  flex-shrink: 0;
  position: relative; z-index: 1;
  font-family: var(--font-mono);
  font-size: 8px;
  color: var(--text-muted);
  font-weight: 500;
}
.seq-item[data-kind="pick"] .seq-dot { border-color: var(--pick); color: var(--pick); }
.seq-item[data-kind="sf"]   .seq-dot { border-color: var(--sf);   color: var(--sf); }
.seq-item .seq-id {
  font-family: var(--font-sans);
  font-size: 12px;
  font-weight: 500;
  color: var(--text);
  letter-spacing: 0.01em;
}
.seq-item .seq-num {
  margin-left: auto;
  font-family: var(--font-mono);
  font-size: 8.5px;
  color: var(--text-muted);
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.seq-item.current .seq-dot {
  background: var(--amber);
  border-color: var(--amber);
  color: var(--bg-deep);
}
.seq-item.current .seq-id {
  color: var(--amber);
  font-family: var(--font-serif);
  font-style: italic;
  font-weight: 400;
  font-size: 14px;
}
.seq-item.done .seq-id  { color: var(--text-muted); }
.seq-item.done .seq-dot { opacity: 0.55; }
.seq-item:hover .seq-id { color: var(--amber); }

/* =========================================================================
   STAGE — the canvas. Calm, no cyan brackets, no neon glow.
   ========================================================================= */
.stage {
  position: relative;
  min-width: 0; min-height: 0;
  overflow: hidden;
  background:
    radial-gradient(ellipse at center, rgba(20, 30, 50, 0.55), transparent 75%),
    repeating-linear-gradient(0deg,  rgba(180, 200, 235, 0.022) 0 1px, transparent 1px 32px),
    repeating-linear-gradient(90deg, rgba(180, 200, 235, 0.022) 0 1px, transparent 1px 32px),
    var(--bg-deep);
}
.stage-frame {
  position: absolute; inset: 14px;
  border: 1px solid var(--hairline);
  border-radius: 3px;
  pointer-events: none;
}
/* No corner brackets — the hairline frame alone is enough. */
.stage-frame::before,
.stage-frame::after,
.stage-frame > span::before,
.stage-frame > span::after {
  content: none;
}

/* The readout — VIEW route — uses the editorial signature. */
.stage-readout {
  position: absolute; top: 22px; left: 24px;
  z-index: 2;
  display: flex; align-items: baseline;
  pointer-events: none;
}
.readout-title {
  font-family: var(--font-mono);
  font-size: 9.5px;
  letter-spacing: 0.26em;
  text-transform: uppercase;
  color: var(--text-muted);
  font-weight: 500;
  display: inline-flex;
  align-items: baseline;
}
.readout-title strong {
  font-family: var(--font-serif);
  font-style: italic;
  font-weight: 400;
  font-size: 26px;
  letter-spacing: -0.005em;
  text-transform: lowercase;
  color: var(--amber);
  margin-left: 14px;
  line-height: 1;
  position: relative;
  top: 4px;
}

.stage-corner {
  position: absolute;
  bottom: 22px; right: 24px;
  z-index: 2;
  display: flex; gap: 18px;
  font-family: var(--font-mono);
  font-size: 9.5px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--text-muted);
  pointer-events: none;
  font-weight: 500;
}
.stage-corner span strong {
  color: var(--text);
  font-weight: 500;
  margin-left: 6px;
}

#viewport {
  position: absolute; inset: 0;
  width: 100%; height: 100%;
  display: block;
}

/* =========================================================================
   SVG ROUTE / AISLES — drastically reduced glow
   The route is now a clean honey line, not a sword of light.
   ========================================================================= */
.aisle-v     { stroke: var(--aisle); stroke-width: 0.18; stroke-linecap: round; opacity: 0.75; }
.aisle-h     { stroke: var(--cross); stroke-width: 0.28; stroke-linecap: round; opacity: 0.80; }
.aisle-other { stroke: var(--cross); stroke-width: 0.20; stroke-linecap: round; opacity: 0.65; }
.junction    { fill: rgba(180, 200, 235, 0.20); }

/* The route: a thin amber line with a barely-there halo. */
.route-glow  { stroke: var(--amber); stroke-width: 0.50; fill: none; stroke-linecap: round; stroke-linejoin: round; opacity: 0.10; }
.route-mid   { stroke: var(--amber); stroke-width: 0.28; fill: none; stroke-linecap: round; stroke-linejoin: round; opacity: 0.45; }
.route-core  { stroke: var(--amber); stroke-width: 0.18; fill: none; stroke-linecap: round; stroke-linejoin: round; opacity: 0.95; }
.route-dim .route-glow { opacity: 0.05; }
.route-dim .route-mid  { opacity: 0.24; }
.route-dim .route-core { opacity: 0.60; }

/* Trail (animation): soft pale-blue moving line. */
.trail-glow  { stroke: var(--picker); stroke-width: 0.55; fill: none; stroke-linecap: round; stroke-linejoin: round; opacity: 0.20; }
.trail-core  { stroke: var(--picker); stroke-width: 0.22; fill: none; stroke-linecap: round; stroke-linejoin: round; opacity: 1; }

/* Picks: muted amber-rose, no glow halo. */
.pick-halo   { fill: var(--pick-soft); }
.pick-fill   { fill: var(--pick); stroke: rgba(14, 22, 35, 0.85); stroke-width: 0.08; }
.pick-label  { font-family: var(--font-mono); font-weight: 500; font-size: 0.58px; fill: var(--bg-deep); text-anchor: middle; dominant-baseline: central; letter-spacing: 0.02em; }

/* Start / Finish: pale, calm. */
.sf-halo     { fill: var(--sf-soft); }
.sf-fill     { fill: var(--sf); stroke: rgba(14, 22, 35, 0.85); stroke-width: 0.08; }
.sf-label    { font-family: var(--font-mono); font-weight: 500; font-size: 0.58px; fill: var(--bg-deep); text-anchor: middle; dominant-baseline: central; }

/* Picker (the moving figure) — pale blue, restrained pulse. */
.picker-halo { fill: var(--picker-soft); }
.picker-ring { fill: none; stroke: var(--picker); stroke-width: 0.08; opacity: 0.7; animation: pickerRing 2.4s ease-in-out infinite; transform-origin: center; transform-box: fill-box; }
.picker-core { fill: var(--picker); stroke: var(--bg-deep); stroke-width: 0.10; }

@keyframes pickerRing {
  0%   { transform: scale(1);   opacity: 0.70; }
  100% { transform: scale(1.7); opacity: 0; }
}

.node-hover {
  fill: rgba(233, 184, 112, 0.08);
  stroke: var(--amber);
  stroke-width: 0.05;
  opacity: 0;
  pointer-events: none;
  transition: opacity 160ms var(--ease);
}
.node-group:hover .node-hover { opacity: 1; }

/* =========================================================================
   TOOLTIP
   ========================================================================= */
.tooltip {
  position: absolute;
  pointer-events: none;
  padding: 7px 11px;
  background: var(--bg-deep);
  border: 1px solid var(--hairline-bright);
  border-radius: 4px;
  font-family: var(--font-mono);
  font-size: 10.5px;
  color: var(--text);
  white-space: nowrap;
  opacity: 0;
  transform: translate(-50%, calc(-100% - 12px));
  transition: opacity 140ms var(--ease);
  z-index: 10;
}
.tooltip.visible { opacity: 1; }
.tooltip .tt-id  { color: var(--amber); font-weight: 500; }
.tooltip .tt-pos { color: var(--text-muted); margin-left: 6px; letter-spacing: 0.04em; }

/* =========================================================================
   TIMELINE — the player. Calm transport, hairline scrubber.
   ========================================================================= */
.timeline {
  display: flex; align-items: center;
  padding: 0 22px;
  border-top: 1px solid var(--hairline);
  background: var(--bg);
  gap: 16px;
  overflow: hidden;
  opacity: 1;
  transition: opacity 240ms var(--ease);
}
.app.no-timeline .timeline {
  opacity: 0;
  pointer-events: none;
}

.transport { display: flex; gap: 6px; align-items: center; }
.transport-btn {
  appearance: none;
  width: 32px; height: 32px;
  border-radius: 50%;
  border: 1px solid var(--hairline);
  background: transparent;
  color: var(--text-dim);
  display: grid; place-items: center;
  cursor: pointer;
  transition: color 180ms var(--ease), border-color 180ms var(--ease);
}
.transport-btn:hover {
  color: var(--amber);
  border-color: var(--amber-edge);
}
.transport-btn.primary {
  width: 38px; height: 38px;
  color: var(--amber);
  background: var(--amber-soft);
  border: 1px solid var(--amber-edge);
}
.transport-btn.primary:hover {
  color: var(--bg-deep);
  background: var(--amber);
  border-color: var(--amber);
}
.transport-btn svg { width: 13px; height: 13px; }
.transport-btn.primary svg { width: 15px; height: 15px; }

.scrubber {
  flex: 1; min-width: 0;
  position: relative;
  height: 32px;
  display: flex; align-items: center;
  cursor: pointer;
  user-select: none;
}
.scrub-track {
  position: relative;
  width: 100%;
  height: 3px;
  border-radius: 1.5px;
  background: rgba(180, 200, 235, 0.10);
  overflow: hidden;
}
.scrub-fill {
  position: absolute; inset: 0;
  width: 0%;
  background: var(--amber);
  border-radius: 1.5px;
  opacity: 0.85;
}
.scrub-handle {
  position: absolute;
  top: 50%; left: 0;
  width: 11px; height: 11px;
  margin-left: -5.5px;
  background: var(--amber);
  border-radius: 50%;
  transform: translateY(-50%);
  border: 2px solid var(--bg);
  transition: transform 80ms var(--ease);
  pointer-events: none;
}
.scrubber:hover .scrub-handle { transform: translateY(-50%) scale(1.15); }

.scrub-ticks {
  position: absolute; left: 0; right: 0;
  top: 50%; height: 10px;
  transform: translateY(-50%);
  pointer-events: none;
}
.scrub-tick {
  position: absolute;
  top: 50%;
  width: 1px; height: 10px;
  margin-left: -0.5px;
  background: var(--pick);
  border-radius: 0;
  transform: translateY(-50%);
  opacity: 0.55;
}

.clock {
  font-family: var(--font-mono);
  font-weight: 500;
  font-size: 11.5px;
  color: var(--text);
  font-variant-numeric: tabular-nums;
  letter-spacing: 0.06em;
  padding: 5px 10px;
  border: 1px solid var(--hairline);
  border-radius: 4px;
  background: transparent;
  min-width: 96px;
  text-align: center;
}
.clock .clock-sep { color: var(--text-muted); margin: 0 4px; }
.clock .clock-tot { color: var(--text-muted); }

.speeds {
  display: flex;
  background: transparent;
  border: 1px solid var(--hairline);
  border-radius: 4px;
  padding: 2px;
  gap: 2px;
}
.speed-btn {
  appearance: none; border: 0;
  background: transparent;
  color: var(--text-dim);
  font-family: var(--font-mono);
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.06em;
  padding: 5px 9px;
  border-radius: 3px;
  cursor: pointer;
  transition: color 160ms var(--ease), background 160ms var(--ease);
}
.speed-btn:hover { color: var(--text); }
.speed-btn.active {
  color: var(--amber);
  background: var(--amber-soft);
}

/* =========================================================================
   RESPONSIVE
   ========================================================================= */
@media (max-width: 880px) {
  :root { --rail-w: 0px; }
  .rail { display: none; }
  .brand { min-width: 0; padding-right: 12px; }
  .nav-tab { padding: 0 10px; font-size: 9.5px; letter-spacing: 0.10em; }
  .status-pill { display: none; }
  .clock { min-width: 80px; font-size: 10.5px; }
  .speeds { display: none; }
}

</style>
</head>
<body>
<div class="app" id="app">

  <!-- ============ TOPBAR ============ -->
  <header class="topbar">
    <div class="brand">
      <div class="brand-mark"></div>
      <div class="brand-text">
        <div class="brand-name">__TITLE__</div>
        <div class="brand-sub">__SUBTITLE__</div>
      </div>
    </div>

    <nav class="nav-tabs" role="tablist">
      <button class="nav-tab" data-view="layout" role="tab">Layout</button>
      <button class="nav-tab active" data-view="route" role="tab">Route</button>
      <button class="nav-tab" data-view="animation" role="tab">Animation</button>
    </nav>

    <div class="nav-spacer"></div>

    <div class="status-pill">
      <span class="status-dot"></span>
      <span id="status-label">ROUTE LOCKED</span>
    </div>

    <button class="icon-btn" id="btn-fit" title="Fit to screen (R)">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <path d="M4 9V5h4"/><path d="M20 9V5h-4"/><path d="M4 15v4h4"/><path d="M20 15v4h-4"/>
      </svg>
    </button>
    <button class="icon-btn" id="btn-fullscreen" title="Fullscreen (F)">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <path d="M8 3H5a2 2 0 0 0-2 2v3"/><path d="M21 8V5a2 2 0 0 0-2-2h-3"/>
        <path d="M3 16v3a2 2 0 0 0 2 2h3"/><path d="M16 21h3a2 2 0 0 0 2-2v-3"/>
      </svg>
    </button>
  </header>

  <!-- ============ WORKSPACE ============ -->
  <main class="workspace">

    <!-- ============ RAIL ============ -->
    <aside class="rail">
      <div class="rail-section">
        <div class="rail-title">Telemetry</div>
        <div class="metric-grid" id="telemetry"></div>
      </div>

      <div class="rail-section">
        <div class="rail-title">Legend</div>
        <ul class="legend">
          <li><span class="legend-sw line" style="color: var(--amber); background: var(--amber);"></span> Route</li>
          <li><span class="legend-sw line" style="color: var(--picker); background: var(--picker);"></span> Picker</li>
          <li><span class="legend-sw dot"  style="color: var(--pick); background: var(--pick);"></span> Item</li>
          <li><span class="legend-sw"      style="color: var(--sf); background: var(--sf);"></span> Start / Finish</li>
        </ul>
      </div>

      <div class="rail-section">
        <div class="rail-title">Pick Sequence</div>
        <ul class="sequence" id="sequence"></ul>
      </div>
    </aside>

    <!-- ============ STAGE ============ -->
    <section class="stage" id="stage">
      <div class="stage-frame"><span></span></div>
      <svg id="viewport" preserveAspectRatio="xMidYMid meet"></svg>
      <div class="tooltip" id="tooltip"></div>
    </section>
  </main>

  <!-- ============ TIMELINE ============ -->
  <footer class="timeline" id="timeline">
    <div class="transport">
      <button class="transport-btn" id="btn-restart" title="Restart (0)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="1 4 1 10 7 10"/>
          <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/>
        </svg>
      </button>
      <button class="transport-btn primary" id="btn-play" title="Play / Pause (Space)">
        <svg viewBox="0 0 24 24" fill="currentColor" id="ic-play">
          <polygon points="6 4 20 12 6 20 6 4"/>
        </svg>
        <svg viewBox="0 0 24 24" fill="currentColor" id="ic-pause" style="display:none;">
          <rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>
        </svg>
      </button>
      <button class="transport-btn" id="btn-step-back" title="Step back (←)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
          <polygon points="19 20 9 12 19 4 19 20" fill="currentColor"/>
          <line x1="5" y1="19" x2="5" y2="5"/>
        </svg>
      </button>
      <button class="transport-btn" id="btn-step-fwd" title="Step forward (→)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
          <polygon points="5 4 15 12 5 20 5 4" fill="currentColor"/>
          <line x1="19" y1="5" x2="19" y2="19"/>
        </svg>
      </button>
    </div>

    <div class="scrubber" id="scrubber">
      <div class="scrub-ticks" id="scrub-ticks"></div>
      <div class="scrub-track">
        <div class="scrub-fill" id="scrub-fill"></div>
      </div>
      <div class="scrub-handle" id="scrub-handle"></div>
    </div>

    <div class="clock" id="clock">
      <span id="clk-now">00</span><span class="clock-sep">/</span><span id="clk-total" class="clock-tot">00</span>
    </div>

    <div class="speeds">
      <button class="speed-btn" data-speed="0.5">&frac12;&times;</button>
      <button class="speed-btn active" data-speed="1">1&times;</button>
      <button class="speed-btn" data-speed="2">2&times;</button>
      <button class="speed-btn" data-speed="4">4&times;</button>
    </div>
  </footer>
</div>

<script>
"use strict";

// ===========================================================================
// Data injected by Python
// ===========================================================================
const DATA = __DATA_JSON__;
const INITIAL_VIEW = "__INITIAL_VIEW__";

const SVG_NS = "http://www.w3.org/2000/svg";

// ===========================================================================
// Coordinate transform
//   data y points UP; SVG y points DOWN.
//   We flip y while rendering by reflecting around (ymin + ymax).
// ===========================================================================
const B = DATA.bounds;
const PAD_X = Math.max((B.xmax - B.xmin) * 0.05, 0.6);
const PAD_Y = Math.max((B.ymax - B.ymin) * 0.04, 0.6);

const VB = {
  x: B.xmin - PAD_X,
  y: B.ymin - PAD_Y,
  w: (B.xmax - B.xmin) + 2 * PAD_X,
  h: (B.ymax - B.ymin) + 2 * PAD_Y,
};
const YFLIP = B.ymin + B.ymax;  // svg_y = YFLIP - data_y

function svgY(y) { return YFLIP - y; }

// ===========================================================================
// SVG construction
// ===========================================================================

const svg = document.getElementById("viewport");
svg.setAttribute("viewBox", `${VB.x} ${VB.y} ${VB.w} ${VB.h}`);

function el(name, attrs, parent) {
  const node = document.createElementNS(SVG_NS, name);
  if (attrs) {
    for (const k in attrs) {
      if (attrs[k] !== undefined && attrs[k] !== null) node.setAttribute(k, attrs[k]);
    }
  }
  if (parent) parent.appendChild(node);
  return node;
}

// ----- defs (filters) -----
const defs = el("defs", {}, svg);
const fCyan = el("filter", { id: "cyanGlow", x: "-50%", y: "-50%", width: "200%", height: "200%" }, defs);
el("feGaussianBlur", { stdDeviation: "0.45" }, fCyan);
const fWarn = el("filter", { id: "warnGlow", x: "-50%", y: "-50%", width: "200%", height: "200%" }, defs);
el("feGaussianBlur", { stdDeviation: "0.55" }, fWarn);

// ----- aisle layer -----
const gAisles = el("g", { id: "g-aisles" }, svg);
function drawEdges(edges, klass) {
  if (!edges || !edges.length) return;
  let d = "";
  for (const e of edges) {
    d += `M${e[0][0]} ${svgY(e[0][1])} L${e[1][0]} ${svgY(e[1][1])} `;
  }
  el("path", { d, class: klass }, gAisles);
}
drawEdges(DATA.edges.vertical,   "aisle-v");
drawEdges(DATA.edges.horizontal, "aisle-h");
drawEdges(DATA.edges.other,      "aisle-other");

// ----- junctions -----
const gJunc = el("g", { id: "g-junc" }, svg);
for (const j of DATA.junctions) {
  el("circle", { cx: j[0], cy: svgY(j[1]), r: 0.10, class: "junction" }, gJunc);
}

// ----- route (drawn beneath nodes) -----
const gRoute = el("g", { id: "g-route" }, svg);
let routePath = null;
if (DATA.route) {
  let d = "";
  const np = DATA.route.node_path;
  for (let i = 0; i < np.length; i++) {
    const cmd = i === 0 ? "M" : "L";
    d += `${cmd}${np[i][0]} ${svgY(np[i][1])} `;
  }
  routePath = d;
  el("path", { d, class: "route-glow" }, gRoute);
  el("path", { d, class: "route-mid"  }, gRoute);
  el("path", { d, class: "route-core" }, gRoute);
}

// ----- trail layer (animation only) -----
const gTrail = el("g", { id: "g-trail" }, svg);
const trailGlow = el("path", { class: "trail-glow", d: "" }, gTrail);
const trailCore = el("path", { class: "trail-core", d: "" }, gTrail);

// ----- nodes (above route, below picker) -----
const gNodes = el("g", { id: "g-nodes" }, svg);

function drawTerminal(entry, opts) {
  const g = el("g", { class: "node-group" }, gNodes);
  g.dataset.id = entry.id;
  g.dataset.x = entry.x;
  g.dataset.y = entry.y;
  const cx = entry.x, cy = svgY(entry.y);

  // halo
  if (opts.halo) {
    el(opts.shape === "square" ? "rect" : "circle",
      opts.shape === "square"
        ? { x: cx - opts.haloR, y: cy - opts.haloR, width: opts.haloR*2, height: opts.haloR*2, rx: opts.haloR*0.35, class: opts.haloClass }
        : { cx, cy, r: opts.haloR, class: opts.haloClass },
      g);
  }
  // hover backdrop
  el("circle", { cx, cy, r: opts.coreR * 1.4, class: "node-hover" }, g);

  // core fill
  if (opts.shape === "square") {
    el("rect",
      { x: cx - opts.coreR, y: cy - opts.coreR, width: opts.coreR*2, height: opts.coreR*2, rx: opts.coreR*0.25, class: opts.fillClass },
      g);
  } else {
    el("circle", { cx, cy, r: opts.coreR, class: opts.fillClass }, g);
  }

  // label
  const t = el("text", { x: cx, y: cy + 0.02, class: opts.labelClass }, g);
  t.textContent = entry.label;

  // hover handlers
  g.addEventListener("mouseenter", () => showTooltip(entry, cx, cy));
  g.addEventListener("mouseleave", hideTooltip);
}

for (const e of DATA.picks) {
  drawTerminal(e, {
    halo: true, haloR: 0.85, haloClass: "pick-halo",
    coreR: 0.55, fillClass: "pick-fill", labelClass: "pick-label",
    shape: "circle",
  });
}
for (const e of DATA.sf) {
  drawTerminal(e, {
    halo: true, haloR: 0.95, haloClass: "sf-halo",
    coreR: 0.62, fillClass: "sf-fill", labelClass: "sf-label",
    shape: "square",
  });
}

// ----- picker (animation only, drawn on top) -----
const gPicker = el("g", { id: "g-picker", style: "display:none;" }, svg);
const pickerHalo = el("circle", { cx: 0, cy: 0, r: 0.95, class: "picker-halo" }, gPicker);
const pickerRing = el("circle", { cx: 0, cy: 0, r: 0.55, class: "picker-ring" }, gPicker);
const pickerCore = el("circle", { cx: 0, cy: 0, r: 0.48, class: "picker-core" }, gPicker);

function placePicker(x, y) {
  const sy = svgY(y);
  pickerHalo.setAttribute("cx", x); pickerHalo.setAttribute("cy", sy);
  pickerRing.setAttribute("cx", x); pickerRing.setAttribute("cy", sy);
  pickerCore.setAttribute("cx", x); pickerCore.setAttribute("cy", sy);
}

// ===========================================================================
// Tooltip
// ===========================================================================
const tooltip = document.getElementById("tooltip");
const stageEl = document.getElementById("stage");

function showTooltip(entry, cx, cy) {
  // Compute screen position from SVG coords using getBoundingClientRect of svg
  const rect = svg.getBoundingClientRect();
  const stageRect = stageEl.getBoundingClientRect();
  // Convert SVG user units to client pixels
  const sx = ((cx - VB.x) / VB.w) * rect.width + (rect.left - stageRect.left);
  const sy = ((cy - VB.y) / VB.h) * rect.height + (rect.top - stageRect.top);
  tooltip.innerHTML = `<span class="tt-id">${entry.id || entry.label}</span><span class="tt-pos">(${entry.x.toFixed(1)}, ${entry.y.toFixed(1)})</span>`;
  tooltip.style.left = sx + "px";
  tooltip.style.top  = sy + "px";
  tooltip.classList.add("visible");
}
function hideTooltip() { tooltip.classList.remove("visible"); }

// ===========================================================================
// Sidebar: telemetry + sequence
// ===========================================================================
const telemetryEl = document.getElementById("telemetry");
function metricCard(label, value, unit, klass, id) {
  // The telemetry panel is optional. If the user removes it from the HTML
  // template, the viewer must still render and animate normally.
  if (!telemetryEl) return null;
  const div = document.createElement("div");
  div.className = "metric" + (klass ? " " + klass : "");
  div.innerHTML = `
    <div class="metric-label">${label}</div>
    <div class="metric-value"${id ? ` id="${id}"` : ""}>${value}${unit ? `<span class="metric-unit">${unit}</span>` : ""}</div>
  `;
  telemetryEl.appendChild(div);
  return div.querySelector(".metric-value");
}

const numAisles = new Set(DATA.edges.vertical.flatMap(e => [e[0][0], e[1][0]])).size;
const numRows   = new Set(DATA.edges.horizontal.flatMap(e => [e[0][1], e[1][1]])).size;
let distanceMetric = null;
let picksMetric = null;

function _routeDistances() {
  return (DATA.route && Array.isArray(DATA.route.frame_distance)) ? DATA.route.frame_distance : [];
}
function _routeTotalDistance() {
  return DATA.route ? Number(DATA.route.total_distance || 0) : 0;
}
function _frameDistance(frame) {
  const distances = _routeDistances();
  if (!distances.length) return 0;
  const safe = Math.max(0, Math.min(Math.floor(frame), distances.length - 1));
  return Number(distances[safe] || 0);
}
function _formatDistance(value) {
  const x = Number(value || 0);
  // One decimal is enough for non-integer coordinates, while integers remain clean.
  return Math.abs(x - Math.round(x)) < 1e-9 ? Math.round(x).toString() : x.toFixed(1);
}
function _frameForDistance(distance) {
  const distances = _routeDistances();
  if (!distances.length) return 0;
  const target = Math.max(0, Math.min(Number(distance || 0), _routeTotalDistance()));
  let lo = 0, hi = distances.length - 1;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (Number(distances[mid] || 0) < target) lo = mid + 1;
    else hi = mid;
  }
  // Choose the closest frame, not merely the first frame after the distance.
  if (lo > 0) {
    const prev = Math.abs(Number(distances[lo - 1] || 0) - target);
    const curr = Math.abs(Number(distances[lo] || 0) - target);
    if (prev <= curr) return lo - 1;
  }
  return lo;
}
function _distanceProgress(frame) {
  const total = _routeTotalDistance();
  if (total > 0) return Math.max(0, Math.min(1, _frameDistance(frame) / total));
  return totalFrames > 1 ? Math.max(0, Math.min(1, Math.floor(frame) / (totalFrames - 1))) : 0;
}
function updateClock(frame) {
  const total = _routeTotalDistance();
  const current = _frameDistance(frame);
  const now = document.getElementById("clk-now");
  const tot = document.getElementById("clk-total");
  if (now) now.textContent = _formatDistance(current);
  if (tot) tot.textContent = total > 0 ? `${_formatDistance(total)}u` : "0u";
}

if (DATA.route) {
  metricCard("STRATEGY", DATA.route.strategy || "—", "", "full");
  distanceMetric = metricCard("DISTANCE", `0 / ${_formatDistance(_routeTotalDistance())}`, "u", "", "metric-distance");
  picksMetric = metricCard("PICKS", `0 / ${DATA.route.n_picks}`, "", "", "metric-picks");
} else {
  metricCard("STATUS", "NO ROUTE", "", "full");
}


function updateTelemetryFrame(frame) {
  if (!DATA.route) return;
  const pickCounts = DATA.route.frame_picks || [];
  const safe = Math.max(0, Math.min(Math.floor(frame), Math.max(pickCounts.length - 1, 0)));
  const d = _frameDistance(frame);
  const p = pickCounts.length ? Number(pickCounts[safe] || 0) : 0;
  if (distanceMetric) {
    distanceMetric.innerHTML = `${_formatDistance(d)} / ${_formatDistance(_routeTotalDistance())}<span class="metric-unit">u</span>`;
  }
  if (picksMetric) {
    picksMetric.textContent = `${p} / ${DATA.route.n_picks}`;
  }
  updateClock(frame);
}

// sequence
const seqEl = document.getElementById("sequence");
let sequence = [];
if (DATA.route) {
  sequence = DATA.route.terminal_sequence.map((id, i) => ({ id, idx: i }));
}
// Build id -> {x,y,kinds} lookup
const idIndex = new Map();
for (const e of DATA.picks) {
  idIndex.set(e.id, { ...e, kind: "pick" });
  for (const alias of (e.aliases || [])) idIndex.set(alias, { ...e, kind: "pick" });
}
for (const e of DATA.sf) {
  idIndex.set(e.id, { ...e, kind: "sf" });
  for (const alias of (e.aliases || [])) idIndex.set(alias, { ...e, kind: "sf" });
}

for (const s of sequence) {
  const meta = idIndex.get(s.id) || { kind: /(START|FINISH|S\/F|^S$|^F$)/.test(s.id) ? "sf" : "pick" };
  const li = document.createElement("li");
  li.className = "seq-item";
  li.dataset.kind = meta.kind;
  li.dataset.idx = s.idx;
  li.innerHTML = `
    <div class="seq-dot">${(s.idx + 1).toString().padStart(2, "0")}</div>
    <div class="seq-id">${s.id}</div>
    <div class="seq-num">${meta.kind.toUpperCase()}</div>
  `;
  if (seqEl) seqEl.appendChild(li);
}

// ===========================================================================
// View switching
// ===========================================================================
const appEl = document.getElementById("app");
const viewLabel = document.getElementById("view-label");
const statusLabel = document.getElementById("status-label");
const boundsLabel = document.getElementById("bounds-label");
const nodesLabel  = document.getElementById("nodes-label");
if (boundsLabel) boundsLabel.textContent = `${VB.w.toFixed(1)} × ${VB.h.toFixed(1)}`;
if (nodesLabel)  nodesLabel.textContent  = (DATA.picks.length + DATA.sf.length + DATA.junctions.length).toString();

const navTabs = document.querySelectorAll(".nav-tab");
let currentView = INITIAL_VIEW;

function setView(view) {
  currentView = view;
  navTabs.forEach(b => b.classList.toggle("active", b.dataset.view === view));
  if (viewLabel) viewLabel.textContent = view.toUpperCase();
  if (view === "layout") {
    gRoute.style.display = "none";
    gTrail.style.display = "none";
    gPicker.style.display = "none";
    appEl.classList.add("no-timeline");
    if (statusLabel) statusLabel.textContent = "LAYOUT VIEW";
    pause();
  } else if (view === "route") {
    gRoute.style.display = "";
    gRoute.classList.remove("route-dim");
    gTrail.style.display = "none";
    gPicker.style.display = "none";
    appEl.classList.add("no-timeline");
    if (statusLabel) statusLabel.textContent = DATA.route ? "ROUTE LOCKED" : "NO ROUTE";
    pause();
  } else { // animation
    gRoute.style.display = DATA.route ? "" : "none";
    gRoute.classList.add("route-dim");
    gTrail.style.display = "";
    gPicker.style.display = DATA.route ? "" : "none";
    appEl.classList.remove("no-timeline");
    if (!DATA.route) {
      if (statusLabel) statusLabel.textContent = "NO ROUTE";
    } else {
      if (statusLabel) statusLabel.textContent = isPlaying ? "PICKING" : "READY";
    }
    seekTo(frameIdx);
  }
}
navTabs.forEach(b => b.addEventListener("click", () => setView(b.dataset.view)));

// ===========================================================================
// Animation engine
// ===========================================================================
const smooth = DATA.route ? DATA.route.smooth_path : [];
const totalFrames = smooth.length;
let frameIdx = 0;
let isPlaying = false;
let speed = 1;
let lastTime = 0;
const baseFPS = Math.max(24, totalFrames / 12);  // at 1×, finish typical route in about 12 seconds

function frameToTrailPath(i) {
  if (i <= 0 || smooth.length === 0) return "";
  let d = `M${smooth[0][0]} ${svgY(smooth[0][1])} `;
  for (let k = 1; k <= i && k < smooth.length; k++) {
    d += `L${smooth[k][0]} ${svgY(smooth[k][1])} `;
  }
  return d;
}

// Map smooth_path index -> nearest node_path index (for sequence highlight).
// We compare positions; the smooth path is a superset of node_path points.
let smoothToNode = [];
if (DATA.route) {
  const np = DATA.route.node_path;
  let cursor = 0;
  for (let i = 0; i < smooth.length; i++) {
    while (cursor + 1 < np.length) {
      const here = Math.abs(smooth[i][0] - np[cursor][0]) + Math.abs(smooth[i][1] - np[cursor][1]);
      const next = Math.abs(smooth[i][0] - np[cursor+1][0]) + Math.abs(smooth[i][1] - np[cursor+1][1]);
      if (next < here) cursor++; else break;
    }
    smoothToNode.push(cursor);
  }
}

// Map node_path index -> sequence index. The terminal_sequence has N+1 elements
// (S, picks..., F), and node_path visits these in order possibly with extra
// corridor steps in between. We tag each terminal_sequence entry with the
// first node_path index whose coordinates match.
const nodeToSeq = new Array(DATA.route ? DATA.route.node_path.length : 0).fill(0);
if (DATA.route) {
  const np = DATA.route.node_path;
  // Build (x,y) -> aliases map from picks + start/finish nodes.
  // Same physical depot can legitimately represent START and FINISH.
  const keyMap = new Map();
  const k = (e) => `${e.x.toFixed(4)}|${e.y.toFixed(4)}`;
  function addAliases(e) {
    const key = k(e);
    const ids = keyMap.get(key) || new Set();
    if (e.id) ids.add(e.id);
    for (const alias of (e.aliases || [])) ids.add(alias);
    keyMap.set(key, ids);
  }
  for (const e of DATA.picks) addAliases(e);
  for (const e of DATA.sf)    addAliases(e);
  let seqCursor = 0;
  for (let i = 0; i < np.length; i++) {
    const key = `${np[i][0].toFixed(4)}|${np[i][1].toFixed(4)}`;
    const aliases = keyMap.get(key);
    const expected = DATA.route.terminal_sequence[seqCursor];
    if (aliases && expected && aliases.has(expected)) {
      seqCursor++;
    }
    nodeToSeq[i] = Math.max(0, seqCursor - 1);
  }
}

const eventFrames = DATA.route ? (DATA.route.event_frames || []) : [];

function sequenceIndexAtFrame(frame) {
  if (!eventFrames.length) return 0;
  let seqIdx = 0;
  for (let i = 0; i < eventFrames.length; i++) {
    if (eventFrames[i] <= frame) seqIdx = i;
    else break;
  }
  return seqIdx;
}

function updateSequenceHighlight(seqIdx) {
  if (!seqEl) return;
  const items = seqEl.querySelectorAll(".seq-item");
  items.forEach((li, i) => {
    li.classList.toggle("current", i === seqIdx);
    li.classList.toggle("done", i < seqIdx);
  });
  if (items[seqIdx]) {
    items[seqIdx].scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

function renderFrame(idx) {
  if (!totalFrames) return;
  const safeIdx = Math.max(0, Math.min(Math.floor(idx), totalFrames - 1));
  const pt = smooth[safeIdx];
  placePicker(pt[0], pt[1]);
  const d = frameToTrailPath(safeIdx);
  trailCore.setAttribute("d", d);
  trailGlow.setAttribute("d", d);
  // scrub UI
  const pct = _distanceProgress(safeIdx);
  const scrubFill = document.getElementById("scrub-fill");
  if (scrubFill) scrubFill.style.width = (pct * 100) + "%";
  const handle = document.getElementById("scrub-handle");
  const track = document.querySelector(".scrub-track");
  if (track && handle) handle.style.left = (pct * track.clientWidth) + "px";
  // sequence + telemetry
  updateSequenceHighlight(sequenceIndexAtFrame(safeIdx));
  updateTelemetryFrame(safeIdx);
}

function seekTo(idx) {
  if (!totalFrames) return;
  frameIdx = Math.max(0, Math.min(Math.floor(idx), totalFrames - 1));
  renderFrame(frameIdx);
}

function tick(t) {
  if (!isPlaying) return;
  if (!lastTime) lastTime = t;
  const dt = (t - lastTime) / 1000;
  lastTime = t;
  const advance = dt * baseFPS * speed;
  let next = frameIdx + advance;
  if (next >= totalFrames - 1) {
    frameIdx = totalFrames - 1;
    renderFrame(frameIdx);
    pause();
    return;
  }
  frameIdx = next;
  renderFrame(frameIdx);
  requestAnimationFrame(tick);
}

const icPlay  = document.getElementById("ic-play");
const icPause = document.getElementById("ic-pause");
function play() {
  if (!totalFrames || isPlaying) return;
  if (currentView !== "animation") setView("animation");
  if (frameIdx >= totalFrames - 1) frameIdx = 0;
  isPlaying = true;
  lastTime = 0;
  if (icPlay)  icPlay.style.display  = "none";
  if (icPause) icPause.style.display = "";
  if (statusLabel) statusLabel.textContent = "PICKING";
  requestAnimationFrame(tick);
}
function pause() {
  isPlaying = false;
  if (icPlay)  icPlay.style.display  = "";
  if (icPause) icPause.style.display = "none";
  if (currentView === "animation" && statusLabel) statusLabel.textContent = "READY";
}

// Transport buttons
const btn_play = document.getElementById("btn-play");
if (btn_play) btn_play.addEventListener("click", () => isPlaying ? pause() : play());
const btn_restart = document.getElementById("btn-restart");
if (btn_restart) btn_restart.addEventListener("click", () => { pause(); seekTo(0); });
const btn_step_back = document.getElementById("btn-step-back");
if (btn_step_back) btn_step_back.addEventListener("click", () => { pause(); seekTo(Math.floor(frameIdx) - 1); });
const btn_step_fwd = document.getElementById("btn-step-fwd");
if (btn_step_fwd) btn_step_fwd.addEventListener("click", () => { pause(); seekTo(Math.floor(frameIdx) + 1); });

// Speed
document.querySelectorAll(".speed-btn").forEach(b => {
  b.addEventListener("click", () => {
    speed = parseFloat(b.dataset.speed);
    document.querySelectorAll(".speed-btn").forEach(x => x.classList.toggle("active", x === b));
  });
});

// Scrubber: click + drag
const scrubber = document.getElementById("scrubber");
let scrubbing = false;
function scrubFromEvent(e) {
  const scrubTrack = document.querySelector(".scrub-track");
  if (!scrubTrack) return;
  const rect = scrubTrack.getBoundingClientRect();
  const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
  const pct = Math.max(0, Math.min(1, x / rect.width));
  if (DATA.route && _routeTotalDistance() > 0) {
    seekTo(_frameForDistance(pct * _routeTotalDistance()));
  } else {
    seekTo(Math.round(pct * (totalFrames - 1)));
  }
}
if (scrubber) scrubber.addEventListener("mousedown", (e) => { pause(); scrubbing = true; scrubFromEvent(e); });
window.addEventListener("mousemove", (e) => { if (scrubbing) scrubFromEvent(e); });
window.addEventListener("mouseup",   () => { scrubbing = false; });
if (scrubber) scrubber.addEventListener("touchstart", (e) => { pause(); scrubbing = true; scrubFromEvent(e); }, { passive: true });
window.addEventListener("touchmove",  (e) => { if (scrubbing) scrubFromEvent(e); }, { passive: true });
window.addEventListener("touchend",   () => { scrubbing = false; });

// Sequence item click -> seek exactly to the event frame for that pick/depot.
if (seqEl) seqEl.addEventListener("click", (e) => {
  const li = e.target.closest(".seq-item");
  if (!li || !DATA.route) return;
  const targetSeq = parseInt(li.dataset.idx, 10);
  const targetFrame = eventFrames[targetSeq];
  if (targetFrame === undefined) return;
  pause();
  setView("animation");
  seekTo(targetFrame);
});

// Scrub ticks (one tiny mark per pick in the sequence)
function renderScrubTicks() {
  const ticks = document.getElementById("scrub-ticks");
  if (!ticks) return;
  ticks.innerHTML = "";
  if (!DATA.route || !eventFrames.length) return;
  const scrubTrack = document.querySelector(".scrub-track");
  if (!scrubTrack) return;
  const trackW = scrubTrack.clientWidth;
  for (let sIdx = 0; sIdx < DATA.route.terminal_sequence.length; sIdx++) {
    const id = DATA.route.terminal_sequence[sIdx];
    const meta = idIndex.get(id);
    if (!id || !meta || meta.kind !== "pick") continue;
    const frame = eventFrames[sIdx];
    if (frame === undefined) continue;
    const eventDistances = DATA.route.event_distance || [];
    const eventDistance = Number(eventDistances[sIdx] || 0);
    const pct = _routeTotalDistance() > 0 ? Math.max(0, Math.min(1, eventDistance / _routeTotalDistance())) : (totalFrames > 1 ? frame / (totalFrames - 1) : 0);
    const t = document.createElement("div");
    t.className = "scrub-tick";
    t.style.left = (pct * trackW) + "px";
    ticks.appendChild(t);
  }
}

// Keyboard shortcuts
document.addEventListener("keydown", (e) => {
  if (e.target.matches("input, textarea")) return;
  switch (e.key) {
    case " ":
      e.preventDefault();
      if (currentView !== "animation") setView("animation");
      isPlaying ? pause() : play();
      break;
    case "0":
      pause(); seekTo(0); break;
    case "ArrowLeft":
      pause(); seekTo(Math.floor(frameIdx) - 1); break;
    case "ArrowRight":
      pause(); seekTo(Math.floor(frameIdx) + 1); break;
    case "1": setView("layout"); break;
    case "2": setView("route"); break;
    case "3": setView("animation"); break;
    case "f": case "F": toggleFullscreen(); break;
  }
});

// Fullscreen
function toggleFullscreen() {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen?.();
  } else {
    document.exitFullscreen?.();
  }
}
const btn_fullscreen = document.getElementById("btn-fullscreen");
if (btn_fullscreen) btn_fullscreen.addEventListener("click", toggleFullscreen);
const btn_fit = document.getElementById("btn-fit");
if (btn_fit) btn_fit.addEventListener("click", () => {
  // Re-apply the viewBox just in case anything has shifted.
  svg.setAttribute("viewBox", `${VB.x} ${VB.y} ${VB.w} ${VB.h}`);
});

// Resize: re-place scrub handle and re-render ticks
window.addEventListener("resize", () => {
  seekTo(Math.floor(frameIdx));
  renderScrubTicks();
});

// ===========================================================================
// Init
// ===========================================================================
if (DATA.route) {
  updateClock(0);
}
setView(INITIAL_VIEW);
// Defer until layout settles, then set up ticks + handle position.
requestAnimationFrame(() => {
  renderScrubTicks();
  seekTo(0);
});
</script>
</body>
</html>
"""
