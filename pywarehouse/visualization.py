"""Plotly visualization for pyWarehouse.

A single, polished "premium" dark theme is provided for both static layouts
and animated routes. Figure dimensions are computed from the data aspect
ratio so warehouses of any shape render without distortion or weird
whitespace. The animation interpolates between corridor nodes so the
picker glides smoothly along long edges instead of jumping.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import plotly.graph_objects as go

from .graph import undirected_key
from .models import Graph, Point, Route, Segment


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

PALETTE: Dict[str, str] = {
    # Background
    "paper":          "#0A0F1E",  # deep navy, slightly cooler than pure black
    "plot":           "#0A0F1E",
    # Grid / aisles
    "aisle_v":        "#2A3458",  # vertical picking aisles
    "aisle_h":        "#3A4775",  # cross / backbone aisles (brighter)
    "junction":       "rgba(180,200,235,0.18)",
    # Route (electric cyan)
    "route_glow":     "rgba(96,205,255,0.16)",
    "route_mid":      "rgba(96,205,255,0.55)",
    "route_core":     "#60CDFF",
    # Nodes
    "pick_fill":      "#FF6B6B",
    "pick_halo":      "rgba(255,107,107,0.18)",
    "pick_edge":      "#1A1F2E",
    "sf_fill":        "#22D3A6",
    "sf_halo":        "rgba(34,211,166,0.22)",
    "sf_edge":        "#0F1525",
    # Animation
    "picker_fill":    "#FFD166",  # warm amber so it stands out against cyan trail
    "picker_edge":    "#1A1F2E",
    "trail":          "#FFD166",
    "trail_glow":     "rgba(255,209,102,0.25)",
    # Text
    "text":           "#FFFFFF",
    "text_muted":     "#9CA8C7",
    "title":          "#FFFFFF",
    # Legend
    "legend_bg":      "rgba(10,15,30,0.55)",
    "legend_border":  "rgba(180,200,235,0.18)",
}

FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def make_item_labels(
    node_attrs: Dict[Point, Dict[str, Any]],
    mode: str = "id",
    prefix: str = "P",
    pad: int = 3,
) -> Dict[Point, str]:
    """Create labels for terminal nodes.

    ``mode='id'`` uses the product/start/finish ids. ``mode='p001'`` produces
    sequential labels ``P001``, ``P002``, .... ``mode='letters'`` uses ``A``,
    ``B``, ``C``, ....
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


# ---------------------------------------------------------------------------
# Edge helpers
# ---------------------------------------------------------------------------

def _xy_from_edges(edges: Iterable[Tuple[Point, Point]]) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for u, v in edges:
        xs.extend([u[0], v[0], None])
        ys.extend([u[1], v[1], None])
    return xs, ys


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


def _edges_from_path(node_path: Sequence[Point]) -> List[Tuple[Point, Point]]:
    return [(node_path[i - 1], node_path[i]) for i in range(1, len(node_path))]


def _edges_from_segments(segments: Sequence[Segment]) -> List[Tuple[Point, Point]]:
    edges: List[Tuple[Point, Point]] = []
    for seg in segments:
        for i in range(1, len(seg.nodes)):
            edges.append((seg.nodes[i - 1], seg.nodes[i]))
    return edges


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _data_extent(
    G: Graph,
    node_attrs: Dict[Point, Dict[str, Any]],
) -> Tuple[float, float, float, float]:
    """Return ``(xmin, xmax, ymin, ymax)`` over every node in the graph."""
    xs = [p[0] for p in G.keys()]
    ys = [p[1] for p in G.keys()]
    # Include any terminals that might live on stubs already covered, but be safe.
    for p in node_attrs.keys():
        xs.append(p[0])
        ys.append(p[1])
    return min(xs), max(xs), min(ys), max(ys)


def _auto_figure_size(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    *,
    base: int = 780,
    min_w: int = 420,
    min_h: int = 380,
    max_w: int = 1400,
    max_h: int = 1100,
    chrome_w: int = 60,        # left+right margins
    chrome_h_static: int = 110,  # title + subtitle + bottom margin (static figure)
    chrome_h_anim: int = 180,    # title + slider + buttons (animated figure)
    animated: bool = False,
) -> Tuple[int, int]:
    """Pick a figure (width, height) that respects the data aspect ratio.

    With ``scaleanchor="x"`` Plotly keeps a 1:1 data ratio inside the plot
    area, so a 15-wide × 100-tall warehouse jammed into a 1000×720 canvas
    becomes a thin vertical sliver flanked by huge empty bands. Picking
    canvas dimensions that mirror the data ratio fixes that without
    distorting geometry.
    """
    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)
    aspect = dy / dx  # > 1 → tall; < 1 → wide

    chrome_h = chrome_h_anim if animated else chrome_h_static

    if aspect >= 1.0:
        plot_h = float(base)
        plot_w = plot_h / aspect
    else:
        plot_w = float(base)
        plot_h = plot_w * aspect

    w = int(round(plot_w + chrome_w))
    h = int(round(plot_h + chrome_h))
    w = max(min_w, min(max_w, w))
    h = max(min_h, min(max_h, h))
    return w, h


def _interpolate_for_animation(
    path: Sequence[Point],
    target_step: float = 1.0,
    max_frames: int = 240,
) -> List[Point]:
    """Subdivide long edges so the picker moves at a roughly constant pace.

    Manhattan corridor edges vary a lot in length (one slot ≈ 1 unit, a cross
    aisle ≈ many units). One frame per node makes long traversals look choppy
    and short ones look frantic. We split any edge longer than ``target_step``
    into evenly spaced sub-points, capping the total so the animation stays
    snappy. The returned list always starts at ``path[0]`` and ends at
    ``path[-1]``; the original corner nodes are preserved.
    """
    if not path:
        return []
    if len(path) == 1:
        return [tuple(path[0])]  # type: ignore[list-item]

    # First pass: subdivide based on edge length.
    out: List[Point] = [tuple(path[0])]  # type: ignore[list-item]
    for i in range(1, len(path)):
        u = path[i - 1]
        v = path[i]
        dx = v[0] - u[0]
        dy = v[1] - u[1]
        # Edges in this graph are axis-aligned, so this is exact.
        edge_len = abs(dx) + abs(dy)
        if edge_len <= target_step + 1e-9:
            out.append(tuple(v))  # type: ignore[arg-type]
            continue
        n_steps = max(1, int(round(edge_len / target_step)))
        for s in range(1, n_steps + 1):
            t = s / n_steps
            out.append((u[0] + t * dx, u[1] + t * dy))

    # Second pass: cap total frames by uniform downsampling (always keep first/last).
    if len(out) > max_frames:
        step = (len(out) - 1) / (max_frames - 1)
        sampled: List[Point] = []
        for i in range(max_frames):
            idx = int(round(i * step))
            idx = min(idx, len(out) - 1)
            sampled.append(out[idx])
        out = sampled

    return out


def _route_summary(route: Route) -> str:
    """Build a small subtitle string with strategy + key metrics."""
    parts: List[str] = []
    strat = getattr(route, "strategy", None)
    if strat:
        parts.append(f"<b>Strategy</b> {strat}")
    dist = getattr(route, "total_distance", None)
    if dist is not None:
        parts.append(f"<b>Distance</b> {dist:.1f}")
    npicks = 0
    seq = getattr(route, "terminal_sequence", None) or []
    if seq:
        npicks = sum(1 for s in seq if str(s).upper() not in {"START", "FINISH", "S", "F", "S/F"})
        if npicks:
            parts.append(f"<b>Picks</b> {npicks}")
    return "  ·  ".join(parts)


# ---------------------------------------------------------------------------
# Plotter
# ---------------------------------------------------------------------------

class Plotter:
    """Premium Plotly drawing for warehouse graphs and routes."""

    def __init__(self, G: Graph, node_attrs: Dict[Point, Dict[str, Any]]):
        self.G = G
        self.node_attrs = node_attrs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw_layout(
        self,
        *,
        label_mode: str = "id",
        width: Optional[int] = None,
        height: Optional[int] = None,
        title: str = "Warehouse Layout",
    ) -> go.Figure:
        """Render the empty warehouse graph (no route highlighted)."""
        return self.draw_route(
            None,
            label_mode=label_mode,
            width=width,
            height=height,
            title=title,
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
    ) -> go.Figure:
        """Render the warehouse with an optional highlighted route."""
        subtitle = ""
        if route is not None:
            node_path = route.node_path
            segments = route.segments
            if title == "Warehouse Route":
                title = "Warehouse Route"
            subtitle = _route_summary(route)

        traces = self._base_traces()
        traces.extend(self._route_traces(node_path, segments))
        traces.extend(self._node_traces(label_mode))

        fig = go.Figure(data=traces)
        self._apply_layout(
            fig,
            width=width,
            height=height,
            title=title,
            subtitle=subtitle,
            show_legend=show_legend,
            animated=False,
        )
        return fig

    def animate_route(
        self,
        route: Route,
        *,
        label_mode: str = "id",
        width: Optional[int] = None,
        height: Optional[int] = None,
        title: str = "Warehouse Route Animation",
        target_step: float = 1.0,
        max_frames: int = 240,
        frame_duration: int = 55,
        transition_duration: int = 40,
    ) -> go.Figure:
        """Animate a picker travelling along ``route``.

        The path is interpolated so motion is smooth across long edges; a
        scrub slider, speed buttons, restart, play and pause are all wired up.
        """
        path = list(route.node_path or [])
        # Build the static figure first so all base traces + node markers are present.
        fig = self.draw_route(
            None,
            label_mode=label_mode,
            width=width,
            height=height,
            title=title,
            show_legend=True,
        )
        # Stamp subtitle from the route even though we didn't pass it as the route.
        subtitle = _route_summary(route)
        if subtitle:
            self._stamp_subtitle(fig, subtitle)

        if not path:
            return fig

        # Re-layout for the animated chrome (slider + buttons need extra bottom).
        # Only resize if the caller didn't pin width/height.
        if width is None or height is None:
            xmin, xmax, ymin, ymax = _data_extent(self.G, self.node_attrs)
            w, h = _auto_figure_size(xmin, xmax, ymin, ymax, animated=True)
            fig.update_layout(
                width=width if width is not None else w,
                height=height if height is not None else h,
            )

        # Interpolated path drives the frames.
        smooth = _interpolate_for_animation(path, target_step=target_step, max_frames=max_frames)
        xs = [p[0] for p in smooth]
        ys = [p[1] for p in smooth]

        # Add the static (route) trail glow + trail + picker, on top of everything.
        trail_glow = go.Scatter(
            x=[xs[0]], y=[ys[0]], mode="lines",
            line=dict(width=12, color=PALETTE["trail_glow"]),
            hoverinfo="skip", showlegend=False,
        )
        trail = go.Scatter(
            x=[xs[0]], y=[ys[0]], mode="lines",
            line=dict(width=4, color=PALETTE["trail"]),
            hoverinfo="skip",
            name="Picker trail",
        )
        picker_halo = go.Scatter(
            x=[xs[0]], y=[ys[0]], mode="markers",
            marker=dict(size=34, color=PALETTE["trail_glow"], line=dict(width=0)),
            hoverinfo="skip", showlegend=False,
        )
        picker = go.Scatter(
            x=[xs[0]], y=[ys[0]], mode="markers",
            marker=dict(
                size=18,
                color=PALETTE["picker_fill"],
                line=dict(width=2, color=PALETTE["picker_edge"]),
                symbol="circle",
            ),
            name="Picker",
        )
        fig.add_traces([trail_glow, trail, picker_halo, picker])

        n_total = len(fig.data)
        idx_trail_glow = n_total - 4
        idx_trail = n_total - 3
        idx_picker_halo = n_total - 2
        idx_picker = n_total - 1
        anim_trace_ids = [idx_trail_glow, idx_trail, idx_picker_halo, idx_picker]

        # Build frames. Each frame only updates x/y of the four animation traces.
        frames: List[go.Frame] = []
        slider_steps: List[Dict[str, Any]] = []
        for i in range(len(xs)):
            x_so_far = xs[: i + 1]
            y_so_far = ys[: i + 1]
            frames.append(
                go.Frame(
                    name=str(i),
                    data=[
                        go.Scatter(x=x_so_far, y=y_so_far),  # trail glow
                        go.Scatter(x=x_so_far, y=y_so_far),  # trail
                        go.Scatter(x=[xs[i]], y=[ys[i]]),    # picker halo
                        go.Scatter(x=[xs[i]], y=[ys[i]]),    # picker
                    ],
                    traces=anim_trace_ids,
                )
            )
            slider_steps.append({
                "method": "animate",
                "label": "",
                "args": [
                    [str(i)],
                    {
                        "mode": "immediate",
                        "frame": {"duration": 0, "redraw": False},
                        "transition": {"duration": 0},
                    },
                ],
            })

        fig.frames = frames

        # Controls: play / pause / restart, plus 1× / 2× / 0.5× speed.
        def _play_args(duration: int) -> List[Any]:
            return [
                None,
                {
                    "frame": {"duration": duration, "redraw": False},
                    "fromcurrent": True,
                    "transition": {"duration": transition_duration, "easing": "linear"},
                    "mode": "immediate",
                },
            ]

        play_button = {
            "label": "▶  Play",
            "method": "animate",
            "args": _play_args(frame_duration),
        }
        pause_button = {
            "label": "❚❚  Pause",
            "method": "animate",
            "args": [[None], {"frame": {"duration": 0, "redraw": False},
                              "mode": "immediate",
                              "transition": {"duration": 0}}],
        }
        restart_button = {
            "label": "↻  Restart",
            "method": "animate",
            "args": [
                None,
                {
                    "frame": {"duration": frame_duration, "redraw": False},
                    "fromcurrent": False,
                    "transition": {"duration": 0},
                    "mode": "immediate",
                },
            ],
        }
        slow_button = {
            "label": "½×",
            "method": "animate",
            "args": _play_args(frame_duration * 2),
        }
        normal_button = {
            "label": "1×",
            "method": "animate",
            "args": _play_args(frame_duration),
        }
        fast_button = {
            "label": "2×",
            "method": "animate",
            "args": _play_args(max(15, frame_duration // 2)),
        }

        fig.update_layout(
            updatemenus=[
                {
                    "type": "buttons",
                    "direction": "right",
                    "showactive": False,
                    "x": 0.0, "xanchor": "left",
                    "y": -0.06, "yanchor": "top",
                    "pad": {"l": 6, "r": 6, "t": 6, "b": 6},
                    "bgcolor": "rgba(96,205,255,0.10)",
                    "bordercolor": "rgba(96,205,255,0.45)",
                    "borderwidth": 1,
                    "font": {"color": PALETTE["text"], "family": FONT_FAMILY, "size": 12},
                    "buttons": [play_button, pause_button, restart_button],
                },
                {
                    "type": "buttons",
                    "direction": "right",
                    "showactive": False,
                    "x": 1.0, "xanchor": "right",
                    "y": -0.06, "yanchor": "top",
                    "pad": {"l": 6, "r": 6, "t": 6, "b": 6},
                    "bgcolor": "rgba(180,200,235,0.08)",
                    "bordercolor": "rgba(180,200,235,0.30)",
                    "borderwidth": 1,
                    "font": {"color": PALETTE["text_muted"], "family": FONT_FAMILY, "size": 11},
                    "buttons": [slow_button, normal_button, fast_button],
                },
            ],
            sliders=[
                {
                    "active": 0,
                    "steps": slider_steps,
                    "x": 0.0, "xanchor": "left",
                    "y": -0.16, "yanchor": "top",
                    "len": 1.0,
                    "pad": {"l": 0, "r": 0, "t": 10, "b": 0},
                    "bgcolor": "rgba(180,200,235,0.18)",
                    "bordercolor": "rgba(180,200,235,0.0)",
                    "tickcolor": "rgba(180,200,235,0.30)",
                    "font": {"color": PALETTE["text_muted"], "family": FONT_FAMILY, "size": 10},
                    "currentvalue": {
                        "visible": True,
                        "prefix": "Step ",
                        "xanchor": "left",
                        "font": {"color": PALETTE["text"], "family": FONT_FAMILY, "size": 12},
                    },
                    "transition": {"duration": 0},
                }
            ],
        )

        return fig

    # ------------------------------------------------------------------
    # Internal building blocks
    # ------------------------------------------------------------------

    def _base_traces(self) -> List[go.Scatter]:
        """Aisle and junction layer (drawn first, lowest z)."""
        v_edges: List[Tuple[Point, Point]] = []
        h_edges: List[Tuple[Point, Point]] = []
        for u, v in _all_edges(self.G):
            if abs(u[0] - v[0]) < 1e-9:
                v_edges.append((u, v))
            else:
                h_edges.append((u, v))
        vx, vy = _xy_from_edges(v_edges)
        hx, hy = _xy_from_edges(h_edges)

        # Junction dots, very subtle so they only register as a faint texture.
        jx: List[float] = []
        jy: List[float] = []
        for p, attrs in self.node_attrs.items():
            if attrs.get("type") == "steiner":
                jx.append(p[0])
                jy.append(p[1])

        return [
            go.Scatter(
                x=vx, y=vy, mode="lines",
                line=dict(width=4.5, color=PALETTE["aisle_v"], shape="linear"),
                opacity=0.85, hoverinfo="skip",
                name="Aisles", showlegend=False,
            ),
            go.Scatter(
                x=hx, y=hy, mode="lines",
                line=dict(width=6.5, color=PALETTE["aisle_h"], shape="linear"),
                opacity=0.9, hoverinfo="skip",
                name="Cross aisles", showlegend=False,
            ),
            go.Scatter(
                x=jx, y=jy, mode="markers",
                marker=dict(size=3.5, color=PALETTE["junction"]),
                hoverinfo="skip", showlegend=False,
                name="Junctions",
            ),
        ]

    def _route_traces(
        self,
        node_path: Optional[Sequence[Point]],
        segments: Optional[Sequence[Segment]],
    ) -> List[go.Scatter]:
        """Multi-layer cyan glow representing the solved route (if any)."""
        edges: List[Tuple[Point, Point]] = []
        if node_path is not None:
            edges = _edges_from_path(node_path)
        elif segments is not None:
            edges = _edges_from_segments(segments)
        if not edges:
            return []

        rx, ry = _xy_from_edges(edges)
        return [
            go.Scatter(
                x=rx, y=ry, mode="lines",
                line=dict(width=18, color=PALETTE["route_glow"]),
                hoverinfo="skip", showlegend=False,
            ),
            go.Scatter(
                x=rx, y=ry, mode="lines",
                line=dict(width=10, color=PALETTE["route_mid"]),
                hoverinfo="skip", showlegend=False,
            ),
            go.Scatter(
                x=rx, y=ry, mode="lines",
                line=dict(width=4, color=PALETTE["route_core"]),
                hoverinfo="skip",
                name="Solution route",
            ),
        ]

    def _node_traces(self, label_mode: str) -> List[go.Scatter]:
        """Pick items and start/finish nodes with text labels."""
        labels = make_item_labels(self.node_attrs, mode=label_mode)
        picks_x: List[float] = []
        picks_y: List[float] = []
        picks_txt: List[str] = []
        picks_hover: List[str] = []
        sf_x: List[float] = []
        sf_y: List[float] = []
        sf_txt: List[str] = []
        sf_hover: List[str] = []

        for p, attrs in self.node_attrs.items():
            if attrs.get("type") != "terminal":
                continue
            kinds = attrs.get("kinds", [])
            label = labels.get(p, attrs.get("id", ""))
            if "pick" in kinds:
                picks_x.append(p[0]); picks_y.append(p[1]); picks_txt.append(label)
                picks_hover.append(
                    f"<b>{attrs.get('id', label)}</b><br>({p[0]:g}, {p[1]:g})"
                )
            else:
                sf_x.append(p[0]); sf_y.append(p[1]); sf_txt.append(label)
                sf_hover.append(
                    f"<b>{attrs.get('label', label)}</b><br>({p[0]:g}, {p[1]:g})"
                )

        out: List[go.Scatter] = []
        if picks_x:
            out.append(
                go.Scatter(
                    x=picks_x, y=picks_y, mode="markers",
                    marker=dict(size=36, color=PALETTE["pick_halo"], line=dict(width=0)),
                    hoverinfo="skip", showlegend=False,
                )
            )
            out.append(
                go.Scatter(
                    x=picks_x, y=picks_y, mode="markers+text",
                    text=picks_txt,
                    textposition="middle center",
                    textfont=dict(color=PALETTE["text"], size=11, family=FONT_FAMILY),
                    marker=dict(
                        size=24,
                        color=PALETTE["pick_fill"],
                        line=dict(width=1.5, color=PALETTE["pick_edge"]),
                    ),
                    name="Items",
                    hovertext=picks_hover, hoverinfo="text",
                )
            )
        if sf_x:
            out.append(
                go.Scatter(
                    x=sf_x, y=sf_y, mode="markers",
                    marker=dict(size=40, color=PALETTE["sf_halo"], line=dict(width=0)),
                    hoverinfo="skip", showlegend=False,
                )
            )
            out.append(
                go.Scatter(
                    x=sf_x, y=sf_y, mode="markers+text",
                    text=sf_txt,
                    textposition="middle center",
                    textfont=dict(color=PALETTE["text"], size=11, family=FONT_FAMILY),
                    marker=dict(
                        size=28,
                        color=PALETTE["sf_fill"],
                        symbol="square",
                        line=dict(width=1.5, color=PALETTE["sf_edge"]),
                    ),
                    name="Start / Finish",
                    hovertext=sf_hover, hoverinfo="text",
                )
            )
        return out

    # ------------------------------------------------------------------
    # Layout / chrome
    # ------------------------------------------------------------------

    def _apply_layout(
        self,
        fig: go.Figure,
        *,
        width: Optional[int],
        height: Optional[int],
        title: str,
        subtitle: str,
        show_legend: bool,
        animated: bool,
    ) -> None:
        xmin, xmax, ymin, ymax = _data_extent(self.G, self.node_attrs)

        if width is None or height is None:
            auto_w, auto_h = _auto_figure_size(xmin, xmax, ymin, ymax, animated=animated)
            if width is None:
                width = auto_w
            if height is None:
                height = auto_h

        # Pad ranges slightly so halos / markers don't get clipped at the edges.
        dx = max(xmax - xmin, 1e-6)
        dy = max(ymax - ymin, 1e-6)
        pad_x = dx * 0.04 + 0.5
        pad_y = dy * 0.03 + 0.5
        x_range = [xmin - pad_x, xmax + pad_x]
        y_range = [ymin - pad_y, ymax + pad_y]

        title_kwargs: Dict[str, Any] = dict(
            text=f"<b>{title}</b>",
            x=0.02, xanchor="left",
            y=0.98, yanchor="top",
            font=dict(size=18, color=PALETTE["title"], family=FONT_FAMILY),
        )
        if subtitle:
            title_kwargs["subtitle"] = dict(
                text=subtitle,
                font=dict(size=12, color=PALETTE["text_muted"], family=FONT_FAMILY),
            )

        # Top margin reserves room for title + native subtitle (rendered together).
        # Legend is placed INSIDE the plot at the top-right so the title band
        # stays free of competing text on narrow figures.
        top_margin = 78 if subtitle else 58
        bottom_margin = 92 if animated else 28

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=PALETTE["paper"],
            plot_bgcolor=PALETTE["plot"],
            width=width,
            height=height,
            font=dict(family=FONT_FAMILY, color=PALETTE["text"]),
            title=title_kwargs,
            margin=dict(l=24, r=24, t=top_margin, b=bottom_margin),
            showlegend=show_legend,
            legend=dict(
                orientation="h",
                yanchor="top", y=0.985,
                xanchor="right", x=0.985,
                bgcolor=PALETTE["legend_bg"],
                bordercolor=PALETTE["legend_border"],
                borderwidth=1,
                font=dict(color=PALETTE["text"], family=FONT_FAMILY, size=11),
                itemsizing="constant",
            ),
            xaxis=dict(
                visible=False, showgrid=False, zeroline=False,
                range=x_range, constrain="domain",
            ),
            yaxis=dict(
                visible=False, showgrid=False, zeroline=False,
                range=y_range,
                scaleanchor="x", scaleratio=1.0,
            ),
            hoverlabel=dict(
                bgcolor="rgba(20,28,50,0.95)",
                bordercolor="rgba(96,205,255,0.5)",
                font=dict(color=PALETTE["text"], family=FONT_FAMILY, size=12),
            ),
        )

    @staticmethod
    def _stamp_subtitle(fig: go.Figure, subtitle: str) -> None:
        """Apply a subtitle line to an already-laid-out figure."""
        if not subtitle:
            return
        fig.update_layout(
            title=dict(
                subtitle=dict(
                    text=subtitle,
                    font=dict(size=12, color=PALETTE["text_muted"], family=FONT_FAMILY),
                )
            )
        )
