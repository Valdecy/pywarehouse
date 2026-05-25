"""Graph utilities for rectilinear warehouse corridor graphs."""

from __future__ import annotations

import heapq
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .models import Graph, Movement, Point, Segment

Dir = Tuple[int, int]

DIRECTIONS: Dict[str, Dir] = {
    "RIGHT": (1, 0),
    "LEFT": (-1, 0),
    "UP": (0, 1),
    "DOWN": (0, -1),
}


def snap_value(value: float, pool: Sequence[float], tol: float = 1e-9) -> float:
    """Snap a coordinate to an existing value when it is numerically close."""
    for p in pool:
        if abs(value - p) <= tol:
            return float(p)
    return float(value)


def unique_points(points: Iterable[Point], tol: float = 1e-9) -> List[Point]:
    """Merge near-duplicate points."""
    uniq: List[Point] = []
    for x, y in points:
        xs = [ux for ux, _ in uniq]
        ys = [uy for _, uy in uniq]
        p = (snap_value(x, xs, tol), snap_value(y, ys, tol))
        if p not in uniq:
            uniq.append(p)
    return uniq


def rectilinear_distance(u: Point, v: Point) -> float:
    """Manhattan distance between two points."""
    return abs(u[0] - v[0]) + abs(u[1] - v[1])


def edge_weight(G: Graph, u: Point, v: Point) -> float:
    """Return an undirected edge weight."""
    if u in G and v in G[u]:
        return G[u][v]
    if v in G and u in G[v]:
        return G[v][u]
    raise KeyError(f"No edge between {u} and {v}")


def add_undirected_edge(G: Graph, u: Point, v: Point, weight: Optional[float] = None) -> None:
    """Add an undirected weighted edge."""
    if weight is None:
        weight = rectilinear_distance(u, v)
    if weight <= 0:
        return
    G.setdefault(u, {})[v] = float(weight)
    G.setdefault(v, {})[u] = float(weight)


def wire_line(G: Graph, nodes_on_line: Sequence[Point]) -> None:
    """Wire consecutive nodes on the same vertical/horizontal line."""
    for i in range(len(nodes_on_line) - 1):
        add_undirected_edge(G, nodes_on_line[i], nodes_on_line[i + 1])


def direction(u: Point, v: Point, tol: float = 1e-9) -> Dir:
    """Direction from u to v on a rectilinear edge."""
    dx = 0 if abs(v[0] - u[0]) <= tol else (1 if v[0] > u[0] else -1)
    dy = 0 if abs(v[1] - u[1]) <= tol else (1 if v[1] > u[1] else -1)
    if dx != 0 and dy != 0:
        raise ValueError(f"Diagonal step detected: {u} -> {v}")
    return dx, dy


def direction_name(d: Dir) -> str:
    """Convert direction tuple to movement name."""
    if d == (1, 0):
        return "RIGHT"
    if d == (-1, 0):
        return "LEFT"
    if d == (0, 1):
        return "UP"
    if d == (0, -1):
        return "DOWN"
    return "STAY"


def neighbor_in_direction(G: Graph, u: Point, dxy: Dir) -> Optional[Point]:
    """Return the neighbor of u in a given direction, if present."""
    for v in G.get(u, {}):
        if direction(u, v) == dxy:
            return v
    return None


def undirected_key(u: Point, v: Point) -> Tuple[Point, Point]:
    """Stable key for an undirected edge."""
    return (u, v) if u <= v else (v, u)


def graph_to_edges_df(G: Graph) -> pd.DataFrame:
    """Convert adjacency dict to an edge-list DataFrame."""
    rows = []
    seen = set()
    for u, nbrs in G.items():
        for v, w in nbrs.items():
            k = undirected_key(u, v)
            if k in seen:
                continue
            seen.add(k)
            rows.append({"x0": u[0], "y0": u[1], "x1": v[0], "y1": v[1], "weight": w})
    if not rows:
        return pd.DataFrame(columns=["x0", "y0", "x1", "y1", "weight"])
    return pd.DataFrame(rows).sort_values(["y0", "x0", "y1", "x1"]).reset_index(drop=True)


def shortest_path(
    G: Graph,
    source: Point,
    target: Point,
    is_edge_allowed: Optional[Callable[[Point, Point, float], bool]] = None,
    is_dir_allowed_at: Optional[Callable[[Point, Dir], bool]] = None,
) -> List[Point]:
    """Dijkstra shortest path with optional edge/direction constraints."""
    if source == target:
        return [source]
    dist = {source: 0.0}
    prev: Dict[Point, Optional[Point]] = {source: None}
    pq: List[Tuple[float, Point]] = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == target:
            break
        if d != dist.get(u, math.inf):
            continue
        for v, w in G.get(u, {}).items():
            if is_edge_allowed and not is_edge_allowed(u, v, w):
                continue
            if is_dir_allowed_at and not is_dir_allowed_at(u, direction(u, v)):
                continue
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if target not in prev:
        raise ValueError(f"No feasible path from {source} to {target}")
    path: List[Point] = []
    cur: Optional[Point] = target
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    return list(reversed(path))



def floyd_warshall_all_pairs(
    G: Graph,
) -> Tuple[Dict[Point, Dict[Point, float]], Dict[Point, Dict[Point, Optional[Point]]]]:
    """Compute all-pairs shortest distances and next hops with Floyd-Warshall.

    The distance matrix is computed on the *walkable corridor graph*, not by
    drawing direct Euclidean/Manhattan shortcuts between terminals. Edge weights
    already contain the proper physical distances, including aisle spacing, slot
    pitch, cross-corridor widths, and start/finish stubs.

    Returns
    -------
    dist:
        ``dist[u][v]`` is the shortest walkable distance from node ``u`` to
        node ``v``.
    next_hop:
        ``next_hop[u][v]`` is the first node after ``u`` on a shortest path to
        ``v``. Use :func:`reconstruct_fw_path` to recover the full path.
    """
    nodes = list(G.keys())
    dist: Dict[Point, Dict[Point, float]] = {u: {v: math.inf for v in nodes} for u in nodes}
    next_hop: Dict[Point, Dict[Point, Optional[Point]]] = {u: {v: None for v in nodes} for u in nodes}

    for u in nodes:
        dist[u][u] = 0.0
        next_hop[u][u] = u
        for v, w in G.get(u, {}).items():
            if w < dist[u].get(v, math.inf):
                dist[u][v] = float(w)
                next_hop[u][v] = v

    # Floyd-Warshall dynamic programming. The explicit loops are intentional:
    # typical warehouse examples have modest node counts, and the resulting
    # matrix makes every route leg use one consistent distance model.
    for k in nodes:
        dk = dist[k]
        for i in nodes:
            dik = dist[i][k]
            if dik == math.inf:
                continue
            di = dist[i]
            ni = next_hop[i]
            for j in nodes:
                alt = dik + dk[j]
                if alt + 1e-12 < di[j]:
                    di[j] = alt
                    ni[j] = ni[k]

    return dist, next_hop


def reconstruct_fw_path(
    next_hop: Dict[Point, Dict[Point, Optional[Point]]],
    source: Point,
    target: Point,
) -> List[Point]:
    """Reconstruct a shortest path from a Floyd-Warshall next-hop matrix."""
    if source == target:
        return [source]
    if source not in next_hop or target not in next_hop[source] or next_hop[source][target] is None:
        raise ValueError(f"No feasible path from {source} to {target}")
    path = [source]
    cur = source
    # Bound path reconstruction to avoid infinite loops if a user mutates the
    # matrix externally.
    max_steps = len(next_hop) + 1
    for _ in range(max_steps):
        nxt = next_hop[cur][target]
        if nxt is None:
            raise ValueError(f"No feasible path from {source} to {target}")
        cur = nxt
        path.append(cur)
        if cur == target:
            return path
    raise ValueError(f"Cycle detected while reconstructing path from {source} to {target}")

def path_length(G: Graph, node_path: Sequence[Point]) -> float:
    """Total distance of a node path."""
    return sum(edge_weight(G, node_path[i - 1], node_path[i]) for i in range(1, len(node_path)))


def path_to_segments(G: Graph, node_attrs: Dict[Point, Dict[str, Any]], node_path: Sequence[Point]) -> List[Segment]:
    """Compress a node path into straight segments split at turns and terminals."""
    if len(node_path) < 2:
        return []
    segments: List[Segment] = []
    run_nodes = [node_path[0]]
    run_start = node_path[0]
    run_len = 0.0
    run_dir: Optional[Dir] = None
    terminals: List[str] = []

    def term_label(p: Point) -> Optional[str]:
        a = node_attrs.get(p, {})
        if a.get("type") == "terminal":
            return a.get("id") or a.get("label")
        return None

    for i in range(1, len(node_path)):
        u, v = node_path[i - 1], node_path[i]
        d = direction(u, v)
        w = edge_weight(G, u, v)
        if run_dir is None:
            run_dir = d
        if d != run_dir:
            segments.append(Segment(run_start, u, run_nodes[:], run_len, direction_name(run_dir), terminals[:]))
            run_start = u
            run_nodes = [u]
            run_len = 0.0
            terminals = []
            run_dir = d
        run_nodes.append(v)
        run_len += w
        label = term_label(v)
        if label and v != run_start:
            terminals.append(label)
            segments.append(Segment(run_start, v, run_nodes[:], run_len, direction_name(run_dir), terminals[:]))
            run_start = v
            run_nodes = [v]
            run_len = 0.0
            terminals = []
            run_dir = None
    if len(run_nodes) > 1 and run_len > 0:
        segments.append(Segment(run_start, run_nodes[-1], run_nodes[:], run_len, direction_name(run_dir or (0, 0)), terminals[:]))
    return segments


def path_to_movements(G: Graph, node_attrs: Dict[Point, Dict[str, Any]], node_path: Sequence[Point]) -> List[Movement]:
    """Convert a node path to compressed MOVE and CHECKPOINT records."""
    if not node_path:
        return []
    movements: List[Movement] = []
    first_attrs = node_attrs.get(node_path[0], {})
    if first_attrs.get("type") == "terminal":
        movements.append(Movement(action="CHECKPOINT", node=node_path[0], label=first_attrs.get("label"), product_id=first_attrs.get("id")))
    for seg in path_to_segments(G, node_attrs, node_path):
        movements.append(Movement(action="MOVE", direction=seg.direction, distance=seg.length, start=seg.start, end=seg.end))
        end_attrs = node_attrs.get(seg.end, {})
        if end_attrs.get("type") == "terminal":
            movements.append(Movement(action="CHECKPOINT", node=seg.end, label=end_attrs.get("label"), product_id=end_attrs.get("id")))
    return movements


def extend_straight_segment(G: Graph, node_attrs: Dict[Point, Dict[str, Any]], start: Point, dxy: Dir) -> Segment:
    """Walk straight from start until a terminal is hit or no straight continuation exists."""
    nodes = [start]
    length = 0.0
    cur = start
    terminals: List[str] = []
    nxt = neighbor_in_direction(G, cur, dxy)
    if nxt is None:
        return Segment(start, start, [start], 0.0, direction_name(dxy), terminals)
    while nxt is not None:
        length += edge_weight(G, cur, nxt)
        nodes.append(nxt)
        attrs = node_attrs.get(nxt, {})
        if attrs.get("type") == "terminal" and nxt != start:
            terminals.append(attrs.get("id") or attrs.get("label", "terminal"))
            break
        cur = nxt
        nxt = neighbor_in_direction(G, cur, dxy)
    return Segment(start, nodes[-1], nodes, length, direction_name(dxy), terminals)


def terminal_action_library(
    G: Graph,
    node_attrs: Dict[Point, Dict[str, Any]],
    allowed_dir_filter: Optional[Dict[Point, List[Dir]]] = None,
) -> Dict[Point, Dict[str, Any]]:
    """For every terminal, list allowed UP/DOWN/LEFT/RIGHT actions and straight segments."""
    lib: Dict[Point, Dict[str, Any]] = {}
    terminals = [p for p, a in node_attrs.items() if a.get("type") == "terminal"]
    for t in terminals:
        moves: Dict[str, Segment] = {}
        mask = {"UP": 0, "DOWN": 0, "LEFT": 0, "RIGHT": 0}
        allowed = set(allowed_dir_filter.get(t, DIRECTIONS.values())) if allowed_dir_filter else set(DIRECTIONS.values())
        for name, dxy in DIRECTIONS.items():
            if dxy not in allowed:
                continue
            if neighbor_in_direction(G, t, dxy) is None:
                continue
            seg = extend_straight_segment(G, node_attrs, t, dxy)
            if seg.length > 0:
                moves[name] = seg
                mask[name] = 1
        lib[t] = {"mask": mask, "moves": moves}
    return lib
