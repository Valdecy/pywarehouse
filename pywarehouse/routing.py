"""Routing heuristics for pyWarehouse."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .graph import (
    floyd_warshall_all_pairs,
    path_length,
    path_to_movements,
    path_to_segments,
    reconstruct_fw_path,
    shortest_path,
)
from .layout import WarehouseLayout
from .models import Graph, Point, Route


class Router:
    """Solve order-picking routes on a WarehouseLayout graph."""

    SUPPORTED_STRATEGIES = {"s_shape", "s", "traversal", "return", "midpoint", "largest_gap", "combined", "custom", "q_learning", "q-learning", "tabular_rl", "rl", "sarsa"}

    def __init__(
        self,
        layout: WarehouseLayout,
        G: Optional[Graph] = None,
        node_attrs: Optional[Dict[Point, Dict[str, Any]]] = None,
        *,
        use_distance_matrix: bool = True,
    ):
        self.layout = layout
        if G is None or node_attrs is None:
            self.G, self.node_attrs, self.edges_df = layout.build_graph()
        else:
            self.G = G
            self.node_attrs = node_attrs
            self.edges_df = None
        self.id_to_point = self._make_id_to_point()

        # Full graph-based distance matrix. This prevents accidental direct
        # terminal-to-terminal Manhattan shortcuts and keeps every strategy,
        # telemetry value, and animation event on the same physical distance
        # model. The graph edge weights already encode aisle_spacing, slot_pitch,
        # corridor widths, and depot stubs.
        self.use_distance_matrix = bool(use_distance_matrix)
        if self.use_distance_matrix:
            self.distance_matrix, self.next_hop = floyd_warshall_all_pairs(self.G)
        else:
            self.distance_matrix, self.next_hop = {}, {}

    def _make_id_to_point(self) -> Dict[str, Point]:
        mapping: Dict[str, Point] = {}
        for p, attrs in self.node_attrs.items():
            if attrs.get("type") == "terminal":
                raw = attrs.get("id") or ""
                for tid in str(raw).split("/"):
                    if tid:
                        mapping[tid] = p
        return mapping

    def terminal_point(self, terminal_id: str) -> Point:
        if terminal_id not in self.id_to_point:
            raise KeyError(f"Unknown terminal id: {terminal_id}")
        return self.id_to_point[terminal_id]

    def _shortest_leg_path(self, source: Point, target: Point) -> List[Point]:
        """Shortest walkable graph path using the FW matrix when available."""
        if source == target:
            return [source]
        if self.use_distance_matrix and self.next_hop:
            return reconstruct_fw_path(self.next_hop, source, target)
        return shortest_path(self.G, source, target)

    def _shortest_leg_distance(self, source: Point, target: Point, path: Optional[Sequence[Point]] = None) -> float:
        """Shortest walkable graph distance using the FW matrix when available."""
        if source == target:
            return 0.0
        if self.use_distance_matrix and self.distance_matrix:
            return float(self.distance_matrix[source][target])
        return path_length(self.G, path if path is not None else self._shortest_leg_path(source, target))

    def _default_start_id(self) -> str:
        if not self.layout.starts:
            raise ValueError("No start point was defined. Use layout.set_start(...) or layout.add_start(...).")
        return next(iter(self.layout.starts.keys()))

    def _default_finish_id(self, start_id: str) -> str:
        if self.layout.finishes:
            return next(iter(self.layout.finishes.keys()))
        return start_id

    def _product_ids(self, product_ids: Optional[Sequence[str]]) -> List[str]:
        ids = list(product_ids) if product_ids is not None else list(self.layout.products.keys())
        unknown = [pid for pid in ids if pid not in self.layout.products]
        if unknown:
            raise KeyError(f"Unknown product ids: {unknown}")
        return ids

    def _group_by_aisle(self, product_ids: Sequence[str]) -> Dict[int, List[str]]:
        groups: Dict[int, List[str]] = defaultdict(list)
        for pid in product_ids:
            groups[self.layout.products[pid].aisle].append(pid)
        return groups

    def _terminal_waypoint_detail(self, terminal_id: str) -> Dict[str, Any]:
        """Structured metadata for a terminal waypoint.

        This is intentionally lightweight and JSON-friendly, so the same route
        object can be used by the HTML viewer and by tabular-RL experiments.
        """
        point = self.terminal_point(terminal_id)
        detail: Dict[str, Any] = {
            "id": str(terminal_id),
            "kind": "terminal",
            "x": float(point[0]),
            "y": float(point[1]),
            "point": [float(point[0]), float(point[1])],
        }
        if terminal_id in self.layout.products:
            prod = self.layout.products[terminal_id]
            detail.update({
                "kind": "pick",
                "product_id": terminal_id,
                "aisle": int(prod.aisle),
                "slot": int(prod.slot),
                "block": int(prod.block),
            })
        elif terminal_id in self.layout.starts and terminal_id in self.layout.finishes:
            detail["kind"] = "start_finish"
        elif terminal_id in self.layout.starts:
            detail["kind"] = "start"
        elif terminal_id in self.layout.finishes:
            detail["kind"] = "finish"
        return detail

    def _corridor_waypoint_label(self, block: int, aisle: int, side: str) -> str:
        return f"B{int(block)}:A{int(aisle)}:{side.upper()}"

    def _corridor_waypoint_detail(self, block: int, aisle: int, side: str, point: Point) -> Dict[str, Any]:
        return {
            "id": self._corridor_waypoint_label(block, aisle, side),
            "kind": "corridor",
            "block": int(block),
            "aisle": int(aisle),
            "side": str(side).lower(),
            "x": float(point[0]),
            "y": float(point[1]),
            "point": [float(point[0]), float(point[1])],
        }

    def _terminal_waypoints(self, terminal_sequence: Sequence[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
        labels = [str(tid) for tid in terminal_sequence]
        return labels, [self._terminal_waypoint_detail(tid) for tid in labels]

    def _sequence_s_shape(self, product_ids: Sequence[str]) -> List[str]:
        groups = self._group_by_aisle(product_ids)
        seq: List[str] = []
        front_to_back = True
        for aisle in sorted(groups):
            ids = sorted(groups[aisle], key=lambda pid: self.layout.products[pid].y, reverse=not front_to_back)
            seq.extend(ids)
            front_to_back = not front_to_back
        return seq

    def _sequence_return(self, product_ids: Sequence[str]) -> List[str]:
        groups = self._group_by_aisle(product_ids)
        seq: List[str] = []
        for aisle in sorted(groups):
            # Enter from the front; visit shallow to deep.
            seq.extend(sorted(groups[aisle], key=lambda pid: self.layout.products[pid].y))
        return seq

    def _sequence_midpoint(self, product_ids: Sequence[str]) -> List[str]:
        groups = self._group_by_aisle(product_ids)
        y_mid = 0.5 * (self.layout.cross_ys[0] + self.layout.cross_ys[-1])
        seq: List[str] = []
        for aisle in sorted(groups):
            lower = [pid for pid in groups[aisle] if self.layout.products[pid].y <= y_mid]
            upper = [pid for pid in groups[aisle] if self.layout.products[pid].y > y_mid]
            seq.extend(sorted(lower, key=lambda pid: self.layout.products[pid].y))
            seq.extend(sorted(upper, key=lambda pid: self.layout.products[pid].y, reverse=True))
        return seq

    def _sequence_largest_gap(self, product_ids: Sequence[str]) -> List[str]:
        groups = self._group_by_aisle(product_ids)
        y_front, y_back = self.layout.cross_ys[0], self.layout.cross_ys[-1]
        seq: List[str] = []
        serpentine = True
        for aisle in sorted(groups):
            ids = groups[aisle]
            ys = sorted([self.layout.products[pid].y for pid in ids])
            anchors = [y_front] + ys + [y_back]
            gaps = [(anchors[i + 1] - anchors[i], anchors[i], anchors[i + 1]) for i in range(len(anchors) - 1)]
            _, lo, hi = max(gaps, key=lambda t: t[0])
            touches_border = abs(lo - y_front) < 1e-9 or abs(hi - y_back) < 1e-9
            if touches_border:
                # Return-like behavior: shallow to deep.
                seq.extend(sorted(ids, key=lambda pid: self.layout.products[pid].y))
            else:
                seq.extend(sorted(ids, key=lambda pid: self.layout.products[pid].y, reverse=not serpentine))
                serpentine = not serpentine
        return seq

    def product_sequence(self, strategy: str, product_ids: Optional[Sequence[str]] = None, custom_order: Optional[Sequence[str]] = None) -> List[str]:
        """Return an ordered list of product ids for a strategy."""
        strategy = strategy.lower()
        ids = self._product_ids(product_ids)
        if strategy in {"s_shape", "s", "traversal"}:
            return self._sequence_s_shape(ids)
        if strategy == "return":
            return self._sequence_return(ids)
        if strategy == "midpoint":
            return self._sequence_midpoint(ids)
        if strategy == "largest_gap":
            return self._sequence_largest_gap(ids)
        if strategy == "custom":
            if not custom_order:
                raise ValueError("custom_order is required for strategy='custom'")
            order = list(custom_order)
            unknown = [pid for pid in order if pid not in self.layout.products]
            if unknown:
                raise KeyError(f"Unknown product ids in custom_order: {unknown}")
            missing = [pid for pid in ids if pid not in order]
            # If user passed product_ids, append missing selected products at the end to avoid silent omission.
            return order + missing
        raise ValueError(f"Unknown strategy '{strategy}'. Supported: {sorted(self.SUPPORTED_STRATEGIES)}")

    def lift_terminal_sequence(self, terminal_ids: Sequence[str]) -> Tuple[List[Point], float]:
        """Concatenate shortest paths between consecutive terminal ids."""
        if not terminal_ids:
            return [], 0.0
        points = [self.terminal_point(tid) for tid in terminal_ids]
        node_path: List[Point] = [points[0]]
        total = 0.0
        for i in range(1, len(points)):
            u, v = points[i - 1], points[i]
            if u == v:
                continue
            leg = self._shortest_leg_path(u, v)
            total += self._shortest_leg_distance(u, v, leg)
            node_path.extend(leg[1:])
        return node_path, total

    def _solve_return_path(
        self,
        start_id: str,
        finish_id: str,
        product_ids: Sequence[str],
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[Point], float]:
        """Construct a true return-route heuristic on the graph.

        Each active aisle is treated independently: move along the lower/front
        corridor to the aisle entrance, go up only as far as the deepest required
        pick in that aisle, then return to the same lower corridor before moving
        to the next aisle.
        """
        start_p = self.terminal_point(start_id)
        finish_p = self.terminal_point(finish_id)
        node_path: List[Point] = [start_p]
        total = 0.0
        picked: List[str] = []
        waypoint_sequence: List[str] = []
        waypoint_details: List[Dict[str, Any]] = []

        def append_leg_to(target: Point) -> None:
            nonlocal total, node_path
            if node_path[-1] == target:
                return
            source = node_path[-1]
            leg = self._shortest_leg_path(source, target)
            total += self._shortest_leg_distance(source, target, leg)
            node_path.extend(leg[1:])

        def record_terminal(tid: str) -> None:
            waypoint_sequence.append(str(tid))
            waypoint_details.append(self._terminal_waypoint_detail(tid))

        def record_corridor(block: int, aisle: int, side: str, point: Point) -> None:
            waypoint_sequence.append(self._corridor_waypoint_label(block, aisle, side))
            waypoint_details.append(self._corridor_waypoint_detail(block, aisle, side, point))

        def add_corridor(block: int, aisle: int, side: str, point: Point) -> None:
            append_leg_to(point)
            record_corridor(block, aisle, side, point)

        def visit_product(pid: str) -> None:
            append_leg_to(self.terminal_point(pid))
            picked.append(pid)
            record_terminal(pid)

        record_terminal(start_id)
        by_block_aisle: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        for pid in product_ids:
            p = self.layout.products[pid]
            by_block_aisle[(p.block, p.aisle)].append(pid)

        for block in range(self.layout.num_blocks):
            lower_y = self.layout.block_lower_y(block)
            active_aisles = sorted(
                aisle for (b, aisle), ids in by_block_aisle.items()
                if b == block and ids
            )
            for aisle in active_aisles:
                x = self.layout.aisle_xs[aisle]
                entrance = (x, lower_y)
                add_corridor(block, aisle, "lower", entrance)
                for pid in sorted(by_block_aisle[(block, aisle)], key=lambda pid: self.layout.products[pid].y):
                    visit_product(pid)
                add_corridor(block, aisle, "lower", entrance)

        append_leg_to(finish_p)
        record_terminal(finish_id)
        return picked, waypoint_sequence, waypoint_details, node_path, total

    def _solve_midpoint_path(
        self,
        start_id: str,
        finish_id: str,
        product_ids: Sequence[str],
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[Point], float]:
        """Construct a classical midpoint route on the graph.

        Front-half picks are visited from the front/lower cross aisle and
        returned to it. Rear-half picks are visited from the rear/upper cross
        aisle. Explicit corridor waypoints keep the policy from degenerating
        into S-shape behavior.
        """
        start_p = self.terminal_point(start_id)
        finish_p = self.terminal_point(finish_id)
        node_path: List[Point] = [start_p]
        total = 0.0
        picked: List[str] = []
        waypoint_sequence: List[str] = []
        waypoint_details: List[Dict[str, Any]] = []

        def append_leg_to(target: Point) -> None:
            nonlocal total, node_path
            if node_path[-1] == target:
                return
            source = node_path[-1]
            leg = self._shortest_leg_path(source, target)
            total += self._shortest_leg_distance(source, target, leg)
            node_path.extend(leg[1:])

        def record_terminal(tid: str) -> None:
            waypoint_sequence.append(str(tid))
            waypoint_details.append(self._terminal_waypoint_detail(tid))

        def add_corridor(block: int, aisle: int, side: str, point: Point) -> None:
            append_leg_to(point)
            waypoint_sequence.append(self._corridor_waypoint_label(block, aisle, side))
            waypoint_details.append(self._corridor_waypoint_detail(block, aisle, side, point))

        def visit_product(pid: str) -> None:
            append_leg_to(self.terminal_point(pid))
            picked.append(pid)
            record_terminal(pid)

        record_terminal(start_id)
        by_block_aisle: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        for pid in product_ids:
            p = self.layout.products[pid]
            by_block_aisle[(p.block, p.aisle)].append(pid)

        for block in range(self.layout.num_blocks):
            lower_y = self.layout.block_lower_y(block)
            upper_y = self.layout.cross_ys[2 * block + 1]
            mid_y = 0.5 * (lower_y + upper_y)

            lower_groups: Dict[int, List[str]] = {}
            upper_groups: Dict[int, List[str]] = {}
            for (b, aisle), ids in by_block_aisle.items():
                if b != block:
                    continue
                lower = [pid for pid in ids if self.layout.products[pid].y <= mid_y]
                upper = [pid for pid in ids if self.layout.products[pid].y > mid_y]
                if lower:
                    lower_groups[aisle] = sorted(lower, key=lambda pid: self.layout.products[pid].y)
                if upper:
                    upper_groups[aisle] = sorted(upper, key=lambda pid: self.layout.products[pid].y)

            # Front half: enter each relevant aisle from the lower/front
            # corridor, collect shallow-to-deep, then return to the corridor.
            for aisle in sorted(lower_groups):
                x = self.layout.aisle_xs[aisle]
                entrance = (x, lower_y)
                add_corridor(block, aisle, "lower", entrance)
                for pid in lower_groups[aisle]:
                    visit_product(pid)
                add_corridor(block, aisle, "lower", entrance)

            if not upper_groups:
                continue

            # Use the rightmost rear-half aisle as the transition from front to rear.
            traverse_aisle = max(upper_groups)
            x = self.layout.aisle_xs[traverse_aisle]
            lower_entrance = (x, lower_y)
            upper_entrance = (x, upper_y)
            add_corridor(block, traverse_aisle, "lower", lower_entrance)
            for pid in sorted(upper_groups.pop(traverse_aisle), key=lambda pid: self.layout.products[pid].y):
                visit_product(pid)
            add_corridor(block, traverse_aisle, "upper", upper_entrance)

            # Rear half: work right-to-left from the upper/rear corridor.
            rear_aisles = sorted(upper_groups, reverse=True)
            for idx, aisle in enumerate(rear_aisles):
                x = self.layout.aisle_xs[aisle]
                entrance = (x, upper_y)
                add_corridor(block, aisle, "upper", entrance)
                for pid in sorted(upper_groups[aisle], key=lambda pid: self.layout.products[pid].y, reverse=True):
                    visit_product(pid)
                if idx < len(rear_aisles) - 1:
                    add_corridor(block, aisle, "upper", entrance)

        append_leg_to(finish_p)
        record_terminal(finish_id)
        return picked, waypoint_sequence, waypoint_details, node_path, total

    def _largest_gap_partition(
        self,
        ids: Sequence[str],
        lower_y: float,
        upper_y: float,
    ) -> Tuple[List[str], List[str], Tuple[float, float]]:
        """Split one aisle's picks around the largest vertical gap.

        Products at or below the lower side of the largest gap are served from
        the lower/front corridor. Products at or above the upper side are served
        from the upper/rear corridor. The open interval of the largest gap is
        therefore not deliberately traversed in intermediate aisles.
        """
        if not ids:
            return [], [], (lower_y, upper_y)

        sorted_ids = sorted(ids, key=lambda pid: self.layout.products[pid].y)
        ys = sorted({float(self.layout.products[pid].y) for pid in sorted_ids})
        anchors = [float(lower_y)] + ys + [float(upper_y)]
        gaps = [
            (anchors[i + 1] - anchors[i], anchors[i], anchors[i + 1])
            for i in range(len(anchors) - 1)
        ]
        _, gap_lo, gap_hi = max(gaps, key=lambda t: (t[0], t[1]))

        lower_ids = [pid for pid in sorted_ids if self.layout.products[pid].y <= gap_lo + 1e-9]
        upper_ids = [pid for pid in sorted_ids if self.layout.products[pid].y >= gap_hi - 1e-9]
        return lower_ids, upper_ids, (float(gap_lo), float(gap_hi))

    def _solve_largest_gap_path(
        self,
        start_id: str,
        finish_id: str,
        product_ids: Sequence[str],
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[Point], float]:
        """Construct a classical largest-gap route on the graph.

        Intermediate aisles are split around their largest vertical gap. The
        first and last active aisles in each block are used as connector aisles
        between lower/front and upper/rear corridors.
        """
        start_p = self.terminal_point(start_id)
        finish_p = self.terminal_point(finish_id)
        node_path: List[Point] = [start_p]
        total = 0.0
        picked: List[str] = []
        picked_set = set()
        waypoint_sequence: List[str] = []
        waypoint_details: List[Dict[str, Any]] = []

        def append_leg_to(target: Point) -> None:
            nonlocal total, node_path
            if node_path[-1] == target:
                return
            source = node_path[-1]
            leg = self._shortest_leg_path(source, target)
            total += self._shortest_leg_distance(source, target, leg)
            node_path.extend(leg[1:])

        def record_terminal(tid: str) -> None:
            waypoint_sequence.append(str(tid))
            waypoint_details.append(self._terminal_waypoint_detail(tid))

        def add_corridor(block: int, aisle: int, side: str, point: Point) -> None:
            append_leg_to(point)
            waypoint_sequence.append(self._corridor_waypoint_label(block, aisle, side))
            waypoint_details.append(self._corridor_waypoint_detail(block, aisle, side, point))

        def visit_product(pid: str) -> None:
            if pid in picked_set:
                append_leg_to(self.terminal_point(pid))
                return
            append_leg_to(self.terminal_point(pid))
            picked.append(pid)
            picked_set.add(pid)
            record_terminal(pid)

        record_terminal(start_id)
        by_block_aisle: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        for pid in product_ids:
            p = self.layout.products[pid]
            by_block_aisle[(p.block, p.aisle)].append(pid)

        for block in range(self.layout.num_blocks):
            lower_y = self.layout.block_lower_y(block)
            upper_y = self.layout.cross_ys[2 * block + 1]

            active_aisles = sorted(
                aisle for (b, aisle), ids in by_block_aisle.items()
                if b == block and ids
            )
            if not active_aisles:
                continue

            # Degenerate one-aisle case: serve both sides of the largest-gap split.
            if len(active_aisles) == 1:
                aisle = active_aisles[0]
                x = self.layout.aisle_xs[aisle]
                lower_ids, upper_ids, _ = self._largest_gap_partition(
                    by_block_aisle[(block, aisle)], lower_y, upper_y
                )
                if lower_ids:
                    entrance = (x, lower_y)
                    add_corridor(block, aisle, "lower", entrance)
                    for pid in sorted(lower_ids, key=lambda pid: self.layout.products[pid].y):
                        visit_product(pid)
                    add_corridor(block, aisle, "lower", entrance)
                if upper_ids:
                    entrance = (x, upper_y)
                    add_corridor(block, aisle, "upper", entrance)
                    for pid in sorted(upper_ids, key=lambda pid: self.layout.products[pid].y, reverse=True):
                        visit_product(pid)
                    add_corridor(block, aisle, "upper", entrance)
                continue

            leftmost = active_aisles[0]
            rightmost = active_aisles[-1]

            # 1) First active aisle: traverse fully from lower/front to upper/rear.
            x_left = self.layout.aisle_xs[leftmost]
            add_corridor(block, leftmost, "lower", (x_left, lower_y))
            for pid in sorted(by_block_aisle[(block, leftmost)], key=lambda pid: self.layout.products[pid].y):
                visit_product(pid)
            add_corridor(block, leftmost, "upper", (x_left, upper_y))

            lower_parts: Dict[int, List[str]] = {}
            upper_parts: Dict[int, List[str]] = {}
            for aisle in active_aisles[1:-1]:
                lower_ids, upper_ids, _ = self._largest_gap_partition(
                    by_block_aisle[(block, aisle)], lower_y, upper_y
                )
                if lower_ids:
                    lower_parts[aisle] = sorted(lower_ids, key=lambda pid: self.layout.products[pid].y)
                if upper_ids:
                    upper_parts[aisle] = sorted(upper_ids, key=lambda pid: self.layout.products[pid].y, reverse=True)

            # 2) Rear/upper side of intermediate aisles, moving left-to-right.
            for aisle in active_aisles[1:-1]:
                ids = upper_parts.get(aisle, [])
                if not ids:
                    continue
                x = self.layout.aisle_xs[aisle]
                entrance = (x, upper_y)
                add_corridor(block, aisle, "upper", entrance)
                for pid in ids:
                    visit_product(pid)
                add_corridor(block, aisle, "upper", entrance)

            # 3) Last active aisle: traverse fully from upper/rear to lower/front.
            x_right = self.layout.aisle_xs[rightmost]
            add_corridor(block, rightmost, "upper", (x_right, upper_y))
            for pid in sorted(by_block_aisle[(block, rightmost)], key=lambda pid: self.layout.products[pid].y, reverse=True):
                visit_product(pid)
            add_corridor(block, rightmost, "lower", (x_right, lower_y))

            # 4) Front/lower side of intermediate aisles, moving right-to-left.
            for aisle in reversed(active_aisles[1:-1]):
                ids = lower_parts.get(aisle, [])
                if not ids:
                    continue
                x = self.layout.aisle_xs[aisle]
                entrance = (x, lower_y)
                add_corridor(block, aisle, "lower", entrance)
                for pid in ids:
                    visit_product(pid)
                add_corridor(block, aisle, "lower", entrance)

        append_leg_to(finish_p)
        record_terminal(finish_id)
        return picked, waypoint_sequence, waypoint_details, node_path, total

    def _solve_combined_path(
        self,
        start_id: str,
        finish_id: str,
        product_ids: Sequence[str],
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[Point], float]:
        """Construct a combined routing heuristic with explicit waypoints.

        The combined strategy decides aisle by aisle whether to use a return
        pattern or to fully traverse the aisle. It is solved here by a compact
        dynamic program over the active aisles. The DP state is the side of the
        warehouse where the picker exits the last processed aisle. Within a
        block, the picker moves horizontally on that same side until a selected
        traversal changes sides.
        """
        start_p = self.terminal_point(start_id)
        finish_p = self.terminal_point(finish_id)

        by_block_aisle: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        for pid in product_ids:
            p = self.layout.products[pid]
            by_block_aisle[(p.block, p.aisle)].append(pid)

        active_keys = sorted(by_block_aisle.keys())
        if not active_keys:
            node_path, total = self.lift_terminal_sequence([start_id, finish_id])
            waypoint_sequence, waypoint_details = self._terminal_waypoints([start_id, finish_id])
            return [], waypoint_sequence, waypoint_details, node_path, total

        def side_y(block: int, side: str) -> float:
            if side == "lower":
                return self.layout.block_lower_y(block)
            return self.layout.cross_ys[2 * block + 1]

        def side_point(block: int, aisle: int, side: str) -> Point:
            return (self.layout.aisle_xs[aisle], side_y(block, side))

        # A state is represented by a dict for readability:
        #   cost: cost up to the exit point of the last processed aisle
        #   point: current graph point
        #   block/side: current side context
        #   actions: selected aisle-service decisions so far
        states: List[Dict[str, Any]] = [{
            "cost": 0.0,
            "point": start_p,
            "block": None,
            "side": None,
            "actions": [],
        }]

        for block, aisle in active_keys:
            ids = sorted(by_block_aisle[(block, aisle)], key=lambda pid: self.layout.products[pid].y)
            if not ids:
                continue
            min_y = min(float(self.layout.products[pid].y) for pid in ids)
            max_y = max(float(self.layout.products[pid].y) for pid in ids)
            lower_y = side_y(block, "lower")
            upper_y = side_y(block, "upper")
            full_traverse = upper_y - lower_y

            candidate_states: List[Dict[str, Any]] = []
            for st in states:
                # Inside the same block, keep moving on the side where the last
                # aisle left the picker. When entering a new block, allow either
                # side because the shortest graph path can legally move through
                # the cross-corridor structure to that block side.
                if st["block"] == block and st["side"] in {"lower", "upper"}:
                    entry_sides = [st["side"]]
                else:
                    entry_sides = ["lower", "upper"]

                for entry_side in entry_sides:
                    entry = side_point(block, aisle, entry_side)
                    move_cost = self._shortest_leg_distance(st["point"], entry)

                    if entry_side == "lower":
                        service_options = [
                            {
                                "mode": "return",
                                "exit_side": "lower",
                                "picks": ids[:],
                                "service_cost": 2.0 * max(0.0, max_y - lower_y),
                            },
                            {
                                "mode": "traverse",
                                "exit_side": "upper",
                                "picks": ids[:],
                                "service_cost": full_traverse,
                            },
                        ]
                    else:
                        desc = list(reversed(ids))
                        service_options = [
                            {
                                "mode": "return",
                                "exit_side": "upper",
                                "picks": desc,
                                "service_cost": 2.0 * max(0.0, upper_y - min_y),
                            },
                            {
                                "mode": "traverse",
                                "exit_side": "lower",
                                "picks": desc,
                                "service_cost": full_traverse,
                            },
                        ]

                    for opt in service_options:
                        exit_side = opt["exit_side"]
                        exit_point = side_point(block, aisle, exit_side)
                        action = {
                            "block": int(block),
                            "aisle": int(aisle),
                            "entry_side": entry_side,
                            "exit_side": exit_side,
                            "mode": opt["mode"],
                            "picks": list(opt["picks"]),
                        }
                        candidate_states.append({
                            "cost": float(st["cost"] + move_cost + opt["service_cost"]),
                            "point": exit_point,
                            "block": int(block),
                            "side": exit_side,
                            "actions": list(st["actions"]) + [action],
                        })

            # At each aisle all survivors are located at that aisle's x-coordinate;
            # only the exit side matters for future movement. Keep the cheapest
            # survivor per side.
            best_by_side: Dict[str, Dict[str, Any]] = {}
            for st in candidate_states:
                key = st["side"]
                if key not in best_by_side or st["cost"] < best_by_side[key]["cost"]:
                    best_by_side[key] = st
            states = list(best_by_side.values())

        best = min(
            states,
            key=lambda st: float(st["cost"] + self._shortest_leg_distance(st["point"], finish_p)),
        )
        actions = list(best["actions"])

        node_path: List[Point] = [start_p]
        total = 0.0
        picked: List[str] = []
        waypoint_sequence: List[str] = []
        waypoint_details: List[Dict[str, Any]] = []

        def append_leg_to(target: Point) -> None:
            nonlocal total, node_path
            if node_path[-1] == target:
                return
            source = node_path[-1]
            leg = self._shortest_leg_path(source, target)
            total += self._shortest_leg_distance(source, target, leg)
            node_path.extend(leg[1:])

        def record_terminal(tid: str) -> None:
            waypoint_sequence.append(str(tid))
            waypoint_details.append(self._terminal_waypoint_detail(tid))

        def add_corridor(block: int, aisle: int, side: str) -> None:
            point = side_point(block, aisle, side)
            append_leg_to(point)
            waypoint_sequence.append(self._corridor_waypoint_label(block, aisle, side))
            waypoint_details.append(self._corridor_waypoint_detail(block, aisle, side, point))

        def visit_product(pid: str) -> None:
            append_leg_to(self.terminal_point(pid))
            picked.append(pid)
            record_terminal(pid)

        record_terminal(start_id)
        for action in actions:
            block = int(action["block"])
            aisle = int(action["aisle"])
            add_corridor(block, aisle, str(action["entry_side"]))
            for pid in action["picks"]:
                visit_product(pid)
            add_corridor(block, aisle, str(action["exit_side"]))

        append_leg_to(finish_p)
        record_terminal(finish_id)
        return picked, waypoint_sequence, waypoint_details, node_path, total

    def solve(
        self,
        strategy: str = "s_shape",
        start: Optional[str] = None,
        finish: Optional[str] = None,
        product_ids: Optional[Sequence[str]] = None,
        custom_order: Optional[Sequence[str]] = None,
        return_to_start: bool = False,
        rl_config: Optional[Any] = None,
    ) -> Route:
        """Solve a route using a classical heuristic or custom product order.

        Parameters
        ----------
        strategy:
            One of ``s_shape``, ``return``, ``midpoint``, ``largest_gap``, ``combined``, ``custom``, ``q_learning``, or ``sarsa``.
        start, finish:
            Terminal ids for start/finish. If finish is omitted and no finish was registered,
            the route returns to start.
        product_ids:
            Subset of products to pick. Defaults to all products.
        custom_order:
            Product id sequence when ``strategy='custom'``.
        return_to_start:
            Force finish to be equal to start.
        """
        start_id = start or self._default_start_id()
        finish_id = start_id if return_to_start else (finish or self._default_finish_id(start_id))

        norm_strategy = strategy.lower()
        if norm_strategy == "return":
            ids = self._product_ids(product_ids)
            seq_products, waypoint_sequence, waypoint_details, node_path, total = self._solve_return_path(start_id, finish_id, ids)
        elif norm_strategy == "midpoint":
            ids = self._product_ids(product_ids)
            seq_products, waypoint_sequence, waypoint_details, node_path, total = self._solve_midpoint_path(start_id, finish_id, ids)
        elif norm_strategy == "largest_gap":
            ids = self._product_ids(product_ids)
            seq_products, waypoint_sequence, waypoint_details, node_path, total = self._solve_largest_gap_path(start_id, finish_id, ids)
        elif norm_strategy == "combined":
            ids = self._product_ids(product_ids)
            seq_products, waypoint_sequence, waypoint_details, node_path, total = self._solve_combined_path(start_id, finish_id, ids)
        elif norm_strategy in {"q_learning", "q-learning", "tabular_rl", "rl", "sarsa"}:
            from .rl import QLearningConfig, TabularQLearningRouter
            cfg = rl_config or QLearningConfig(algorithm="sarsa" if norm_strategy == "sarsa" else "q_learning")
            # If a config object was supplied but the user asked for SARSA via strategy,
            # honor the explicit strategy name without mutating the caller's config.
            if norm_strategy == "sarsa" and getattr(cfg, "algorithm", "q_learning").lower() != "sarsa":
                cfg = QLearningConfig(**{**cfg.__dict__, "algorithm": "sarsa"})
            result = TabularQLearningRouter(self, cfg).train(product_ids=product_ids, start=start_id, finish=finish_id)
            return result.route
        else:
            seq_products = self.product_sequence(strategy, product_ids, custom_order)
            terminal_ids = [start_id] + seq_products + [finish_id]
            node_path, total = self.lift_terminal_sequence(terminal_ids)
            waypoint_sequence, waypoint_details = self._terminal_waypoints(terminal_ids)

        terminal_sequence = [start_id] + seq_products + [finish_id]
        segments = path_to_segments(self.G, self.node_attrs, node_path)
        movements = path_to_movements(self.G, self.node_attrs, node_path)
        return Route(
            strategy=strategy,
            terminal_sequence=terminal_sequence,
            node_path=node_path,
            segments=segments,
            movements=movements,
            total_distance=total,
            metadata={
                "start": start_id,
                "finish": finish_id,
                "product_sequence": seq_products,
            },
            waypoint_sequence=waypoint_sequence,
            waypoint_details=waypoint_details,
        )

    def waypoint_transition_table(self, route: Route) -> List[Dict[str, Any]]:
        """Return a tabular-RL friendly transition table from route waypoints.

        Each row is a deterministic transition between consecutive waypoints in
        ``route.waypoint_sequence``. The state id is occurrence-indexed, so
        repeated waypoints such as ``B0:A2:LOWER`` remain distinct visits.
        Rewards are negative graph distances, which is the usual convention
        when minimizing travel length with RL.
        """
        details = list(getattr(route, "waypoint_details", []) or [])
        if len(details) < 2:
            return []

        rows: List[Dict[str, Any]] = []
        for i in range(len(details) - 1):
            src = details[i]
            dst = details[i + 1]
            u = (float(src["x"]), float(src["y"]))
            v = (float(dst["x"]), float(dst["y"]))
            distance = self._shortest_leg_distance(u, v)
            rows.append({
                "step": i,
                "state": f"{i}:{src['id']}",
                "state_label": src["id"],
                "state_kind": src.get("kind"),
                "action": f"GO_TO:{dst['id']}",
                "next_state": f"{i + 1}:{dst['id']}",
                "next_state_label": dst["id"],
                "next_state_kind": dst.get("kind"),
                "distance": float(distance),
                "reward": -float(distance),
                "done": i == len(details) - 2,
            })
        return rows

    def compare(self, strategies: Sequence[str], **kwargs: Any) -> Dict[str, Route]:
        """Solve and return several strategies for comparison."""
        return {s: self.solve(strategy=s, **kwargs) for s in strategies}
