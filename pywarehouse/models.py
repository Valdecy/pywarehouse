"""Core dataclasses used by pyWarehouse."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Point = Tuple[float, float]
Graph = Dict[Point, Dict[Point, float]]


@dataclass(frozen=True)
class Product:
    """A product/pick address inside the warehouse."""

    id: str
    aisle: int
    slot: int
    block: int
    x: float
    y: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def point(self) -> Point:
        return (self.x, self.y)


@dataclass(frozen=True)
class Depot:
    """Start or finish point."""

    id: str
    x: float
    y: float
    kind: str = "start"  # start | finish | start_finish
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def point(self) -> Point:
        return (self.x, self.y)


@dataclass
class Segment:
    """A straight route segment between two graph nodes."""

    start: Point
    end: Point
    nodes: List[Point]
    length: float
    direction: str
    terminals: List[str] = field(default_factory=list)


@dataclass
class Movement:
    """A compact movement instruction for handheld execution or animation."""

    action: str  # MOVE | CHECKPOINT
    direction: Optional[str] = None
    distance: float = 0.0
    start: Optional[Point] = None
    end: Optional[Point] = None
    label: Optional[str] = None
    node: Optional[Point] = None
    product_id: Optional[str] = None


@dataclass
class Route:
    """A solved picking route."""

    strategy: str
    terminal_sequence: List[str]
    node_path: List[Point]
    segments: List[Segment]
    movements: List[Movement]
    total_distance: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    waypoint_sequence: List[str] = field(default_factory=list)
    waypoint_details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "terminal_sequence": self.terminal_sequence,
            "waypoint_sequence": self.waypoint_sequence,
            "waypoint_details": self.waypoint_details,
            "node_path": self.node_path,
            "segments": [s.__dict__ for s in self.segments],
            "movements": [m.__dict__ for m in self.movements],
            "total_distance": self.total_distance,
            "metadata": self.metadata,
        }
