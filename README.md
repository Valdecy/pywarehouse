# pyWarehouse

<p align="center">
  <img src="https://github.com/Valdecy/Datasets/raw/master/Data%20Science/pw_architecture.svg" alt="Logo" width="700" />
</p>

`pyWarehouse` models a warehouse as a **Walkable Graph** and computes picker routes using classical warehouse heuristics, custom product orders, and tabular RL policies. It also exports a polished, dependency-light HTML viewer with route animation, live telemetry, pick sequence, and a scrubbable timeline.

The package is designed for three uses:

- **Warehouse Routing Experiments**: Compare `s_shape`, `return`, `midpoint`, `largest_gap`, `combined`, `q_learning`, and `sarsa` on the same layout.
- **Algorithm Teaching**: Make routing policies visible, not only numerical.
- **RL Research Prototyping**: Expose terminal sequences, waypoint sequences, graph paths, and transition tables.

---

## Core Idea

A warehouse is represented as a **Walkable Graph**:

| Component | Meaning |
|---|---|
| **Steiner Nodes** | Aisle/cross-aisle intersections, corridor corners, auxiliary points |
| **Terminal Nodes** | Product locations, start nodes, finish nodes |
| **Edges** | Walkable corridor segments |
| **Distance** | Manhattan distance along the warehouse graph, not Euclidean shortcuts |

The router builds a full terminal distance structure over the graph, so route distance, animation distance, telemetry, and RL rewards are all computed from the same physical model.

---

## Installation

```bash
pip install pywarehouse-routing
```

---

##  Colab Demos

- [ Single Block ](https://colab.research.google.com/drive/1m1cYGzh55UUGULFmku1MPKolpae8k0pw?usp=sharing)
- [ Multi-Block ](https://colab.research.google.com/drive/1dpOexz4hDQSG5bVGsKqwhnmB-jz3VPN7?usp=sharing)
- [ Tabular RL ](https://colab.research.google.com/drive/1x4IHOSp9Ljtvl0TjJVPJkT0vcflvzJgv?usp=sharing)

---

## Quick Start

```python
from pywarehouse import WarehouseLayout, Router, Plotter

# 1) Build a rectangular warehouse layout
layout = WarehouseLayout.rectangular(
    num_aisles      = 6,
    slots_per_block = 11,
    num_blocks      = 1,
    aisle_spacing   = 5.0,
    slot_pitch      = 1.0,
    front_corridor  = 1.0,
    cross_corridor  = 3.0,
    back_corridor   = 0.0,
)

# 2) Add products as: (product_id, aisle, slot, block)
layout.add_products([
    ("A", 0,  5, 0),
    ("B", 0,  9, 0),
    ("D", 1,  9, 0),
    ("C", 1, 10, 0),
    ("H", 2,  1, 0),
    ("G", 2,  4, 0),
    ("E", 2,  5, 0),
    ("F", 2, 10, 0),
])

# 3) Start and finish can be the same depot
layout.set_start("START",   x = 0.0, y = -1.0)
layout.set_finish("FINISH", x = 0.0, y = -1.0)

# 4) Build graph and solve
G, node_attrs, edges_df = layout.build_graph()
router                  = Router(layout, G, node_attrs)
route                   = router.solve(strategy = "s_shape")
print("Distance:",  route.total_distance)
print("Terminals:", route.terminal_sequence)
print("Waypoints:", route.waypoint_sequence)

# 5) Export a route viewer
plotter = Plotter(G, node_attrs)
viewer  = plotter.draw_route(route, label_mode = "id", title = "Route")
viewer.write_html("route.html")

```

---

## Strategies

`pyWarehouse` supports classical warehouse heuristics, custom orders, and tabular RL strategies.

| Strategy | Type | Behavior |
|---|---|---|
| `s_shape` | Classical | Traverses active aisles and alternates side when useful. |
| `return` | Classical | Enters each active aisle, reaches the deepest required pick, and returns to the same side. |
| `midpoint` | Classical | Splits aisle work around the midpoint; lower picks are served from the lower corridor and upper picks from the upper corridor. |
| `largest_gap` | Classical | Avoids the largest unused vertical gap in each aisle. |
| `combined` | Hybrid classical | Chooses aisle-by-aisle whether to return or traverse, producing a mixed policy. |
| `q_learning` | RL | Learns a terminal-level picking sequence using tabular Q-learning. |
| `sarsa` | RL | Learns a terminal-level sequence using on-policy SARSA. |
| `custom` | User-defined | Follows a user-provided terminal order. |

---

## Route Representation

<p align="center">
  <img src="https://github.com/Valdecy/Datasets/raw/master/Data%20Science/pw_route_layers.svg" alt="Logo" width="700" />
</p>

A solved `Route` separates policy, physical path, and visualization data:

```python
route.strategy
route.terminal_sequence
route.waypoint_sequence
route.waypoint_details
route.node_path
route.segments
route.movements
route.total_distance
```

The distinction matters:

| Attribute | Purpose |
|---|---|
| `terminal_sequence` | The sequence of `START`, products, and `FINISH`. |
| `waypoint_sequence` | Policy-level milestones, including corridor waypoints such as `B0:A2:LOWER`. |
| `waypoint_details` | JSON-friendly metadata for each waypoint. |
| `node_path` | Full graph path used for distance, telemetry, and animation. |
| `segments` | Edge-level route geometry. |
| `movements` | Compressed movement commands such as `UP`, `DOWN`, `LEFT`, `RIGHT`, and `CHECKPOINT`. |

Classical heuristics such as `return`, `midpoint`, `largest_gap`, and `combined` are **not only product permutations**. They are aisle policies. The waypoint layer prevents these strategies from collapsing into generic shortest-path chaining.

---

## Tabular RL

<p align="center">
  <img src="https://github.com/Valdecy/Datasets/raw/master/Data%20Science/pw_rl_mdp.svg" alt="Logo" width="700"/>
</p>

`pyWarehouse` includes a compact tabular RL interface. The default RL environment is a **terminal metric-closure MDP**:

```text
state  = (current terminal id, picked-product bit mask)
action = next unpicked product, or FINISH after all products are picked
reward = - graph shortest-path distance
```

### Q-learning / SARSA

```python
from pywarehouse import QLearningConfig, TabularQLearningRouter

cfg_q = QLearningConfig(
    algorithm       = "q_learning",
    episodes        = 30000,
    alpha           = 0.25,
    gamma           = 1.0,
    epsilon         = 1.0,
    epsilon_min     = 0.02,
    epsilon_decay   = 0.9995,
    seed            = 11,
    route_selection = "best", # "best" or "greedy"
)

result_q = TabularQLearningRouter(router, cfg_q).train()
route_q  = result_q.route

print(route_q.terminal_sequence)
print(route_q.total_distance)
print(result_q.metadata)

cfg_s = QLearningConfig(
    algorithm       = "sarsa",
    episodes        = 30000,
    alpha           = 0.25,
    gamma           = 1.0,
    epsilon         = 1.0,
    epsilon_min     = 0.02,
    epsilon_decay   = 0.9995,
    seed            = 11,
    route_selection = "best", # "best" or "greedy"
)

result_s = TabularQLearningRouter(router, cfg_s).train()
route_s  = result_s.route

print(route_s.terminal_sequence)
print(route_s.total_distance)
print(result_s.metadata)
```

### Direct Router API

```python
route_q = router.solve(strategy = "q_learning", rl_config = cfg_q)
route_s = router.solve(strategy = "sarsa",      rl_config = cfg_s)
```

### Waypoint Transition Table

For RL diagnostics, imitation learning, or route-guidance experiments:

```python
route       = router.solve(strategy = "combined")
transitions = router.waypoint_transition_table(route)
for row in transitions:
    print(row["state"], row["action"], row["next_state"], row["reward"])
```

Each row has the form:

```python
{
    "state": "0:START",
    "action": "GO_TO:B0:A0:LOWER",
    "next_state": "1:B0:A0:LOWER",
    "distance": 1.0,
    "reward": -1.0,
    "done": False,
}
```

Repeated waypoints are occurrence-indexed, so a waypoint such as `B0:A2:LOWER` can appear multiple times without losing its visit identity.

---
