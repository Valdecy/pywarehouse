"""Tabular reinforcement-learning tools for pyWarehouse.

The classes in this module intentionally stay small and dependency-light.  They
model a picking route as an episodic deterministic MDP over terminal states:

    state  = (current terminal id, picked-product bit mask)
    action = next unpicked product id, or the finish id after all picks are done
    reward = - graph shortest-path distance

Distances are read from :class:`pywarehouse.routing.Router`, so the learner uses
exactly the same Floyd-Warshall warehouse distance matrix used by the classical
heuristics and by the HTML telemetry.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import math
import random
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .graph import path_to_movements, path_to_segments
from .models import Point, Route

State = Tuple[str, int]
QTable = Dict[State, Dict[str, float]]


@dataclass
class QLearningConfig:
    """Configuration for tabular picking-route learning.

    Parameters
    ----------
    episodes:
        Number of complete picking episodes used for training.
    alpha:
        Learning rate.
    gamma:
        Discount factor.  Use ``1.0`` for pure distance minimization in a
        finite-horizon route problem.
    epsilon:
        Initial epsilon-greedy exploration probability.
    epsilon_min:
        Lower bound for epsilon after decay.
    epsilon_decay:
        Multiplicative decay applied after each episode.
    seed:
        Random seed for reproducible exploration.
    max_steps:
        Safety cap per episode.  If ``None``, it is set to ``n_products + 1``.
    algorithm:
        ``"q_learning"`` or ``"sarsa"``.
    optimistic_initial_value:
        Initial Q-value for unseen state-action pairs.  With negative distance
        rewards, ``0.0`` is optimistic and encourages exploration.
    finish_bonus:
        Optional terminal bonus when the picker reaches finish after all picks.
        Keep this at ``0.0`` for pure route-distance minimization.
    route_selection:
        Which route to return after training. ``"best"`` selects the shorter
        of the best observed training episode and the greedy Q-table rollout;
        ``"greedy"`` returns the greedy Q-table rollout; ``"best_episode"``
        returns the best complete episode seen during exploration.
    """

    episodes: int = 20000
    alpha: float = 0.25
    gamma: float = 1.0
    epsilon: float = 1.0
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.9993
    seed: Optional[int] = 7
    max_steps: Optional[int] = None
    algorithm: str = "q_learning"
    optimistic_initial_value: float = 0.0
    finish_bonus: float = 0.0
    route_selection: str = "best"


@dataclass
class TabularRLResult:
    """Result returned by :class:`TabularQLearningRouter`."""

    route: Route
    q_table: QTable
    policy: Dict[State, str]
    episode_rewards: List[float]
    episode_distances: List[float]
    config: QLearningConfig
    product_ids: List[str]
    start: str
    finish: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def q_table_rows(self) -> List[Dict[str, Any]]:
        """Return the Q-table as JSON/CSV-friendly rows."""
        rows: List[Dict[str, Any]] = []
        for (current, mask), actions in sorted(self.q_table.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            picked = [pid for i, pid in enumerate(self.product_ids) if mask & (1 << i)]
            for action, value in sorted(actions.items()):
                rows.append({
                    "current": current,
                    "mask": int(mask),
                    "picked": tuple(picked),
                    "action": action,
                    "q_value": float(value),
                })
        return rows

    def policy_rows(self) -> List[Dict[str, Any]]:
        """Return the greedy policy as JSON/CSV-friendly rows."""
        rows: List[Dict[str, Any]] = []
        for (current, mask), action in sorted(self.policy.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            picked = [pid for i, pid in enumerate(self.product_ids) if mask & (1 << i)]
            rows.append({
                "current": current,
                "mask": int(mask),
                "picked": tuple(picked),
                "action": action,
            })
        return rows

    def q_table_dataframe(self):
        """Return a pandas DataFrame with Q-values.

        pandas is already a pyWarehouse dependency.  The import is kept local so
        this module remains light for users who only need the route object.
        """
        import pandas as pd
        return pd.DataFrame(self.q_table_rows())

    def policy_dataframe(self):
        """Return a pandas DataFrame with the greedy policy."""
        import pandas as pd
        return pd.DataFrame(self.policy_rows())


class TabularPickingEnv:
    """Deterministic terminal-state MDP for order picking.

    The environment does not move cell-by-cell.  One action sends the picker to
    the next terminal through the warehouse graph shortest path.  This makes the
    state space small enough for tabular RL while preserving the real physical
    distance matrix.
    """

    def __init__(
        self,
        router: Any,
        product_ids: Optional[Sequence[str]] = None,
        start: Optional[str] = None,
        finish: Optional[str] = None,
        *,
        finish_bonus: float = 0.0,
    ) -> None:
        self.router = router
        self.start = start or router._default_start_id()
        self.finish = finish or router._default_finish_id(self.start)
        self.product_ids = router._product_ids(product_ids)
        self.product_to_bit = {pid: i for i, pid in enumerate(self.product_ids)}
        self.full_mask = (1 << len(self.product_ids)) - 1
        self.finish_bonus = float(finish_bonus)
        self.state: State = (self.start, 0)

    def reset(self) -> State:
        self.state = (self.start, 0)
        return self.state

    def available_actions(self, state: Optional[State] = None) -> List[str]:
        current, mask = state if state is not None else self.state
        if mask == self.full_mask:
            return [self.finish]
        return [pid for pid in self.product_ids if not (mask & (1 << self.product_to_bit[pid]))]

    def distance(self, source_id: str, target_id: str) -> float:
        u = self.router.terminal_point(source_id)
        v = self.router.terminal_point(target_id)
        return float(self.router._shortest_leg_distance(u, v))

    def step(self, action: str) -> Tuple[State, float, bool, Dict[str, Any]]:
        current, mask = self.state
        legal = self.available_actions(self.state)
        if action not in legal:
            raise ValueError(f"Illegal action {action!r} from state {self.state!r}; legal actions are {legal!r}")

        dist = self.distance(current, action)
        next_mask = mask
        done = False
        picked_product: Optional[str] = None

        if action in self.product_to_bit:
            next_mask = mask | (1 << self.product_to_bit[action])
            picked_product = action
        elif action == self.finish and mask == self.full_mask:
            done = True

        reward = -float(dist)
        if done:
            reward += self.finish_bonus

        self.state = (action, next_mask)
        return self.state, reward, done, {
            "distance": float(dist),
            "picked_product": picked_product,
            "done": done,
            "mask": int(next_mask),
        }


class TabularQLearningRouter:
    """Learn a picking route using tabular Q-learning or SARSA."""

    def __init__(self, router: Any, config: Optional[QLearningConfig] = None) -> None:
        self.router = router
        self.config = config or QLearningConfig()

    def _q(self, q_table: QTable, state: State, action: str) -> float:
        return float(q_table.get(state, {}).get(action, self.config.optimistic_initial_value))

    def _choose_action(
        self,
        env: TabularPickingEnv,
        q_table: QTable,
        state: State,
        epsilon: float,
        rng: random.Random,
    ) -> str:
        actions = env.available_actions(state)
        if not actions:
            raise RuntimeError(f"No legal actions from state {state!r}")
        if rng.random() < epsilon:
            return rng.choice(actions)

        values = [(self._q(q_table, state, a), a) for a in actions]
        max_q = max(v for v, _ in values)
        # Tie-break by immediate distance, then by id, to keep the greedy route
        # deterministic and physically plausible when Q-values are equal.
        best = [a for v, a in values if abs(v - max_q) <= 1e-12]
        current = state[0]
        best.sort(key=lambda a: (env.distance(current, a), a))
        return best[0]

    def _greedy_action(self, env: TabularPickingEnv, q_table: QTable, state: State) -> str:
        return self._choose_action(env, q_table, state, epsilon=0.0, rng=random.Random(0))

    def _build_route_from_terminal_sequence(
        self,
        strategy: str,
        terminal_sequence: Sequence[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Route:
        node_path, total = self.router.lift_terminal_sequence(terminal_sequence)
        segments = path_to_segments(self.router.G, self.router.node_attrs, node_path)
        movements = path_to_movements(self.router.G, self.router.node_attrs, node_path)
        waypoint_sequence, waypoint_details = self.router._terminal_waypoints(terminal_sequence)
        start_id = terminal_sequence[0] if terminal_sequence else None
        finish_id = terminal_sequence[-1] if terminal_sequence else None
        products = [tid for tid in terminal_sequence[1:-1] if tid in self.router.layout.products]
        route_metadata = {
            "start": start_id,
            "finish": finish_id,
            "product_sequence": products,
            "rl_mode": "terminal_metric_closure",
        }
        if metadata:
            route_metadata.update(metadata)
        return Route(
            strategy=strategy,
            terminal_sequence=list(terminal_sequence),
            waypoint_sequence=list(terminal_sequence),
            waypoint_details=waypoint_details,
            node_path=node_path,
            segments=segments,
            movements=movements,
            total_distance=float(total),
            metadata=route_metadata,
        )

    def train(
        self,
        product_ids: Optional[Sequence[str]] = None,
        start: Optional[str] = None,
        finish: Optional[str] = None,
    ) -> TabularRLResult:
        """Train and return a greedy route from the learned Q-table."""
        cfg = self.config
        if cfg.algorithm.lower() not in {"q_learning", "q-learning", "q", "sarsa"}:
            raise ValueError("algorithm must be 'q_learning' or 'sarsa'")
        if cfg.episodes <= 0:
            raise ValueError("episodes must be positive")
        if not (0.0 <= cfg.epsilon <= 1.0):
            raise ValueError("epsilon must be between 0 and 1")
        if not (0.0 <= cfg.epsilon_min <= 1.0):
            raise ValueError("epsilon_min must be between 0 and 1")

        env = TabularPickingEnv(
            self.router,
            product_ids=product_ids,
            start=start,
            finish=finish,
            finish_bonus=cfg.finish_bonus,
        )
        rng = random.Random(cfg.seed)
        q_default: DefaultDict[State, Dict[str, float]] = defaultdict(dict)
        q_table: QTable = q_default
        episode_rewards: List[float] = []
        episode_distances: List[float] = []
        best_episode_sequence: Optional[List[str]] = None
        best_episode_distance = math.inf
        epsilon = float(cfg.epsilon)
        max_steps = cfg.max_steps or (len(env.product_ids) + 1)
        is_sarsa = cfg.algorithm.lower() == "sarsa"

        for _episode in range(cfg.episodes):
            state = env.reset()
            total_reward = 0.0
            total_distance = 0.0
            done = False
            episode_sequence = [env.start]
            action = self._choose_action(env, q_table, state, epsilon, rng) if is_sarsa else None

            for _step in range(max_steps):
                if action is None:
                    action = self._choose_action(env, q_table, state, epsilon, rng)

                next_state, reward, done, info = env.step(action)
                total_reward += reward
                total_distance += float(info["distance"])
                episode_sequence.append(action)

                legal_next = env.available_actions(next_state) if not done else []
                old = self._q(q_table, state, action)

                if done:
                    target = reward
                    next_action = None
                elif is_sarsa:
                    next_action = self._choose_action(env, q_table, next_state, epsilon, rng)
                    target = reward + cfg.gamma * self._q(q_table, next_state, next_action)
                else:
                    max_next = max((self._q(q_table, next_state, a) for a in legal_next), default=0.0)
                    target = reward + cfg.gamma * max_next
                    next_action = None

                q_table.setdefault(state, {})[action] = old + cfg.alpha * (target - old)
                state = next_state
                action = next_action
                if done:
                    break

            episode_rewards.append(float(total_reward))
            episode_distances.append(float(total_distance))
            if done and total_distance < best_episode_distance:
                best_episode_distance = float(total_distance)
                best_episode_sequence = list(episode_sequence)
            epsilon = max(float(cfg.epsilon_min), epsilon * float(cfg.epsilon_decay))

        # Extract greedy terminal sequence from the trained table.
        state = env.reset()
        terminal_sequence = [env.start]
        done = False
        greedy_distance = 0.0
        for _step in range(max_steps):
            action = self._greedy_action(env, q_table, state)
            next_state, _reward, done, info = env.step(action)
            terminal_sequence.append(action)
            greedy_distance += float(info["distance"])
            state = next_state
            if done:
                break
        if not done:
            raise RuntimeError(
                "The learned greedy policy did not finish within max_steps. "
                "Increase episodes or max_steps, or inspect the Q-table."
            )

        selection = cfg.route_selection.lower().replace("-", "_")
        if selection not in {"best", "greedy", "best_episode"}:
            raise ValueError("route_selection must be 'best', 'greedy', or 'best_episode'")
        if best_episode_sequence is None:
            best_episode_sequence = list(terminal_sequence)
            best_episode_distance = float(greedy_distance)

        if selection == "greedy":
            selected_sequence = list(terminal_sequence)
            selected_source = "greedy_policy"
        elif selection == "best_episode":
            selected_sequence = list(best_episode_sequence)
            selected_source = "best_episode"
        else:
            if best_episode_distance <= greedy_distance:
                selected_sequence = list(best_episode_sequence)
                selected_source = "best_episode"
            else:
                selected_sequence = list(terminal_sequence)
                selected_source = "greedy_policy"

        policy: Dict[State, str] = {}
        # Enumerate visited states from the greedy route plus any states learned
        # during exploration. This is compact but enough for inspection/export.
        for st in q_table.keys():
            actions = env.available_actions(st)
            if actions:
                policy[st] = self._greedy_action(env, q_table, st)

        tail = episode_distances[-min(100, len(episode_distances)):]
        metadata = {
            "algorithm": cfg.algorithm,
            "episodes": int(cfg.episodes),
            "alpha": float(cfg.alpha),
            "gamma": float(cfg.gamma),
            "epsilon_final": float(epsilon),
            "training_best_distance": float(min(episode_distances)) if episode_distances else math.nan,
            "training_last_distance": float(episode_distances[-1]) if episode_distances else math.nan,
            "training_mean_last_100_distance": float(sum(tail) / len(tail)) if tail else math.nan,
            "greedy_distance": float(greedy_distance),
            "best_episode_distance": float(best_episode_distance),
            "selected_route_source": selected_source,
            "route_selection": cfg.route_selection,
        }
        route = self._build_route_from_terminal_sequence(
            "q_learning" if not is_sarsa else "sarsa",
            selected_sequence,
            metadata={"rl": metadata},
        )

        return TabularRLResult(
            route=route,
            q_table=dict(q_table),
            policy=policy,
            episode_rewards=episode_rewards,
            episode_distances=episode_distances,
            config=cfg,
            product_ids=list(env.product_ids),
            start=env.start,
            finish=env.finish,
            metadata=metadata,
        )


def learn_tabular_route(
    router: Any,
    product_ids: Optional[Sequence[str]] = None,
    start: Optional[str] = None,
    finish: Optional[str] = None,
    config: Optional[QLearningConfig] = None,
) -> TabularRLResult:
    """Convenience function for tabular Q-learning route search."""
    return TabularQLearningRouter(router, config=config).train(
        product_ids=product_ids,
        start=start,
        finish=finish,
    )
