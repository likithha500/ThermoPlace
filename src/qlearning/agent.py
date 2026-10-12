"""Tabular Q-Learning placement agent interface and scaffold.

NOTE FOR RESEARCH TEAM:
Per project roadmap, Milestone 1 focuses exclusively on validating the First Fit
heuristic baseline, synthetic workload, datacenter structures, thermal modeling,
and telemetry. Full tabular Q-Learning implementation and training are scheduled
for Milestone 2.

Planned Milestone 2 Architecture:
1. State Representation (Discretized Tuple):
   - Discrete bins for host temperature tiers (e.g., Cool < 55°C, Normal 55-70°C, Hot > 70°C).
   - Discrete bins for host CPU utilization tiers (e.g., Low, Medium, High).
   - Arriving VM resource tier (Small, Medium, Large).
2. Action Space:
   - Discrete action index in [0, N_hosts - 1], corresponding to selecting host i.
3. Reward Function:
   - R = - w_temp * (T_host / T_max) - w_hotspot * IsHotspot - w_var * Var(T) + w_succ * Success
   - Heavy penalty for causing or worsening a hotspot.
4. Q-Table Update Rule:
   - Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
"""

from typing import Optional, Any
from src.placement.base import BasePlacementStrategy
from src.datacenter import DataCenter, Host
from src.workload import VM


class QLearningPlacementAgent(BasePlacementStrategy):
    """Tabular Q-Learning agent for thermal-aware VM placement.

    Implementation will be completed in Milestone 2 after First Fit baseline
    verification is concluded.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
    ):
        super().__init__(name="Q-Learning")
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table: dict = {}

    def select_host(self, vm: VM, datacenter: DataCenter) -> Optional[Host]:
        """Placeholder for Q-learning host selection.

        Raises:
            NotImplementedError: As specified for Milestone 1, Q-learning is deferred
            until the First Fit baseline is verified.
        """
        raise NotImplementedError(
            "QLearningPlacementAgent is scheduled for Milestone 2. "
            "Please use FirstFitPlacement for Milestone 1 baseline evaluation."
        )

    def train_step(self, state: Any, action: int, reward: float, next_state: Any) -> None:
        """Placeholder for Q-table Bellman update in Milestone 2."""
        raise NotImplementedError("Q-Learning training is scheduled for Milestone 2.")
