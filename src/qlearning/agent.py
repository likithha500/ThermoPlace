"""Tabular Q-Learning placement agent for intelligent VM scheduling.

Implements a tabular reinforcement learning agent for thermal-aware and
resource-constrained virtual machine (VM) placement in data centers.

State Representation:
- Incoming VM CPU requirement (discretized).
- Incoming VM RAM requirement (discretized).
- Candidate host CPU utilization (discretized).
- Candidate host RAM utilization (discretized).
- Candidate host simulated temperature (discretized).
- Overall data center thermal condition (discretized).

Action Space:
- Each eligible physical host in the data center constitutes an action.
- Only hosts satisfying CPU and RAM capacity constraints are eligible.

Q-Learning Update Rule:
    Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
    For terminal transitions:
    Q(s, a) <- Q(s, a) + alpha * [r - Q(s, a)]
"""

from dataclasses import dataclass
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from src.placement.base import BasePlacementStrategy
from src.datacenter import DataCenter, Host
from src.workload import VM


@dataclass
class StateDiscretizerConfig:
    """Configurable bin thresholds for continuous state discretization.

    Attributes:
        vm_cpu_thresholds: Thresholds partitioning VM CPU demand (default: (2.0, 4.0)).
            <= 2 -> 'SMALL', <= 4 -> 'MEDIUM', > 4 -> 'LARGE'.
        vm_ram_thresholds: Thresholds partitioning VM RAM demand in GB (default: (4.0, 16.0)).
            <= 4 -> 'SMALL', <= 16 -> 'MEDIUM', > 16 -> 'LARGE'.
        host_cpu_thresholds: Thresholds partitioning host CPU utilization ratio in [0, 1] (default: (0.35, 0.70)).
            < 0.35 -> 'LOW', < 0.70 -> 'MEDIUM', >= 0.70 -> 'HIGH'.
        host_ram_thresholds: Thresholds partitioning host RAM utilization ratio in [0, 1] (default: (0.35, 0.70)).
            < 0.35 -> 'LOW', < 0.70 -> 'MEDIUM', >= 0.70 -> 'HIGH'.
        host_temp_thresholds: Thresholds partitioning host temperature in °C (default: (55.0, 70.0)).
            < 55.0 -> 'COOL', < 70.0 -> 'NORMAL', >= 70.0 -> 'HOT'.
        hotspot_threshold_c: Temperature threshold defining critical hotspots in °C (default: 75.0).
        dc_warm_threshold_c: Average datacenter temperature threshold for warm state (default: 55.0).
    """
    vm_cpu_thresholds: Tuple[float, float] = (2.0, 4.0)
    vm_ram_thresholds: Tuple[float, float] = (4.0, 16.0)
    host_cpu_thresholds: Tuple[float, float] = (0.35, 0.70)
    host_ram_thresholds: Tuple[float, float] = (0.35, 0.70)
    host_temp_thresholds: Tuple[float, float] = (55.0, 70.0)
    hotspot_threshold_c: float = 75.0
    dc_warm_threshold_c: float = 55.0


class QTable(dict):
    """Sparse dictionary-based Q-table for state-action values.

    Defaults unseen state-action pairs to Q-value 0.0. Supports normalized
    lookup for Host instances and host identifier keys.
    """

    def __missing__(self, key: Any) -> float:
        return 0.0

    def __getitem__(self, key: Any) -> float:
        norm_key = self._normalize_key(key)
        if norm_key in self:
            return super().__getitem__(norm_key)
        return 0.0

    def __setitem__(self, key: Any, value: float) -> None:
        norm_key = self._normalize_key(key)
        super().__setitem__(norm_key, float(value))

    def get(self, key: Any, default: float = 0.0) -> float:
        norm_key = self._normalize_key(key)
        return super().get(norm_key, default)

    @staticmethod
    def _normalize_key(key: Any) -> Any:
        if isinstance(key, tuple) and len(key) == 2:
            state, action = key
            action_id = action.host_id if hasattr(action, "host_id") else action
            return (state, action_id)
        return key


class QLearningPlacementAgent(BasePlacementStrategy):
    """Tabular Q-Learning agent for intelligent, thermal-aware VM placement."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        min_epsilon: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None,
        state_config: Optional[StateDiscretizerConfig] = None,
        name: str = "Q-Learning",
    ):
        """Initializes the Q-Learning placement agent with hyperparameters.

        Args:
            learning_rate: Step size alpha in [0, 1] for value updates (default: 0.1).
            discount_factor: Discount factor gamma in [0, 1] for future rewards (default: 0.95).
            epsilon: Initial exploration rate epsilon in [0, 1] (default: 1.0).
            min_epsilon: Lower bound for epsilon after decay (default: 0.05).
            epsilon_decay: Multiplicative decay factor applied per decay step (default: 0.995).
            seed: Optional integer seed for pseudo-random number generation.
            state_config: Optional discretization thresholds configuration.
            name: Human-readable name for placement strategy reports (default: 'Q-Learning').
        """
        super().__init__(name=name)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        self.rng = random.Random(seed)
        self.state_config = state_config or StateDiscretizerConfig()
        self.q_table: QTable = QTable()

    # -------------------------------------------------------------------------
    # 1. State Discretization
    # -------------------------------------------------------------------------

    def discretize_state(
        self,
        vm: Optional[Union[VM, int, float]] = None,
        host: Optional[Union[Host, float]] = None,
        datacenter: Optional[DataCenter] = None,
        *,
        vm_cpu: Optional[float] = None,
        vm_ram: Optional[float] = None,
        host_cpu: Optional[float] = None,
        host_ram: Optional[float] = None,
        host_temp: Optional[float] = None,
        dc_thermal: Optional[str] = None,
    ) -> Tuple[str, str, str, str, str, str]:
        """Discretizes continuous placement features into categorical state bins.

        Accepts either entity objects (VM, Host, DataCenter) or explicit numeric values.

        Features:
        1. Incoming VM CPU requirement ('SMALL', 'MEDIUM', 'LARGE')
        2. Incoming VM RAM requirement ('SMALL', 'MEDIUM', 'LARGE')
        3. Candidate host CPU utilization ('LOW', 'MEDIUM', 'HIGH')
        4. Candidate host RAM utilization ('LOW', 'MEDIUM', 'HIGH')
        5. Candidate host simulated temperature ('COOL', 'NORMAL', 'HOT')
        6. Overall datacenter thermal condition ('COOL', 'NORMAL', 'WARM', 'CRITICAL')

        Returns:
            6-tuple of strings representing the discrete placement state.
        """
        cfg = self.state_config

        # 1. Extract VM CPU requirement
        if vm_cpu is None:
            if isinstance(vm, VM):
                vm_cpu = float(vm.cpu_demand)
            elif isinstance(vm, (int, float)):
                vm_cpu = float(vm)
            else:
                vm_cpu = 0.0

        if vm_cpu <= cfg.vm_cpu_thresholds[0]:
            bin_vm_cpu = "SMALL"
        elif vm_cpu <= cfg.vm_cpu_thresholds[1]:
            bin_vm_cpu = "MEDIUM"
        else:
            bin_vm_cpu = "LARGE"

        # 2. Extract VM RAM requirement
        if vm_ram is None:
            if isinstance(vm, VM):
                vm_ram = float(vm.ram_demand)
            elif isinstance(vm, (int, float)):
                vm_ram = float(vm)
            else:
                vm_ram = 0.0

        if vm_ram <= cfg.vm_ram_thresholds[0]:
            bin_vm_ram = "SMALL"
        elif vm_ram <= cfg.vm_ram_thresholds[1]:
            bin_vm_ram = "MEDIUM"
        else:
            bin_vm_ram = "LARGE"

        # 3. Extract Candidate Host CPU utilization
        if host_cpu is None:
            if isinstance(host, Host):
                host_cpu = float(host.cpu_utilization)
            elif isinstance(host, (int, float)):
                host_cpu = float(host)
            else:
                host_cpu = 0.0

        if host_cpu < cfg.host_cpu_thresholds[0]:
            bin_host_cpu = "LOW"
        elif host_cpu < cfg.host_cpu_thresholds[1]:
            bin_host_cpu = "MEDIUM"
        else:
            bin_host_cpu = "HIGH"

        # 4. Extract Candidate Host RAM utilization
        if host_ram is None:
            if isinstance(host, Host):
                host_ram = float(host.ram_utilization)
            elif isinstance(host, (int, float)):
                host_ram = float(host)
            else:
                host_ram = 0.0

        if host_ram < cfg.host_ram_thresholds[0]:
            bin_host_ram = "LOW"
        elif host_ram < cfg.host_ram_thresholds[1]:
            bin_host_ram = "MEDIUM"
        else:
            bin_host_ram = "HIGH"

        # 5. Extract Candidate Host Temperature
        if host_temp is None:
            if isinstance(host, Host):
                host_temp = float(host.simulated_temperature_c)
            elif isinstance(host, (int, float)):
                host_temp = float(host)
            else:
                host_temp = 25.0

        if host_temp < cfg.host_temp_thresholds[0]:
            bin_host_temp = "COOL"
        elif host_temp < cfg.host_temp_thresholds[1]:
            bin_host_temp = "NORMAL"
        else:
            bin_host_temp = "HOT"

        # 6. Extract Overall Datacenter Thermal Condition (if available)
        if dc_thermal is not None:
            bin_dc_thermal = str(dc_thermal)
        elif datacenter is not None and datacenter.hosts:
            host_temps = [h.simulated_temperature_c for h in datacenter.hosts]
            max_t = max(host_temps) if host_temps else 25.0
            avg_t = sum(host_temps) / len(host_temps) if host_temps else 25.0

            if max_t >= cfg.hotspot_threshold_c:
                bin_dc_thermal = "CRITICAL"
            elif avg_t >= cfg.dc_warm_threshold_c or max_t >= cfg.host_temp_thresholds[1]:
                bin_dc_thermal = "WARM"
            elif avg_t < cfg.host_temp_thresholds[0]:
                bin_dc_thermal = "COOL"
            else:
                bin_dc_thermal = "NORMAL"
        else:
            bin_dc_thermal = "NORMAL"

        return (bin_vm_cpu, bin_vm_ram, bin_host_cpu, bin_host_ram, bin_host_temp, bin_dc_thermal)

    # -------------------------------------------------------------------------
    # 2. Eligible-Host Identification
    # -------------------------------------------------------------------------

    def get_eligible_hosts(self, vm: VM, datacenter: DataCenter) -> List[Host]:
        """Identifies physical hosts with sufficient CPU and RAM capacity.

        Args:
            vm: Virtual Machine requesting placement.
            datacenter: Physical data center topology and allocation state.

        Returns:
            List of Host instances capable of accommodating the VM.
        """
        return [host for host in datacenter.hosts if host.can_accommodate(vm)]

    # -------------------------------------------------------------------------
    # 3. Q-Value Lookup & Modification
    # -------------------------------------------------------------------------

    def get_q_value(self, state: Any, action: Any) -> float:
        """Retrieves the Q-value for a state-action pair, defaulting to 0.0 for unseen pairs.

        Args:
            state: Discretized state representation.
            action: Selected action (e.g. host_id string or Host instance).

        Returns:
            Current numeric Q-value Q(s, a).
        """
        action_key = action.host_id if hasattr(action, "host_id") else action
        return self.q_table[(state, action_key)]

    def set_q_value(self, state: Any, action: Any, value: float) -> None:
        """Sets the Q-value for a state-action pair.

        Args:
            state: Discretized state representation.
            action: Action identifier or Host instance.
            value: New numeric Q-value to store.
        """
        action_key = action.host_id if hasattr(action, "host_id") else action
        self.q_table[(state, action_key)] = float(value)

    # -------------------------------------------------------------------------
    # 4. Action Selection (Epsilon-Greedy Policy)
    # -------------------------------------------------------------------------

    def select_action(self, state: Any, eligible_actions: List[Any]) -> Optional[Any]:
        """Selects an action from candidate eligible actions using epsilon-greedy policy.

        Args:
            state: Discretized state representation.
            eligible_actions: List of valid candidate actions (e.g. host IDs).

        Returns:
            Selected action, or None if eligible_actions is empty.
        """
        if not eligible_actions:
            return None

        # Exploration: choose random eligible action
        if self.rng.random() < self.epsilon:
            return self.rng.choice(eligible_actions)

        # Exploitation (greedy): choose eligible action with highest Q-value
        best_action = eligible_actions[0]
        best_q = self.get_q_value(state, best_action)

        for action in eligible_actions[1:]:
            q = self.get_q_value(state, action)
            if q > best_q:
                best_q = q
                best_action = action

        return best_action

    def select_host(self, vm: VM, datacenter: DataCenter) -> Optional[Host]:
        """Selects a host machine for VM placement under capacity constraints.

        Implements BasePlacementStrategy.select_host(). Considers only eligible hosts
        satisfying both CPU and RAM capacity. Uses epsilon-greedy policy.

        Args:
            vm: Virtual Machine requesting placement.
            datacenter: Current state of the data center.

        Returns:
            The selected Host instance if capacity allows, or None if rejected.
        """
        eligible_hosts = self.get_eligible_hosts(vm, datacenter)
        if not eligible_hosts:
            return None

        # Exploration: pick a random eligible host
        if self.rng.random() < self.epsilon:
            return self.rng.choice(eligible_hosts)

        # Exploitation (greedy): select eligible host with the highest Q-value
        best_host = eligible_hosts[0]
        state_first = self.discretize_state(vm, best_host, datacenter)
        best_q = self.get_q_value(state_first, best_host.host_id)

        for host in eligible_hosts[1:]:
            state_host = self.discretize_state(vm, host, datacenter)
            q = self.get_q_value(state_host, host.host_id)
            if q > best_q:
                best_q = q
                best_host = host

        return best_host

    # -------------------------------------------------------------------------
    # 5. Q-Learning Update
    # -------------------------------------------------------------------------

    def update(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Optional[Any] = None,
        terminal: bool = False,
        next_eligible_actions: Optional[List[Any]] = None,
    ) -> float:
        """Executes standard tabular Q-learning Bellman update:

            Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]

        For terminal transitions (or when next_state is None), the future Q term
        is not included:
            Q(s, a) <- Q(s, a) + alpha * [r - Q(s, a)]

        Args:
            state: Current state s.
            action: Selected action a.
            reward: Immediate scalar reward r received.
            next_state: Resulting state s' (optional).
            terminal: Whether this transition leads to a terminal state (default: False).
            next_eligible_actions: Optional list of candidate actions available in s'.

        Returns:
            The updated Q-value Q(s, a).
        """
        action_key = action.host_id if hasattr(action, "host_id") else action
        current_q = self.get_q_value(state, action_key)

        if terminal or next_state is None:
            max_future_q = 0.0
        elif next_eligible_actions is not None:
            if len(next_eligible_actions) > 0:
                max_future_q = max(
                    self.get_q_value(next_state, a_next)
                    for a_next in next_eligible_actions
                )
            else:
                max_future_q = 0.0
        else:
            # Query known actions in Q-table for next_state
            known_qs = [
                val for (s_k, _), val in self.q_table.items()
                if s_k == next_state
            ]
            max_future_q = max(known_qs) if known_qs else 0.0

        td_target = reward + (0.0 if terminal else self.discount_factor * max_future_q)
        td_error = td_target - current_q
        new_q = current_q + self.learning_rate * td_error

        self.set_q_value(state, action_key, new_q)
        return new_q

    def train_step(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Optional[Any] = None,
        done: bool = False,
        next_actions: Optional[List[Any]] = None,
    ) -> float:
        """Executes a training step. Alias for update() preserving interface compatibility."""
        return self.update(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            terminal=done,
            next_eligible_actions=next_actions,
        )

    # -------------------------------------------------------------------------
    # 6. Exploration Decay & Maintenance
    # -------------------------------------------------------------------------

    def decay_epsilon(self) -> float:
        """Decays exploration parameter epsilon down to min_epsilon.

        Equation:
            epsilon <- max(min_epsilon, epsilon * epsilon_decay)

        Returns:
            The new epsilon value.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def reset_q_table(self) -> None:
        """Clears all stored entries in the Q-table."""
        self.q_table.clear()
