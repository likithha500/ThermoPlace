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
import json
import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from src.placement.base import BasePlacementStrategy
from src.datacenter import DataCenter, Host
from src.workload import VM

logger = logging.getLogger(__name__)


@dataclass
class StateDiscretizerConfig:
    """Configurable bin thresholds for continuous state discretization.

    Attributes:
        vm_cpu_thresholds: Thresholds partitioning VM CPU demand (default: (2.0, 4.0)).
            <= 2 -> 'SMALL', <= 4 -> 'MEDIUM', > 4 -> 'LARGE'.
        vm_ram_thresholds: Thresholds partitioning VM RAM demand in GB (default: (4.0, 16.0)).
            <= 4 -> 'SMALL', <= 16 -> 'MEDIUM', > 16 -> 'LARGE'.
        host_cpu_thresholds: Thresholds partitioning host CPU utilization ratio in [0, 1] (default: (0.40, 0.70)).
            < 0.40 -> 'LOW', < 0.70 -> 'MEDIUM', >= 0.70 -> 'HIGH'.
        host_ram_thresholds: Thresholds partitioning host RAM utilization ratio in [0, 1] (default: (0.40, 0.70)).
            < 0.40 -> 'LOW', < 0.70 -> 'MEDIUM', >= 0.70 -> 'HIGH'.
        host_disk_thresholds: Thresholds partitioning host disk utilization ratio in [0, 1] (default: (0.40, 0.70)).
            < 0.40 -> 'LOW', < 0.70 -> 'MEDIUM', >= 0.70 -> 'HIGH'.
        host_temp_thresholds: Thresholds partitioning host temperature in °C (default: (55.0, 65.0, 75.0)).
            < 55.0 -> 'COOL', < 65.0 -> 'NORMAL', < 75.0 -> 'WARM', >= 75.0 -> 'HOT'.
        hotspot_threshold_c: Temperature threshold defining critical hotspots in °C (default: 75.0).
        warm_threshold_c: Temperature threshold defining warm state in °C (default: 65.0).
    """
    vm_cpu_thresholds: Tuple[float, float] = (2.0, 4.0)
    vm_ram_thresholds: Tuple[float, float] = (4.0, 16.0)
    host_cpu_thresholds: Tuple[float, float] = (0.40, 0.70)
    host_ram_thresholds: Tuple[float, float] = (0.40, 0.70)
    host_disk_thresholds: Tuple[float, float] = (0.40, 0.70)
    host_temp_thresholds: Tuple[float, float, float] = (55.0, 65.0, 75.0)
    hotspot_threshold_c: float = 75.0
    warm_threshold_c: float = 65.0


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
        feature_set: str = "cpu_ram",
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
            feature_set: Resource feature set ('cpu_ram', 'cpu_ram_disk', etc.; default: 'cpu_ram').
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
        self.feature_set = feature_set

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
        host_disk: Optional[float] = None,
        host_temp: Optional[float] = None,
        dc_thermal: Optional[str] = None,
        feature_set: Optional[str] = None,
    ) -> Tuple:
        """Discretizes continuous placement features into categorical state bins.

        Accepts either entity objects (VM, Host, DataCenter) or explicit numeric values.

        Features:
        1. Incoming VM CPU requirement ('SMALL', 'MEDIUM', 'LARGE')
        2. Incoming VM RAM requirement ('SMALL', 'MEDIUM', 'LARGE')
        3. Candidate host CPU utilization ('LOW', 'MEDIUM', 'HIGH')
        4. Candidate host RAM utilization ('LOW', 'MEDIUM', 'HIGH')
        5. (Optional, if disk enabled) Candidate host disk utilization ('LOW', 'MEDIUM', 'HIGH')
        6. Candidate host simulated temperature ('COOL', 'NORMAL', 'HOT')
        7. Overall datacenter thermal condition ('COOL', 'NORMAL', 'WARM', 'CRITICAL')

        Returns:
            6-tuple of strings for 'cpu_ram', or 7-tuple when disk features are enabled.
        """
        cfg = self.state_config
        active_fs = feature_set if feature_set is not None else self.feature_set

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

        # 5. Extract Candidate Host Temperature (4 bins: COOL, NORMAL, WARM, HOT)
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
        elif host_temp < cfg.host_temp_thresholds[2]:
            bin_host_temp = "WARM"
        else:
            bin_host_temp = "HOT"

        # 6. Extract Overall Datacenter Thermal Condition (3 bins: COOL, MODERATE, CRITICAL)
        if dc_thermal is not None:
            bin_dc_thermal = str(dc_thermal)
        elif datacenter is not None and datacenter.hosts:
            host_temps = [h.simulated_temperature_c for h in datacenter.hosts]
            max_t = max(host_temps) if host_temps else 25.0

            if max_t >= cfg.hotspot_threshold_c:
                bin_dc_thermal = "CRITICAL"
            elif max_t >= cfg.warm_threshold_c:
                bin_dc_thermal = "MODERATE"
            else:
                bin_dc_thermal = "COOL"
        else:
            bin_dc_thermal = "COOL"

        if active_fs == "cpu_ram":
            return (bin_vm_cpu, bin_vm_ram, bin_host_cpu, bin_host_ram, bin_host_temp, bin_dc_thermal)

        # 7. Extract Candidate Host Disk utilization for disk-aware features
        if host_disk is None:
            if isinstance(host, Host):
                if active_fs == "cpu_ram_disk_read":
                    host_disk = float(host.disk_read_utilization)
                elif active_fs == "cpu_ram_disk_write":
                    host_disk = float(host.disk_write_utilization)
                else:
                    host_disk = float(host.disk_io_utilization)
            elif isinstance(host, (int, float)):
                host_disk = float(host)
            else:
                host_disk = 0.0

        host_disk = min(1.0, max(0.0, float(host_disk)))
        if host_disk < cfg.host_disk_thresholds[0]:
            bin_host_disk = "LOW"
        elif host_disk < cfg.host_disk_thresholds[1]:
            bin_host_disk = "MEDIUM"
        else:
            bin_host_disk = "HIGH"

        return (bin_vm_cpu, bin_vm_ram, bin_host_cpu, bin_host_ram, bin_host_disk, bin_host_temp, bin_dc_thermal)

    # -------------------------------------------------------------------------
    # 2. Eligible-Host Identification
    # -------------------------------------------------------------------------

    def get_eligible_hosts(self, vm: VM, datacenter: DataCenter) -> List[Host]:
        """Identifies physical hosts with sufficient capacity across enabled resources.

        Args:
            vm: Virtual Machine requesting placement.
            datacenter: Physical data center topology and allocation state.

        Returns:
            List of Host instances capable of accommodating the VM.
        """
        return [
            host for host in datacenter.hosts
            if host.can_accommodate(vm, feature_set=self.feature_set)
        ]

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

    # -------------------------------------------------------------------------
    # 7. Model Persistence (Save / Load)
    # -------------------------------------------------------------------------

    # @flows QLearningPlacementAgent -> FileSystem via JSON -- "Saves trained Q-table and metadata to disk"
    # @handles model_weights on QLearningPlacementAgent -- "Serializes discrete state-action Q-table safely"
    def save(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Saves the trained Q-table, state configuration, and hyperparameters to a JSON file.

        Args:
            file_path: Output file path (e.g. 'models/qlearning/q_table.json').
            metadata: Optional additional training metadata or context dict.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize Q-table non-zero entries
        entries = []
        for (state, action), q_val in sorted(self.q_table.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
            if abs(q_val) > 1e-9:
                entries.append({
                    "state": list(state),
                    "action": str(action),
                    "q_value": float(q_val),
                })

        cfg = self.state_config
        state_cfg_dict = {
            "vm_cpu_thresholds": list(cfg.vm_cpu_thresholds),
            "vm_ram_thresholds": list(cfg.vm_ram_thresholds),
            "host_cpu_thresholds": list(cfg.host_cpu_thresholds),
            "host_ram_thresholds": list(cfg.host_ram_thresholds),
            "host_disk_thresholds": list(cfg.host_disk_thresholds),
            "host_temp_thresholds": list(cfg.host_temp_thresholds),
            "hotspot_threshold_c": float(cfg.hotspot_threshold_c),
            "warm_threshold_c": float(cfg.warm_threshold_c),
        }

        hyperparameters = {
            "learning_rate": float(self.learning_rate),
            "discount_factor": float(self.discount_factor),
            "epsilon": float(self.epsilon),
            "min_epsilon": float(self.min_epsilon),
            "epsilon_decay": float(self.epsilon_decay),
            "seed": self.seed,
        }

        payload = {
            "version": "1.0",
            "algorithm": "Q-Learning",
            "feature_set": self.feature_set,
            "hyperparameters": hyperparameters,
            "state_config": state_cfg_dict,
            "training_metadata": metadata or {},
            "num_entries": len(entries),
            "entries": entries,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    # @flows FileSystem -> QLearningPlacementAgent via JSON -- "Loads trained Q-table and validates schema"
    # @handles model_weights on QLearningPlacementAgent -- "Deserializes discrete state-action Q-table safely"
    # @validates #qtable-schema for QLearningPlacementAgent -- "Validates state configuration thresholds before deserialization"
    def load(
        self,
        file_path: Union[str, Path],
        validate_config: bool = True,
        restore_hyperparameters: bool = False,
    ) -> None:
        """Loads a saved Q-table from a JSON file into this agent.

        Args:
            file_path: Path to the JSON model file.
            validate_config: If True, checks that the saved state discretization
                thresholds match this agent's state_config, raising ValueError on mismatch.
            restore_hyperparameters: If True, also updates learning_rate, discount_factor,
                epsilon, min_epsilon, epsilon_decay, and seed from the file.

        Raises:
            FileNotFoundError: If the specified model file does not exist.
            ValueError: If the file format is invalid or state configurations are incompatible.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Q-table model file not found: '{file_path}'")

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, dict) or "entries" not in payload:
            raise ValueError(f"Invalid Q-table model file format in '{file_path}'. Missing 'entries'.")

        saved_features = payload.get("feature_set", "cpu_ram")
        if validate_config and saved_features != self.feature_set:
            raise ValueError(
                f"Incompatible feature set in '{file_path}': "
                f"saved='{saved_features}', agent='{self.feature_set}'"
            )

        if validate_config and "state_config" in payload:
            saved_sc = payload["state_config"]
            cfg = self.state_config
            mismatches = []
            if tuple(saved_sc.get("vm_cpu_thresholds", [])) != tuple(cfg.vm_cpu_thresholds):
                mismatches.append(
                    f"vm_cpu_thresholds (saved: {saved_sc.get('vm_cpu_thresholds')}, current: {cfg.vm_cpu_thresholds})"
                )
            if tuple(saved_sc.get("vm_ram_thresholds", [])) != tuple(cfg.vm_ram_thresholds):
                mismatches.append(
                    f"vm_ram_thresholds (saved: {saved_sc.get('vm_ram_thresholds')}, current: {cfg.vm_ram_thresholds})"
                )
            if tuple(saved_sc.get("host_cpu_thresholds", [])) != tuple(cfg.host_cpu_thresholds):
                mismatches.append(
                    f"host_cpu_thresholds (saved: {saved_sc.get('host_cpu_thresholds')}, current: {cfg.host_cpu_thresholds})"
                )
            if tuple(saved_sc.get("host_ram_thresholds", [])) != tuple(cfg.host_ram_thresholds):
                mismatches.append(
                    f"host_ram_thresholds (saved: {saved_sc.get('host_ram_thresholds')}, current: {cfg.host_ram_thresholds})"
                )
            if self.feature_set != "cpu_ram" and "host_disk_thresholds" in saved_sc:
                if tuple(saved_sc.get("host_disk_thresholds", [])) != tuple(cfg.host_disk_thresholds):
                    mismatches.append(
                        f"host_disk_thresholds (saved: {saved_sc.get('host_disk_thresholds')}, current: {cfg.host_disk_thresholds})"
                    )
            if tuple(saved_sc.get("host_temp_thresholds", [])) != tuple(cfg.host_temp_thresholds):
                mismatches.append(
                    f"host_temp_thresholds (saved: {saved_sc.get('host_temp_thresholds')}, current: {cfg.host_temp_thresholds})"
                )
            if float(saved_sc.get("hotspot_threshold_c", -1)) != float(cfg.hotspot_threshold_c):
                mismatches.append(
                    f"hotspot_threshold_c (saved: {saved_sc.get('hotspot_threshold_c')}, current: {cfg.hotspot_threshold_c})"
                )

            if mismatches:
                raise ValueError(
                    f"Incompatible state configuration in '{file_path}':\n  " + "\n  ".join(mismatches)
                )

        if restore_hyperparameters and "hyperparameters" in payload:
            hp = payload["hyperparameters"]
            self.learning_rate = hp.get("learning_rate", self.learning_rate)
            self.discount_factor = hp.get("discount_factor", self.discount_factor)
            self.epsilon = hp.get("epsilon", self.epsilon)
            self.min_epsilon = hp.get("min_epsilon", self.min_epsilon)
            self.epsilon_decay = hp.get("epsilon_decay", self.epsilon_decay)
            if "seed" in hp:
                self.seed = hp["seed"]
                if self.seed is not None:
                    self.rng = random.Random(self.seed)

        self.q_table.clear()
        for entry in payload["entries"]:
            state = tuple(entry["state"])
            action = entry["action"]
            q_val = float(entry["q_value"])
            self.q_table[(state, action)] = q_val

    @classmethod
    def load_from_file(
        cls,
        file_path: Union[str, Path],
        validate_config: bool = True,
        feature_set: Optional[str] = None,
    ) -> "QLearningPlacementAgent":
        """Instantiates a new QLearningPlacementAgent directly from a saved JSON model file.

        Args:
            file_path: Path to the JSON model file.
            validate_config: Kept for API symmetry.
            feature_set: Optional expected feature set for compatibility check.

        Returns:
            Configured QLearningPlacementAgent instance with loaded Q-table.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Q-table model file not found: '{file_path}'")

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        saved_features = payload.get("feature_set", "cpu_ram")
        if feature_set is not None and validate_config and feature_set != saved_features:
            raise ValueError(
                f"Incompatible feature set: expected '{feature_set}', model has '{saved_features}'"
            )

        state_cfg = StateDiscretizerConfig()
        if "state_config" in payload:
            sc = payload["state_config"]
            state_cfg = StateDiscretizerConfig(
                vm_cpu_thresholds=tuple(sc.get("vm_cpu_thresholds", (2.0, 4.0))),
                vm_ram_thresholds=tuple(sc.get("vm_ram_thresholds", (4.0, 16.0))),
                host_cpu_thresholds=tuple(sc.get("host_cpu_thresholds", (0.40, 0.70))),
                host_ram_thresholds=tuple(sc.get("host_ram_thresholds", (0.40, 0.70))),
                host_disk_thresholds=tuple(sc.get("host_disk_thresholds", (0.40, 0.70))),
                host_temp_thresholds=tuple(sc.get("host_temp_thresholds", (55.0, 65.0, 75.0))),
                hotspot_threshold_c=float(sc.get("hotspot_threshold_c", 75.0)),
                warm_threshold_c=float(sc.get("warm_threshold_c", 65.0)),
            )

        hp = payload.get("hyperparameters", {})
        agent = cls(
            learning_rate=hp.get("learning_rate", 0.1),
            discount_factor=hp.get("discount_factor", 0.95),
            epsilon=hp.get("epsilon", 1.0),
            min_epsilon=hp.get("min_epsilon", 0.05),
            epsilon_decay=hp.get("epsilon_decay", 0.995),
            seed=hp.get("seed", None),
            state_config=state_cfg,
            feature_set=saved_features,
        )

        agent.q_table.clear()
        for entry in payload.get("entries", []):
            state = tuple(entry["state"])
            action = entry["action"]
            q_val = float(entry["q_value"])
            agent.q_table[(state, action)] = q_val

        return agent
