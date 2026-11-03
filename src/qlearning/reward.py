"""Thermal-aware reward function module for ThermoPlace Q-Learning placement.

Evaluates the quality of virtual machine (VM) placements considering:
1. Placement Success: Base positive reward for accommodating workloads.
2. Resource Efficiency: Modest incentive for efficient CPU utilization.
3. Host Temperature Penalty: Normalized penalty proportional to thermal elevation.
4. Hotspot Penalties: Strong penalty for triggering or worsening thermal hotspots.
5. Thermal Imbalance Penalty / Benefit: Reward for reducing spatial variance across hosts,
   penalty for increasing thermal divergence.
6. Rejection Penalty: Negative penalty ensuring placement is preferred over rejection.

Mathematical Structure:
    R = R_success - P_temperature - P_hotspot - P_imbalance
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import numpy as np

from src.datacenter import DataCenter, Host


@dataclass
class RewardConfig:
    """Configurable weights and physical thresholds for the reward function.

    Attributes:
        w_success: Base positive reward for successful placement (default: 10.0).
        w_util: Scaling weight for resulting host CPU utilization (default: 1.0).
        w_temp: Weight for normalized host temperature penalty (default: 4.0).
        w_new_hotspot: Substantial penalty for creating a new hotspot (default: 15.0).
        w_hotspot_active: Additional penalty if candidate host is already a hotspot (default: 5.0).
        w_imbalance: Weight for delta in thermal imbalance standard deviation (default: 2.0).
        w_reject: Fixed negative penalty when placement is rejected (default: 10.0).
        ambient_temp_c: Reference ambient CRAH temperature in Celsius (default: 25.0).
        hotspot_threshold_c: Critical temperature threshold defining a hotspot in Celsius (default: 75.0).
        imbalance_scale_c: Reference temperature standard deviation for normalizing imbalance delta (default: 5.0).
    """
    w_success: float = 10.0
    w_util: float = 1.0
    w_temp: float = 4.0
    w_new_hotspot: float = 15.0
    w_hotspot_active: float = 5.0
    w_imbalance: float = 2.0
    w_reject: float = 10.0
    ambient_temp_c: float = 25.0
    hotspot_threshold_c: float = 75.0
    imbalance_scale_c: float = 5.0


@dataclass
class RewardBreakdown:
    """Detailed decomposition of individual reward components for a placement outcome."""
    total_reward: float
    success_reward: float
    temperature_penalty: float
    hotspot_penalty: float
    imbalance_penalty: float
    rejection_penalty: float
    is_successful: bool
    is_new_hotspot: bool
    host_temperature_c: float
    imbalance_delta: float

    def to_dict(self) -> Dict[str, Any]:
        """Converts reward breakdown to a dictionary for logging and telemetry."""
        return asdict(self)


class ThermalRewardCalculator:
    """Calculates modular thermal-aware rewards for VM placement actions."""

    def __init__(self, config: Optional[RewardConfig] = None):
        """Initializes calculator with configurable reward weights.

        Args:
            config: Optional RewardConfig instance. If None, uses default weights.
        """
        self.config = config or RewardConfig()

    def compute_reward(
        self,
        placement_success: bool,
        host_temp_c: float = 25.0,
        host_util: float = 0.0,
        is_new_hotspot: bool = False,
        is_existing_hotspot: bool = False,
        imbalance_before: float = 0.0,
        imbalance_after: float = 0.0,
    ) -> RewardBreakdown:
        """Computes the scalar reward and component breakdown from numerical observations.

        Args:
            placement_success: Whether the VM was successfully placed on an eligible host.
            host_temp_c: Resulting temperature of the placement host in Celsius.
            host_util: Resulting CPU utilization ratio of the host in [0.0, 1.0].
            is_new_hotspot: Whether this placement caused a new host to breach hotspot threshold.
            is_existing_hotspot: Whether the host was already in a hotspot condition.
            imbalance_before: Data center temperature standard deviation before placement (°C).
            imbalance_after: Data center temperature standard deviation after placement (°C).

        Returns:
            RewardBreakdown containing total reward and individual term contributions.
        """
        cfg = self.config

        # 1. Rejection Case: No host allocated
        if not placement_success:
            total = -float(cfg.w_reject)
            return RewardBreakdown(
                total_reward=total,
                success_reward=0.0,
                temperature_penalty=0.0,
                hotspot_penalty=0.0,
                imbalance_penalty=0.0,
                rejection_penalty=float(cfg.w_reject),
                is_successful=False,
                is_new_hotspot=False,
                host_temperature_c=float(host_temp_c),
                imbalance_delta=0.0,
            )

        # 2. Success Reward: Base reward plus efficiency bonus
        util_clamped = min(1.0, max(0.0, float(host_util)))
        r_success = cfg.w_success + (cfg.w_util * util_clamped)

        # 3. Normalized Host Temperature Penalty
        # Normalize relative to the safe thermal operating window [ambient, hotspot]
        safe_range = max(1.0, cfg.hotspot_threshold_c - cfg.ambient_temp_c)
        temp_elevation = max(0.0, host_temp_c - cfg.ambient_temp_c)
        normalized_temp = temp_elevation / safe_range
        p_temp = cfg.w_temp * normalized_temp

        # 4. Hotspot Penalties
        p_hotspot = 0.0
        if is_new_hotspot:
            p_hotspot += cfg.w_new_hotspot
        if is_existing_hotspot or host_temp_c >= cfg.hotspot_threshold_c:
            p_hotspot += cfg.w_hotspot_active

        # 5. Normalized Thermal Imbalance Penalty / Benefit
        # Positive delta (increased imbalance) penalizes; negative delta (improved balance) rewards
        imbalance_delta = float(imbalance_after - imbalance_before)
        imbalance_scale = max(0.1, cfg.imbalance_scale_c)
        normalized_imbalance_delta = imbalance_delta / imbalance_scale
        p_imbalance = cfg.w_imbalance * normalized_imbalance_delta

        # Total Composite Reward: R = R_success - P_temp - P_hotspot - P_imbalance
        total_reward = r_success - p_temp - p_hotspot - p_imbalance

        return RewardBreakdown(
            total_reward=float(total_reward),
            success_reward=float(r_success),
            temperature_penalty=float(p_temp),
            hotspot_penalty=float(p_hotspot),
            imbalance_penalty=float(p_imbalance),
            rejection_penalty=0.0,
            is_successful=True,
            is_new_hotspot=bool(is_new_hotspot),
            host_temperature_c=float(host_temp_c),
            imbalance_delta=float(imbalance_delta),
        )

    def calculate_from_states(
        self,
        placement_success: bool,
        host: Optional[Host] = None,
        hosts_before: Optional[List[Host]] = None,
        hosts_after: Optional[List[Host]] = None,
        datacenter_before: Optional[DataCenter] = None,
        datacenter_after: Optional[DataCenter] = None,
    ) -> RewardBreakdown:
        """Extracts observations from Host and DataCenter snapshots and computes reward.

        Args:
            placement_success: Whether placement succeeded.
            host: Target host where VM was placed (or attempted).
            hosts_before: List of Host instances prior to placement.
            hosts_after: List of Host instances following placement and thermal update.
            datacenter_before: DataCenter instance prior to placement.
            datacenter_after: DataCenter instance following placement.

        Returns:
            RewardBreakdown with detailed numerical outcomes.
        """
        if not placement_success or host is None:
            return self.compute_reward(placement_success=False)

        # Extract host telemetry
        host_temp = getattr(host, "simulated_temperature_c", self.config.ambient_temp_c)
        host_util = getattr(host, "cpu_utilization", 0.0)

        # Extract datacenter host lists
        h_list_before = hosts_before or (datacenter_before.hosts if datacenter_before else [])
        h_list_after = hosts_after or (datacenter_after.hosts if datacenter_after else [])

        # Imbalance calculation
        if h_list_before:
            temps_before = [h.simulated_temperature_c for h in h_list_before]
            imbalance_before = float(np.std(temps_before)) if len(temps_before) > 1 else 0.0
            hotspots_before = sum(1 for t in temps_before if t >= self.config.hotspot_threshold_c)
        else:
            imbalance_before = 0.0
            hotspots_before = 0

        if h_list_after:
            temps_after = [h.simulated_temperature_c for h in h_list_after]
            imbalance_after = float(np.std(temps_after)) if len(temps_after) > 1 else 0.0
            hotspots_after = sum(1 for t in temps_after if t >= self.config.hotspot_threshold_c)
        else:
            imbalance_after = imbalance_before
            hotspots_after = hotspots_before

        is_new_hotspot = (hotspots_after > hotspots_before)
        is_existing_hotspot = (host_temp >= self.config.hotspot_threshold_c and not is_new_hotspot)

        return self.compute_reward(
            placement_success=True,
            host_temp_c=host_temp,
            host_util=host_util,
            is_new_hotspot=is_new_hotspot,
            is_existing_hotspot=is_existing_hotspot,
            imbalance_before=imbalance_before,
            imbalance_after=imbalance_after,
        )
