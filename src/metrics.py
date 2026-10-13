"""Metric collection and telemetry module for ThermoPlace.

Collects, aggregates, and summarizes thermal and resource metrics across simulation
steps:
- Hotspot Count: Number of hosts whose simulated temperature exceeds the safety threshold.
- Maximum Temperature: Highest simulated host temperature across all hosts at each step.
- Thermal Imbalance: Standard deviation and range (max - min) of temperatures across hosts.
- Resource Utilization: Mean CPU and RAM utilization across the data center.
- Placement Success & Rejection: Tracking capacity bottlenecks.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from src.datacenter import DataCenter
from src.thermal import ThermalModel


@dataclass
class StepSnapshot:
    """Telemetry snapshot captured at a single simulation time step."""
    time_step: int
    num_vms_arrived: int
    num_vms_placed: int
    num_vms_rejected: int
    active_vm_count: int
    mean_cpu_utilization: float
    mean_ram_utilization: float
    max_temperature_c: float
    min_temperature_c: float
    mean_temperature_c: float
    thermal_imbalance_std: float
    thermal_imbalance_range: float
    hotspot_count: int
    total_power_w: float
    host_temperatures: Dict[str, float] = field(default_factory=dict)
    mean_actual_cpu_utilization: float = 0.0


class MetricsCollector:
    """Tracks and records simulation telemetry at each time step."""

    def __init__(self, algorithm_name: str, thermal_model: ThermalModel):
        self.algorithm_name = algorithm_name
        self.thermal_model = thermal_model
        self.snapshots: List[StepSnapshot] = []
        self.total_arrived_vms: int = 0
        self.total_placed_vms: int = 0
        self.total_rejected_vms: int = 0

    def record_step(
        self,
        time_step: int,
        datacenter: DataCenter,
        vms_arrived_this_step: int,
        vms_placed_this_step: int,
        vms_rejected_this_step: int,
    ) -> StepSnapshot:
        """Captures telemetry for the current simulation time step.

        Args:
            time_step: Current discrete time step.
            datacenter: Current state of the data center.
            vms_arrived_this_step: Count of VMs that arrived at this step.
            vms_placed_this_step: Count of VMs successfully allocated at this step.
            vms_rejected_this_step: Count of VMs rejected due to capacity limits.

        Returns:
            The recorded StepSnapshot.
        """
        self.total_arrived_vms += vms_arrived_this_step
        self.total_placed_vms += vms_placed_this_step
        self.total_rejected_vms += vms_rejected_this_step

        hosts = datacenter.hosts
        temps = np.array([h.simulated_temperature_c for h in hosts], dtype=float)
        cpu_utils = np.array([h.cpu_utilization for h in hosts], dtype=float)
        actual_cpu_utils = np.array([h.actual_cpu_utilization for h in hosts], dtype=float)
        ram_utils = np.array([h.ram_utilization for h in hosts], dtype=float)

        total_active_vms = sum(len(h.allocated_vms) for h in hosts)
        total_power = sum(self.thermal_model.estimate_host_power_w(h) for h in hosts)

        max_temp = float(np.max(temps))
        min_temp = float(np.min(temps))
        mean_temp = float(np.mean(temps))
        std_temp = float(np.std(temps))
        range_temp = float(max_temp - min_temp)

        hotspots = sum(1 for h in hosts if self.thermal_model.is_hotspot(h))

        temp_dict = {h.host_id: h.simulated_temperature_c for h in hosts}

        snapshot = StepSnapshot(
            time_step=time_step,
            num_vms_arrived=vms_arrived_this_step,
            num_vms_placed=vms_placed_this_step,
            num_vms_rejected=vms_rejected_this_step,
            active_vm_count=total_active_vms,
            mean_cpu_utilization=float(np.mean(cpu_utils)),
            mean_ram_utilization=float(np.mean(ram_utils)),
            max_temperature_c=max_temp,
            min_temperature_c=min_temp,
            mean_temperature_c=mean_temp,
            thermal_imbalance_std=std_temp,
            thermal_imbalance_range=range_temp,
            hotspot_count=hotspots,
            total_power_w=float(total_power),
            host_temperatures=temp_dict,
            mean_actual_cpu_utilization=float(np.mean(actual_cpu_utils)),
        )

        self.snapshots.append(snapshot)
        return snapshot

    def to_dataframe(self) -> pd.DataFrame:
        """Converts collected step snapshots into a pandas DataFrame."""
        rows = []
        for s in self.snapshots:
            row = {
                "time_step": s.time_step,
                "vms_arrived": s.num_vms_arrived,
                "vms_placed": s.num_vms_placed,
                "vms_rejected": s.num_vms_rejected,
                "active_vms": s.active_vm_count,
                "mean_cpu_util": s.mean_cpu_utilization,
                "mean_actual_cpu_util": s.mean_actual_cpu_utilization,
                "mean_ram_util": s.mean_ram_utilization,
                "max_temp_c": s.max_temperature_c,
                "min_temp_c": s.min_temperature_c,
                "mean_temp_c": s.mean_temperature_c,
                "thermal_imbalance_std": s.thermal_imbalance_std,
                "thermal_imbalance_range": s.thermal_imbalance_range,
                "hotspot_count": s.hotspot_count,
                "total_power_w": s.total_power_w,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> Dict[str, Any]:
        """Calculates aggregate summary statistics across the entire simulation run."""
        if not self.snapshots:
            return {"status": "No data collected"}

        df = self.to_dataframe()
        return {
            "algorithm": self.algorithm_name,
            "total_steps": len(self.snapshots),
            "total_arrived_vms": self.total_arrived_vms,
            "total_placed_vms": self.total_placed_vms,
            "total_rejected_vms": self.total_rejected_vms,
            "acceptance_rate": (
                (self.total_placed_vms / self.total_arrived_vms)
                if self.total_arrived_vms > 0
                else 1.0
            ),
            "peak_temperature_c": float(df["max_temp_c"].max()),
            "average_max_temperature_c": float(df["max_temp_c"].mean()),
            "average_thermal_imbalance_std": float(df["thermal_imbalance_std"].mean()),
            "peak_thermal_imbalance_std": float(df["thermal_imbalance_std"].max()),
            "cumulative_hotspot_events": int(df["hotspot_count"].sum()),
            "steps_with_hotspots": int((df["hotspot_count"] > 0).sum()),
            "average_cpu_utilization": float(df["mean_cpu_util"].mean()),
            "average_actual_cpu_utilization": float(df["mean_actual_cpu_util"].mean()),
            "average_ram_utilization": float(df["mean_ram_util"].mean()),
            "average_total_power_w": float(df["total_power_w"].mean()),
        }

    def reset(self) -> None:
        """Clears all accumulated records."""
        self.snapshots.clear()
        self.total_arrived_vms = 0
        self.total_placed_vms = 0
        self.total_rejected_vms = 0
