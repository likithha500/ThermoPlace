"""Simulation engine for ThermoPlace.

Coordinates the discrete-time execution loop:
1. Releasing finished VMs from hosts.
2. Dispatching newly arriving VMs to the placement strategy under CPU/RAM constraints.
3. Updating simulated thermal state via the analytical RC thermal model.
4. Logging metrics and telemetry at each step.
"""

from typing import Dict, List, Optional
from src.datacenter import DataCenter, Host
from src.workload import VM, WorkloadGenerator
from src.thermal import ThermalModel, ThermalConfig
from src.placement.base import BasePlacementStrategy
from src.metrics import MetricsCollector


class Simulator:
    """Discrete-time thermal-aware data center simulator."""

    def __init__(
        self,
        datacenter: DataCenter,
        thermal_model: Optional[ThermalModel] = None,
        thermal_config: Optional[ThermalConfig] = None,
    ):
        self.datacenter = datacenter
        self.thermal_model = thermal_model or ThermalModel(thermal_config or ThermalConfig())
        self.current_step: int = 0

    def step(
        self,
        time_step: int,
        arriving_vms: List[VM],
        strategy: BasePlacementStrategy,
        metrics: MetricsCollector,
    ) -> None:
        """Executes a single discrete simulation time step.

        Lifecycle:
        1. Tick and release departing VMs from hosts.
        2. Attempt placement for all newly arriving VMs using the strategy.
        3. Step thermal model forward across all racks and hosts.
        4. Record telemetry snapshot.

        Args:
            time_step: Integer index of current time step.
            arriving_vms: List of VM instances arriving at this time step.
            strategy: Placement algorithm to decide host allocation.
            metrics: MetricsCollector to record step telemetry.
        """
        # 1. Tick active VMs and release finished ones
        for host in self.datacenter.hosts:
            finished_vm_ids = []
            for vm_id, vm in host.allocated_vms.items():
                vm.tick()
                if vm.is_finished():
                    finished_vm_ids.append(vm_id)

            for vm_id in finished_vm_ids:
                host.release(vm_id)

        # 2. Place arriving VMs
        placed_count = 0
        rejected_count = 0

        for vm in arriving_vms:
            selected_host = strategy.select_host(vm, self.datacenter)
            if selected_host is not None and selected_host.allocate(vm):
                placed_count += 1
            else:
                rejected_count += 1

        # 3. Update simulated thermal dynamics
        self.thermal_model.update_datacenter(self.datacenter, dt=1.0)

        # 4. Telemetry collection
        metrics.record_step(
            time_step=time_step,
            datacenter=self.datacenter,
            vms_arrived_this_step=len(arriving_vms),
            vms_placed_this_step=placed_count,
            vms_rejected_this_step=rejected_count,
        )

        self.current_step += 1

    def run(
        self,
        workload_schedule: Dict[int, List[VM]],
        strategy: BasePlacementStrategy,
        total_steps: Optional[int] = None,
    ) -> MetricsCollector:
        """Executes a full simulation run given a pre-generated workload schedule.

        Args:
            workload_schedule: Mapping of time_step -> list of arriving VMs.
            strategy: Placement strategy instance (e.g. FirstFitPlacement).
            total_steps: Optional total steps override.

        Returns:
            MetricsCollector containing complete telemetry and summary statistics.
        """
        metrics = MetricsCollector(
            algorithm_name=strategy.name,
            thermal_model=self.thermal_model,
        )

        steps_to_run = total_steps or max(workload_schedule.keys(), default=0) + 1

        for t in range(steps_to_run):
            arriving = workload_schedule.get(t, [])
            self.step(
                time_step=t,
                arriving_vms=arriving,
                strategy=strategy,
                metrics=metrics,
            )

        return metrics

    def reset(self, initial_temp: Optional[float] = None) -> None:
        """Resets the simulator and data center to fresh initial state."""
        self.current_step = 0
        self.datacenter.reset(initial_temp)
