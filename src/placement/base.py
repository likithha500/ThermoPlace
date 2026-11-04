"""Base placement strategy interface for ThermoPlace."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional
from src.datacenter import DataCenter, Host
from src.workload import VM


class BasePlacementStrategy(ABC):
    """Abstract base class for VM placement algorithms.

    Subclasses must implement ``select_host``.
    All other lifecycle hooks are optional no-ops that subclasses may override.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_host(self, vm: VM, datacenter: DataCenter) -> Optional[Host]:
        """Selects a host machine to place the arriving VM.

        Args:
            vm: Virtual Machine requesting placement.
            datacenter: The physical data center topology and state.

        Returns:
            The selected Host instance if capacity allows, or None if rejected.
        """
        pass

    # ------------------------------------------------------------------
    # Optional lifecycle hooks — default implementations are no-ops so
    # that existing strategies (e.g. FirstFitPlacement) need no changes.
    # ------------------------------------------------------------------

    def on_step_begin(
        self,
        time_step: int,
        arriving_vms: List[VM],
        datacenter: DataCenter,
    ) -> None:
        """Called at the start of every simulator time step, before any placements.

        Args:
            time_step: Index of the current discrete time step.
            arriving_vms: VMs that will arrive during this step.
            datacenter: Current datacenter state.
        """
        pass

    def on_placement_outcome(
        self,
        vm: VM,
        host: Optional[Host],
        placed: bool,
        time_step: int,
        datacenter: DataCenter,
    ) -> None:
        """Called immediately after a VM placement attempt (success or rejection).

        Args:
            vm: The VM that was attempted to be placed.
            host: The Host selected (None if rejected or allocation failed).
            placed: True if the VM was successfully allocated to ``host``.
            time_step: Current simulation time step.
            datacenter: Current datacenter state (after allocation, before thermal update).
        """
        pass

    def on_step_end(
        self,
        time_step: int,
        datacenter: DataCenter,
        thermal_model: Any = None,
    ) -> None:
        """Called at the end of a simulator time step, after the thermal model update.

        This is the correct point to compute rewards and perform Q-table updates
        for strategies that require post-thermal observations.

        Args:
            time_step: Index of the completed time step.
            datacenter: Datacenter state after thermal update.
            thermal_model: The ThermalModel instance (may be None for strategies
                           that do not need it).
        """
        pass

    def on_simulation_end(self, datacenter: DataCenter) -> None:
        """Called once after the entire simulation run completes.

        Args:
            datacenter: Final state of the datacenter.
        """
        pass
