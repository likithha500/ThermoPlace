"""Traditional First Fit VM placement strategy for ThermoPlace.

First Fit is a standard heuristic that iterates sequentially through available
hosts and places the VM on the very first host that satisfies CPU and RAM
capacity constraints.

Thermal Characteristics:
Because First Fit always packs workload onto the earliest servers in the list,
it tends to create severe local hotspots on the first few hosts/racks while
leaving remaining servers idle. This serves as the primary baseline to benchmark
against thermal-aware RL schedulers.
"""

from typing import Optional
from src.placement.base import BasePlacementStrategy
from src.datacenter import DataCenter, Host
from src.workload import VM


class FirstFitPlacement(BasePlacementStrategy):
    """Places arriving VMs on the first host with sufficient CPU and RAM."""

    def __init__(self):
        super().__init__(name="First-Fit")

    def select_host(self, vm: VM, datacenter: DataCenter) -> Optional[Host]:
        """Iterates through hosts sequentially and returns the first capable host.

        Args:
            vm: Virtual Machine requesting placement.
            datacenter: Current state of the data center.

        Returns:
            The first Host instance that can accommodate the VM, or None if none found.
        """
        for host in datacenter.hosts:
            if host.can_accommodate(vm):
                return host
        return None
