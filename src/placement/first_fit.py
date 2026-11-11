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
    """Places arriving VMs on the first host with sufficient capacity across enabled resources."""

    def __init__(self, feature_set: str = "cpu_ram", name: str = "First-Fit"):
        super().__init__(name=name)
        self.feature_set = feature_set

    def select_host(self, vm: VM, datacenter: DataCenter) -> Optional[Host]:
        """Iterates through hosts sequentially and returns the first capable host.

        Enforces CPU, RAM, and Disk I/O constraints according to self.feature_set.

        Args:
            vm: Virtual Machine requesting placement.
            datacenter: Current state of the data center.

        Returns:
            The first Host instance that can accommodate the VM, or None if none found.
        """
        for host in datacenter.hosts:
            if host.can_accommodate(vm, feature_set=self.feature_set):
                return host
        return None
