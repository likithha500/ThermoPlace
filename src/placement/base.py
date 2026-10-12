"""Base placement strategy interface for ThermoPlace."""

from abc import ABC, abstractmethod
from typing import Optional
from src.datacenter import DataCenter, Host
from src.workload import VM


class BasePlacementStrategy(ABC):
    """Abstract base class for VM placement algorithms."""

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
