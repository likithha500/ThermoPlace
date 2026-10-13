"""Data center infrastructure module for ThermoPlace.

Defines physical hierarchy: Host, Rack, and DataCenter entities.
Enforces CPU and RAM capacity constraints, tracks active VM allocations,
and provides complete state reset functionality to ensure identical initial
conditions across different scheduling algorithms.
"""

from typing import List, Dict, Optional, Union
from src.workload import VM


class Host:
    """Represents a physical server (host machine) in a data center rack.

    Attributes:
        host_id: Unique identifier for the host (e.g., 'R0_H1').
        rack_id: Identifier of the rack containing this host.
        rack_slot: Vertical slot position in the rack (0 = bottom, ascending).
        total_cpu: Total CPU capacity units (e.g., cores).
        total_ram: Total memory capacity in Gigabytes (GB).
        used_cpu: Currently allocated CPU capacity units.
        used_ram: Currently allocated memory in Gigabytes (GB).
        idle_power_w: Baseline idle power consumption in Watts.
        max_power_w: Peak power consumption at 100% CPU utilization in Watts.
        simulated_temperature_c: Current simulated temperature estimate in Celsius (°C).
        initial_temperature_c: Baseline starting temperature in Celsius (°C).
        allocated_vms: Dictionary mapping VM ID to active VM objects.
    """

    def __init__(
        self,
        host_id: str,
        rack_id: int,
        rack_slot: int,
        total_cpu: int = 16,
        total_ram: int = 64,
        idle_power_w: float = 150.0,
        max_power_w: float = 350.0,
        initial_temperature_c: float = 25.0,
    ):
        self.host_id = host_id
        self.rack_id = rack_id
        self.rack_slot = rack_slot
        self.total_cpu = total_cpu
        self.total_ram = total_ram
        self.idle_power_w = idle_power_w
        self.max_power_w = max_power_w
        self.initial_temperature_c = initial_temperature_c

        self.used_cpu: int = 0
        self.used_ram: int = 0
        self.simulated_temperature_c: float = initial_temperature_c
        self.allocated_vms: Dict[Union[int, str], VM] = {}

    def can_accommodate(self, vm: VM) -> bool:
        """Checks whether the host has sufficient available CPU and RAM.

        Args:
            vm: Virtual Machine requesting placement.

        Returns:
            True if both CPU and RAM capacity constraints are satisfied, False otherwise.
        """
        has_cpu = (self.used_cpu + vm.cpu_demand) <= self.total_cpu
        has_ram = (self.used_ram + vm.ram_demand) <= self.total_ram
        return has_cpu and has_ram

    def allocate(self, vm: VM) -> bool:
        """Allocates a VM to this host if capacity constraints allow.

        Args:
            vm: Virtual Machine to allocate.

        Returns:
            True if allocation succeeded, False if rejected due to capacity limits.
        """
        if not self.can_accommodate(vm):
            return False

        self.allocated_vms[vm.vm_id] = vm
        self.used_cpu += vm.cpu_demand
        self.used_ram += vm.ram_demand
        return True

    def release(self, vm_id: Union[int, str]) -> Optional[VM]:
        """Releases and removes a completed VM from this host.

        Args:
            vm_id: Identifier of the VM to release.

        Returns:
            The released VM object if found, or None if not present on this host.
        """
        if vm_id in self.allocated_vms:
            vm = self.allocated_vms.pop(vm_id)
            self.used_cpu = max(0, self.used_cpu - vm.cpu_demand)
            self.used_ram = max(0, self.used_ram - vm.ram_demand)
            return vm
        return None

    @property
    def cpu_utilization(self) -> float:
        """Returns the current CPU allocation ratio in the range [0.0, 1.0]."""
        if self.total_cpu <= 0:
            return 0.0
        return min(1.0, max(0.0, self.used_cpu / self.total_cpu))

    @property
    def actual_used_cpu(self) -> float:
        """Returns the sum of actual CPU cores utilized across active VMs based on real traces."""
        return sum(vm.current_actual_cpu for vm in self.allocated_vms.values())

    @property
    def actual_cpu_utilization(self) -> float:
        """Returns the actual CPU utilization ratio based on trace samples in the range [0.0, 1.0]."""
        if self.total_cpu <= 0:
            return 0.0
        return min(1.0, max(0.0, self.actual_used_cpu / self.total_cpu))

    @property
    def ram_utilization(self) -> float:
        """Returns the current RAM utilization ratio in the range [0.0, 1.0]."""
        if self.total_ram <= 0:
            return 0.0
        return min(1.0, max(0.0, self.used_ram / self.total_ram))

    @property
    def available_cpu(self) -> int:
        """Returns remaining unallocated CPU units."""
        return max(0, self.total_cpu - self.used_cpu)

    @property
    def available_ram(self) -> int:
        """Returns remaining unallocated RAM in GB."""
        return max(0, self.total_ram - self.used_ram)

    def reset(self, initial_temp: Optional[float] = None) -> None:
        """Resets the host to its pristine initial state with zero workload.

        Args:
            initial_temp: Optional override for reset temperature in Celsius.
        """
        self.allocated_vms.clear()
        self.used_cpu = 0
        self.used_ram = 0
        self.simulated_temperature_c = (
            initial_temp if initial_temp is not None else self.initial_temperature_c
        )

    def __repr__(self) -> str:
        return (
            f"Host({self.host_id}, rack={self.rack_id}, slot={self.rack_slot}, "
            f"CPU={self.used_cpu}/{self.total_cpu}, RAM={self.used_ram}/{self.total_ram}GB, "
            f"T_sim={self.simulated_temperature_c:.1f}°C)"
        )


class Rack:
    """Represents a physical server rack housing multiple vertically stacked hosts.

    Attributes:
        rack_id: Unique integer identifier for the rack.
        hosts: List of Host instances positioned inside this rack.
    """

    def __init__(self, rack_id: int, hosts: Optional[List[Host]] = None):
        self.rack_id = rack_id
        self.hosts: List[Host] = hosts or []

    def add_host(self, host: Host) -> None:
        """Adds a host to this rack."""
        self.hosts.append(host)

    def reset(self, initial_temp: Optional[float] = None) -> None:
        """Resets all hosts within this rack."""
        for host in self.hosts:
            host.reset(initial_temp)

    def total_used_cpu(self) -> int:
        """Calculates total CPU cores utilized across all hosts in the rack."""
        return sum(h.used_cpu for h in self.hosts)

    def total_used_ram(self) -> int:
        """Calculates total RAM utilized across all hosts in the rack."""
        return sum(h.used_ram for h in self.hosts)

    def __len__(self) -> int:
        return len(self.hosts)


class DataCenter:
    """Encapsulates the complete physical data center topology.

    Provides global access to racks and hosts, and centralizes state resets
    to guarantee consistent initial conditions for comparative experiments.
    """

    def __init__(self, racks: Optional[List[Rack]] = None):
        self.racks: List[Rack] = racks or []
        self._host_map: Dict[str, Host] = {}
        self._sync_host_map()

    def _sync_host_map(self) -> None:
        """Updates internal host lookup index."""
        self._host_map.clear()
        for rack in self.racks:
            for host in rack.hosts:
                self._host_map[host.host_id] = host

    @property
    def hosts(self) -> List[Host]:
        """Returns a flat list of all hosts across all racks."""
        all_hosts = []
        for rack in self.racks:
            all_hosts.extend(rack.hosts)
        return all_hosts

    def get_host(self, host_id: str) -> Optional[Host]:
        """Retrieves a host by unique identifier."""
        return self._host_map.get(host_id)

    def reset(self, initial_temp: Optional[float] = None) -> None:
        """Resets all hosts and racks in the data center to pristine initial conditions."""
        for rack in self.racks:
            rack.reset(initial_temp)

    @classmethod
    def create_standard_datacenter(
        cls,
        num_racks: int = 2,
        hosts_per_rack: int = 4,
        cpu_per_host: int = 16,
        ram_per_host: int = 64,
        idle_power_w: float = 150.0,
        max_power_w: float = 350.0,
        initial_temperature_c: float = 25.0,
    ) -> "DataCenter":
        """Factory method to instantiate a standard homogeneous data center.

        Args:
            num_racks: Number of physical racks.
            hosts_per_rack: Number of hosts stacked in each rack.
            cpu_per_host: CPU cores per host.
            ram_per_host: RAM capacity per host in GB.
            idle_power_w: Idle power consumption per host in Watts.
            max_power_w: Peak power consumption per host in Watts.
            initial_temperature_c: Baseline starting temperature in Celsius.

        Returns:
            Configured DataCenter instance.
        """
        racks: List[Rack] = []
        for r_idx in range(num_racks):
            rack = Rack(rack_id=r_idx)
            for h_idx in range(hosts_per_rack):
                host = Host(
                    host_id=f"R{r_idx}_H{h_idx}",
                    rack_id=r_idx,
                    rack_slot=h_idx,
                    total_cpu=cpu_per_host,
                    total_ram=ram_per_host,
                    idle_power_w=idle_power_w,
                    max_power_w=max_power_w,
                    initial_temperature_c=initial_temperature_c,
                )
                rack.add_host(host)
            racks.append(rack)

        dc = cls(racks=racks)
        return dc
