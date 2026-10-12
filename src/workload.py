"""Workload generation module for ThermoPlace.

Provides deterministic, reproducible synthetic Virtual Machine (VM) arrival
traces using fixed random seeds. Both baseline and future reinforcement
learning schedulers must receive the exact same workload trace for fair comparison.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np


@dataclass
class VM:
    """Represents a Virtual Machine request in the simulation.

    Attributes:
        vm_id: Unique integer or string identifier for the VM.
        cpu_demand: Required CPU capacity units (e.g., CPU cores).
        ram_demand: Required memory in Gigabytes (GB).
        arrival_time: Discrete simulation time step when the VM arrives.
        duration: Lifetime duration of the VM in discrete time steps.
        remaining_duration: Remaining time steps left before completion.
    """
    vm_id: int
    cpu_demand: int
    ram_demand: int
    arrival_time: int
    duration: int
    remaining_duration: int = 0

    def __post_init__(self):
        if self.remaining_duration == 0:
            self.remaining_duration = self.duration

    def tick(self) -> bool:
        """Decrements remaining duration by one time step.

        Returns:
            True if the VM is still active, False if the VM has completed.
        """
        self.remaining_duration -= 1
        return self.remaining_duration > 0

    def is_finished(self) -> bool:
        """Checks if the VM has completed its execution lifetime."""
        return self.remaining_duration <= 0


class WorkloadGenerator:
    """Generates synthetic, repeatable VM workloads using a fixed random seed.

    Ensures that multiple runs of different scheduling algorithms evaluate on
    an identical workload trace.
    """

    def __init__(
        self,
        seed: int = 42,
        total_time_steps: int = 100,
        arrival_rate_lambda: float = 2.0,
        cpu_options: Optional[List[int]] = None,
        ram_options: Optional[List[int]] = None,
        duration_min: int = 5,
        duration_max: int = 25,
    ):
        """Initializes the workload generator.

        Args:
            seed: Fixed random seed for reproducibility.
            total_time_steps: Number of simulation steps to generate arrivals for.
            arrival_rate_lambda: Mean number of VM arrivals per time step (Poisson).
            cpu_options: List of possible CPU core demands (default: [1, 2, 4, 8]).
            ram_options: List of possible RAM demands in GB (default: [2, 4, 8, 16]).
            duration_min: Minimum duration in time steps.
            duration_max: Maximum duration in time steps.
        """
        self.seed = seed
        self.total_time_steps = total_time_steps
        self.arrival_rate_lambda = arrival_rate_lambda
        self.cpu_options = cpu_options or [1, 2, 4, 8]
        self.ram_options = ram_options or [2, 4, 8, 16]
        self.duration_min = duration_min
        self.duration_max = duration_max

    def generate_workload(self) -> List[VM]:
        """Generates a complete list of VM requests across all time steps.

        Returns:
            List of VM instances sorted by arrival_time.
        """
        rng = np.random.default_rng(self.seed)
        vms: List[VM] = []
        vm_counter = 0

        for t in range(self.total_time_steps):
            # Number of arrivals at time step t follows a Poisson distribution
            num_arrivals = rng.poisson(lam=self.arrival_rate_lambda)
            for _ in range(num_arrivals):
                cpu = int(rng.choice(self.cpu_options))
                ram = int(rng.choice(self.ram_options))
                duration = int(rng.integers(self.duration_min, self.duration_max + 1))

                vm = VM(
                    vm_id=vm_counter,
                    cpu_demand=cpu,
                    ram_demand=ram,
                    arrival_time=t,
                    duration=duration,
                    remaining_duration=duration,
                )
                vms.append(vm)
                vm_counter += 1

        return vms

    def generate_schedule_by_step(self) -> Dict[int, List[VM]]:
        """Generates a mapping from simulation time step to arriving VMs.

        Returns:
            Dictionary where keys are time steps and values are lists of arriving VMs.
        """
        all_vms = self.generate_workload()
        schedule: Dict[int, List[VM]] = {t: [] for t in range(self.total_time_steps)}
        for vm in all_vms:
            if vm.arrival_time in schedule:
                schedule[vm.arrival_time].append(vm)
        return schedule
