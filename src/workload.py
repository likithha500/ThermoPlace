"""Workload generation module for ThermoPlace.

Provides:
1. Virtual Machine (VM) model supporting both synthetic workloads and real
   PlanetLab CPU utilization traces.
2. Deterministic, reproducible synthetic VM arrival generation (WorkloadGenerator).
3. PlanetLab workload generation mapping real trace files to simulation VMs (PlanetLabWorkloadGenerator).

Scientific & Modeling Assumptions:
- CPU Capacity vs Utilization:
  `cpu_demand` represents the reserved/allocated CPU capacity (cores/slots)
  enforced during host placement and admission control.
  `current_cpu_utilization` is the dynamic percentage [0.0, 100.0] from real
  PlanetLab traces.
  `current_actual_cpu` is the actual utilized capacity: cpu_demand * (utilization / 100.0).
- RAM Provisioning:
  PlanetLab dataset traces exclusively record CPU utilization percentages.
  Memory (RAM) measurements are NOT present in the dataset and are NOT fabricated.
  `ram_demand` represents the statically provisioned memory capacity reserved on
  the host for VM scheduling and capacity constraints.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np

try:
    from src.dataset_loader import VMTrace
except ImportError:
    VMTrace = Any  # type: ignore


@dataclass
class VM:
    """Represents a Virtual Machine request in the simulation.

    Attributes:
        vm_id: Unique integer or string identifier for the VM.
        cpu_demand: Reserved CPU capacity units (e.g., CPU cores) for placement.
        ram_demand: Reserved memory in Gigabytes (GB) for placement.
            NOTE ON RAM: PlanetLab dataset files only record CPU utilization percentages.
            RAM utilization is not measured in the dataset and is NOT fabricated.
            ram_demand represents the statically provisioned RAM allocated on the host.
        arrival_time: Discrete simulation time step when the VM arrives.
        duration: Lifetime duration of the VM in discrete time steps.
        remaining_duration: Remaining time steps left before completion.
        cpu_trace: Chronological sequence of real CPU utilization samples [%, 0.0-100.0].
        current_sample_index: Index of the current CPU utilization sample in cpu_trace.
        trace_id: Optional string identifier for the originating trace source.
    """
    vm_id: Union[int, str]
    cpu_demand: int
    ram_demand: int
    arrival_time: int
    duration: int = 0
    remaining_duration: int = 0
    cpu_trace: Optional[List[float]] = None
    current_sample_index: int = 0
    trace_id: Optional[str] = None
    loop_trace: bool = False

    def __post_init__(self):
        # Normalize cpu_trace if provided as a VMTrace or similar object
        if self.cpu_trace is not None:
            if hasattr(self.cpu_trace, "cpu_samples"):
                if self.trace_id is None and hasattr(self.cpu_trace, "vm_id"):
                    self.trace_id = str(self.cpu_trace.vm_id)
                self.cpu_trace = [float(x) for x in self.cpu_trace.cpu_samples]
            elif isinstance(self.cpu_trace, (list, tuple)):
                self.cpu_trace = [float(x) for x in self.cpu_trace]

            # If duration was not explicitly specified (or <= 0), infer from trace length
            if self.duration <= 0 and len(self.cpu_trace) > 0:
                self.duration = len(self.cpu_trace)

        if self.remaining_duration == 0:
            self.remaining_duration = self.duration

    @property
    def current_cpu_utilization(self) -> Optional[float]:
        """Returns the current CPU utilization sample in percentage [0.0, 100.0].

        Returns None if no trace is assigned or if all trace samples have been consumed
        without loop_trace configured.
        """
        if self.cpu_trace is None or len(self.cpu_trace) == 0:
            return None
        if self.loop_trace:
            idx = self.current_sample_index % len(self.cpu_trace)
            return self.cpu_trace[idx]
        if self.current_sample_index < len(self.cpu_trace):
            return self.cpu_trace[self.current_sample_index]
        return None

    @property
    def current_actual_cpu(self) -> float:
        """Returns the actual CPU capacity units/cores utilized at current time step.

        If a trace is attached: cpu_demand * (current_cpu_utilization / 100.0).
        If no trace is attached: assumes 100% of reserved cpu_demand is utilized.
        """
        util = self.current_cpu_utilization
        if util is not None:
            return self.cpu_demand * (util / 100.0)
        return float(self.cpu_demand)

    @property
    def has_trace(self) -> bool:
        """Indicates whether this VM has an associated real CPU trace."""
        return self.cpu_trace is not None and len(self.cpu_trace) > 0

    @property
    def is_trace_finished(self) -> bool:
        """Returns True if a trace is attached and all samples have been consumed."""
        if self.cpu_trace is None:
            return False
        if self.loop_trace:
            return False
        return self.current_sample_index >= len(self.cpu_trace)

    @property
    def total_trace_samples(self) -> int:
        """Returns total number of samples in the attached trace, or 0."""
        return len(self.cpu_trace) if self.cpu_trace is not None else 0

    @property
    def remaining_trace_samples(self) -> int:
        """Returns number of unconsumed samples remaining in the trace."""
        if self.cpu_trace is None:
            return 0
        if self.loop_trace:
            return self.remaining_duration
        return max(0, len(self.cpu_trace) - self.current_sample_index)

    def tick(self) -> bool:
        """Advances VM execution by one time step and steps to the next trace sample.

        Returns:
            True if the VM is still active, False if the VM has completed.
        """
        if self.cpu_trace is not None:
            if self.loop_trace and len(self.cpu_trace) > 0:
                self.current_sample_index += 1
            elif self.current_sample_index < len(self.cpu_trace):
                self.current_sample_index += 1

        self.remaining_duration -= 1
        return not self.is_finished()

    def is_finished(self) -> bool:
        """Checks if the VM has completed its execution lifetime."""
        if self.cpu_trace is not None and not self.loop_trace:
            if self.current_sample_index >= len(self.cpu_trace):
                return True
        return self.remaining_duration <= 0

    def reset(self) -> None:
        """Resets VM remaining duration and trace sample index to initial states."""
        self.remaining_duration = self.duration
        self.current_sample_index = 0

    @classmethod
    def from_trace(
        cls,
        trace: Union["VMTrace", List[float]],
        vm_id: Optional[Union[int, str]] = None,
        cpu_demand: int = 2,
        ram_demand: int = 4,
        arrival_time: int = 0,
        duration: Optional[int] = None,
        loop_trace: bool = False,
    ) -> "VM":
        """Instantiates a VM backed by a PlanetLab CPU utilization trace.

        Args:
            trace: VMTrace instance (from dataset_loader) or a list of float utilization values.
            vm_id: Optional ID override (defaults to trace.vm_id if available, or 'vm_0').
            cpu_demand: Reserved CPU capacity units/cores for placement (default: 2).
            ram_demand: Statically reserved RAM capacity in GB for placement (default: 4).
                NOTE: PlanetLab does not measure RAM. This is provisioned placement capacity.
            arrival_time: Simulation time step when VM arrives (default: 0).
            duration: Optional duration limit; defaults to len(cpu_samples).
            loop_trace: Whether to cycle the trace continuously when duration exceeds samples.

        Returns:
            Initialized VM instance with attached CPU trace.
        """
        trace_id = None
        if hasattr(trace, "cpu_samples"):
            samples = list(trace.cpu_samples)
            trace_id = str(trace.vm_id)
            if vm_id is None:
                vm_id = trace.vm_id
        else:
            samples = [float(s) for s in trace]

        if vm_id is None:
            vm_id = "vm_0"

        trace_len = len(samples)
        dur = duration if duration is not None and duration > 0 else trace_len

        return cls(
            vm_id=vm_id,
            cpu_demand=cpu_demand,
            ram_demand=ram_demand,
            arrival_time=arrival_time,
            duration=dur,
            remaining_duration=dur,
            cpu_trace=samples,
            current_sample_index=0,
            trace_id=trace_id,
            loop_trace=loop_trace,
        )


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


class PlanetLabWorkloadGenerator:
    """Generates reproducible simulation workloads from real PlanetLab traces.

    Maps loaded PlanetLab VM traces into VM simulation entities, assigning
    arrival times, reserved CPU cores, and provisioned RAM capacity.

    RAM Allocation Note:
        The PlanetLab dataset records exclusively CPU utilization percentages.
        RAM consumption is not present in the dataset and is NOT fabricated.
        RAM capacity is provisioned based on standard datacenter VM flavor profiles
        (e.g., [2, 4, 8, 16] GB).
    """

    def __init__(
        self,
        traces: List[Union["VMTrace", List[float]]],
        cpu_options: Optional[List[int]] = None,
        ram_options: Optional[List[int]] = None,
        arrival_rate_lambda: float = 2.0,
        max_arrival_time: int = 50,
        seed: int = 42,
    ):
        """Initializes the PlanetLab workload generator.

        Args:
            traces: List of VMTrace objects or raw sample lists.
            cpu_options: Possible reserved CPU cores (default: [1, 2, 4, 8]).
            ram_options: Possible provisioned RAM GB (default: [2, 4, 8, 16]).
            arrival_rate_lambda: Mean VM arrivals per time step (Poisson).
            max_arrival_time: Latest time step at which VMs can arrive.
            seed: Random seed for deterministic flavor and arrival assignment.
        """
        self.traces = traces
        self.cpu_options = cpu_options or [1, 2, 4, 8]
        self.ram_options = ram_options or [2, 4, 8, 16]
        self.arrival_rate_lambda = arrival_rate_lambda
        self.max_arrival_time = max_arrival_time
        self.seed = seed

    def generate_workload(self) -> List[VM]:
        """Creates a list of VM entities for each trace with deterministic arrivals and flavors.

        Returns:
            List of VM instances with attached traces, sorted by arrival_time.
        """
        rng = np.random.default_rng(self.seed)
        vms: List[VM] = []

        if self.max_arrival_time <= 0 or self.arrival_rate_lambda <= 0:
            for idx, trace in enumerate(self.traces):
                cpu = int(rng.choice(self.cpu_options))
                ram = int(rng.choice(self.ram_options))
                vm = VM.from_trace(
                    trace=trace,
                    vm_id=getattr(trace, "vm_id", f"vm_{idx}"),
                    cpu_demand=cpu,
                    ram_demand=ram,
                    arrival_time=0,
                )
                vms.append(vm)
            return vms

        trace_idx = 0
        total_traces = len(self.traces)

        for t in range(self.max_arrival_time + 1):
            if trace_idx >= total_traces:
                break
            num_arrivals = rng.poisson(lam=self.arrival_rate_lambda)
            for _ in range(num_arrivals):
                if trace_idx >= total_traces:
                    break
                trace = self.traces[trace_idx]
                cpu = int(rng.choice(self.cpu_options))
                ram = int(rng.choice(self.ram_options))
                vm = VM.from_trace(
                    trace=trace,
                    vm_id=getattr(trace, "vm_id", f"vm_{trace_idx}"),
                    cpu_demand=cpu,
                    ram_demand=ram,
                    arrival_time=t,
                )
                vms.append(vm)
                trace_idx += 1

        # If any traces remain, allocate them at max_arrival_time
        while trace_idx < total_traces:
            trace = self.traces[trace_idx]
            cpu = int(rng.choice(self.cpu_options))
            ram = int(rng.choice(self.ram_options))
            vm = VM.from_trace(
                trace=trace,
                vm_id=getattr(trace, "vm_id", f"vm_{trace_idx}"),
                cpu_demand=cpu,
                ram_demand=ram,
                arrival_time=self.max_arrival_time,
            )
            vms.append(vm)
            trace_idx += 1

        vms.sort(key=lambda v: (v.arrival_time, str(v.vm_id)))
        return vms

    def generate_schedule_by_step(self, total_time_steps: Optional[int] = None) -> Dict[int, List[VM]]:
        """Maps simulation time steps to arriving VMs.

        Returns:
            Dictionary mapping time step to list of arriving VMs.
        """
        all_vms = self.generate_workload()
        max_step = max((vm.arrival_time for vm in all_vms), default=0)
        steps = total_time_steps or (max_step + 1)
        schedule: Dict[int, List[VM]] = {t: [] for t in range(steps)}
        for vm in all_vms:
            if vm.arrival_time in schedule:
                schedule[vm.arrival_time].append(vm)
        return schedule
