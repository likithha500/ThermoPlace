"""Dataset loader module for ThermoPlace.

Provides reusable data structures and report representations for virtual machine (VM)
chronological CPU utilization traces and full Bitbrains telemetry columns.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, Iterator, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class VMTrace:
    """Represents a loaded VM trace with chronological CPU utilization samples.

    Attributes:
        vm_id: Unique string identifier for the VM.
        cpu_samples: Chronological sequence of CPU utilization values in percentage [0.0, 100.0].
            This represents the primary CPU workload feature.
        source_file: Absolute or relative path to the original trace file.
        cpu_cores: Provisioned CPU core count from trace.
        cpu_capacity_provisioned_mhz: Provisioned CPU capacity in MHz.
        cpu_usage_mhz: Chronological CPU usage in MHz (retained for reporting/validation).
        memory_provisioned_kb: Chronological provisioned memory capacity in KB.
        memory_usage_kb: Chronological memory usage in KB.
        disk_read_kbps: Chronological disk read throughput in KB/s.
        disk_write_kbps: Chronological disk write throughput in KB/s.
        network_rx_kbps: Chronological network received throughput in KB/s.
        network_tx_kbps: Chronological network transmitted throughput in KB/s.
        metadata: Dictionary holding raw header/file metadata.
    """
    vm_id: str
    cpu_samples: List[float]
    source_file: str
    cpu_cores: Optional[int] = None
    cpu_capacity_provisioned_mhz: Optional[float] = None
    cpu_usage_mhz: Optional[List[float]] = None
    memory_provisioned_kb: Optional[List[float]] = None
    memory_usage_kb: Optional[List[float]] = None
    disk_read_kbps: Optional[List[float]] = None
    disk_write_kbps: Optional[List[float]] = None
    network_rx_kbps: Optional[List[float]] = None
    network_tx_kbps: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_samples(self) -> int:
        """Returns the number of time samples recorded for this VM."""
        return len(self.cpu_samples)

    @property
    def min_cpu(self) -> float:
        """Returns minimum CPU utilization across all chronological samples."""
        return min(self.cpu_samples) if self.cpu_samples else 0.0

    @property
    def max_cpu(self) -> float:
        """Returns peak CPU utilization across all chronological samples."""
        return max(self.cpu_samples) if self.cpu_samples else 0.0

    @property
    def mean_cpu(self) -> float:
        """Returns arithmetic mean CPU utilization across all samples."""
        return sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0

    def __len__(self) -> int:
        """Returns the total number of CPU samples."""
        return len(self.cpu_samples)

    def __getitem__(self, index: Union[int, slice]) -> Union[float, List[float]]:
        """Allows direct indexing or slicing of chronological CPU samples."""
        return self.cpu_samples[index]

    def __iter__(self) -> Iterator[float]:
        """Allows iteration over chronological CPU samples."""
        return iter(self.cpu_samples)


@dataclass
class DatasetLoadReport:
    """Detailed report summarizing loaded traces and skipped files."""
    valid_traces: List[VMTrace] = field(default_factory=list)
    skipped_files: Dict[str, str] = field(default_factory=dict)

    @property
    def total_loaded(self) -> int:
        """Returns the count of successfully parsed VM traces."""
        return len(self.valid_traces)

    @property
    def total_skipped(self) -> int:
        """Returns the count of skipped/invalid files."""
        return len(self.skipped_files)
