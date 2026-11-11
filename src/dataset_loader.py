"""Dataset loader module for ThermoPlace.

Provides reusable data structures and report representations for virtual machine (VM)
chronological CPU utilization traces.
"""

from dataclasses import dataclass, field
import logging
from typing import Dict, Iterator, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class VMTrace:
    """Represents a loaded VM trace with chronological CPU and optional disk utilization samples.

    Attributes:
        vm_id: Unique string identifier for the VM.
        cpu_samples: Chronological sequence of CPU utilization values in percentage [0.0, 100.0].
        source_file: Absolute or relative path to the original trace file.
        disk_read_samples: Optional chronological sequence of disk read throughput in KB/s.
        disk_write_samples: Optional chronological sequence of disk write throughput in KB/s.
        timestamps: Optional chronological sequence of measurement timestamps (e.g. Unix epoch ms).
    """
    vm_id: str
    cpu_samples: List[float]
    source_file: str
    disk_read_samples: Optional[List[float]] = None
    disk_write_samples: Optional[List[float]] = None
    timestamps: Optional[List[int]] = None

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

    @property
    def has_disk_trace(self) -> bool:
        """Indicates whether disk throughput measurements are present."""
        return bool(self.disk_read_samples or self.disk_write_samples)

    @property
    def total_disk_io_samples(self) -> Optional[List[float]]:
        """Returns combined chronological disk I/O throughput samples (read + write) in KB/s."""
        if self.disk_read_samples is not None and self.disk_write_samples is not None:
            return [
                float(r + w)
                for r, w in zip(self.disk_read_samples, self.disk_write_samples)
            ]
        if self.disk_read_samples is not None:
            return [float(r) for r in self.disk_read_samples]
        if self.disk_write_samples is not None:
            return [float(w) for w in self.disk_write_samples]
        return None

    @property
    def mean_disk_read(self) -> float:
        """Returns arithmetic mean disk read throughput in KB/s."""
        if not self.disk_read_samples:
            return 0.0
        return sum(self.disk_read_samples) / len(self.disk_read_samples)

    @property
    def max_disk_read(self) -> float:
        """Returns peak disk read throughput in KB/s."""
        if not self.disk_read_samples:
            return 0.0
        return max(self.disk_read_samples)

    @property
    def mean_disk_write(self) -> float:
        """Returns arithmetic mean disk write throughput in KB/s."""
        if not self.disk_write_samples:
            return 0.0
        return sum(self.disk_write_samples) / len(self.disk_write_samples)

    @property
    def max_disk_write(self) -> float:
        """Returns peak disk write throughput in KB/s."""
        if not self.disk_write_samples:
            return 0.0
        return max(self.disk_write_samples)

    @property
    def mean_disk_io(self) -> float:
        """Returns arithmetic mean combined disk I/O throughput in KB/s."""
        samples = self.total_disk_io_samples
        if not samples:
            return 0.0
        return sum(samples) / len(samples)

    @property
    def max_disk_io(self) -> float:
        """Returns peak combined disk I/O throughput in KB/s."""
        samples = self.total_disk_io_samples
        if not samples:
            return 0.0
        return max(samples)

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
