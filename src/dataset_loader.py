"""Dataset loader module for ThermoPlace.

Provides reusable data structures and report representations for virtual machine (VM)
chronological CPU utilization traces.
"""

from dataclasses import dataclass, field
import logging
from typing import Dict, Iterator, List, Union

logger = logging.getLogger(__name__)


@dataclass
class VMTrace:
    """Represents a loaded VM trace with chronological CPU utilization samples.

    Attributes:
        vm_id: Unique string identifier for the VM.
        cpu_samples: Chronological sequence of CPU utilization values in percentage [0.0, 100.0].
        source_file: Absolute or relative path to the original trace file.
    """
    vm_id: str
    cpu_samples: List[float]
    source_file: str

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
