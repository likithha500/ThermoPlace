"""PlanetLab dataset loader for ThermoPlace.

Loads real-world VM CPU utilization trace files from the PlanetLab workload dataset.
Preserves chronological order of utilization samples without generating synthetic data
or inventing missing values. Validates all inputs and reports invalid or skipped files.
"""

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class VMTrace:
    """Represents a loaded VM trace with chronological CPU utilization samples.

    Attributes:
        vm_id: Unique string identifier for the VM (typically derived from filename).
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


class PlanetLabDatasetLoader:
    """Dedicated loader for PlanetLab VM CPU utilization traces.

    PlanetLab trace files contain one CPU utilization measurement per line (in percent, 0-100),
    sampled at discrete periodic epochs (e.g., 5-minute intervals) over 24 hours (288 samples).
    """

    def __init__(
        self,
        min_allowed_cpu: float = 0.0,
        max_allowed_cpu: float = 100.0,
    ):
        """Initializes the dataset loader with validation thresholds.

        Args:
            min_allowed_cpu: Minimum valid CPU utilization percentage (default: 0.0).
            max_allowed_cpu: Maximum valid CPU utilization percentage (default: 100.0).

        Raises:
            ValueError: If min_allowed_cpu > max_allowed_cpu.
        """
        if min_allowed_cpu > max_allowed_cpu:
            raise ValueError(
                f"min_allowed_cpu ({min_allowed_cpu}) cannot exceed "
                f"max_allowed_cpu ({max_allowed_cpu})"
            )
        self.min_allowed_cpu = min_allowed_cpu
        self.max_allowed_cpu = max_allowed_cpu
        self.skipped_files: Dict[str, str] = {}

    def parse_trace_file(self, file_path: Union[str, Path]) -> VMTrace:
        """Parses a single PlanetLab trace file into a VMTrace object.

        Reads lines in chronological order and validates numeric format and range.
        Does not impute missing values or generate synthetic samples.

        Args:
            file_path: Path to the PlanetLab trace file.

        Returns:
            VMTrace object containing chronological CPU utilization samples.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file is empty, contains non-numeric lines, contains
                internal missing lines, or has values outside the valid percentage range.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Trace file not found: '{file_path}'")
        if not path.is_file():
            raise ValueError(f"Path is not a regular file: '{file_path}'")

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raise OSError(f"Could not read trace file '{file_path}': {e}") from e

        raw_lines = content.splitlines()

        # Check for internal empty lines (missing samples in the middle of a trace)
        first_non_empty = None
        last_non_empty = None
        for idx, line in enumerate(raw_lines):
            if line.strip():
                if first_non_empty is None:
                    first_non_empty = idx
                last_non_empty = idx

        if first_non_empty is None:
            raise ValueError(f"Trace file '{path.name}' contains no valid CPU utilization samples.")

        for idx in range(first_non_empty, last_non_empty + 1):
            if not raw_lines[idx].strip():
                raise ValueError(
                    f"Missing sample / internal empty line at line {idx + 1} in '{path.name}'. "
                    f"Traces must be continuous without missing epochs."
                )

        samples: List[float] = []
        for line_num, raw_line in enumerate(raw_lines[:last_non_empty + 1], start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                val = float(line)
            except ValueError as e:
                raise ValueError(
                    f"Invalid non-numeric value '{line}' at line {line_num} in '{path.name}'"
                ) from e

            if math.isnan(val) or math.isinf(val):
                raise ValueError(
                    f"Invalid non-finite CPU sample '{line}' at line {line_num} in '{path.name}'"
                )

            if val < self.min_allowed_cpu or val > self.max_allowed_cpu:
                raise ValueError(
                    f"CPU sample {val} at line {line_num} in '{path.name}' out of valid range "
                    f"[{self.min_allowed_cpu}, {self.max_allowed_cpu}]"
                )

            samples.append(val)

        vm_id = path.name
        return VMTrace(
            vm_id=vm_id,
            cpu_samples=samples,
            source_file=str(path.resolve()),
        )

    def load_with_report(
        self,
        directory_path: Union[str, Path],
        max_files: Optional[int] = None,
        sort_files: bool = True,
        recursive: bool = False,
    ) -> DatasetLoadReport:
        """Loads all valid trace files from a directory and returns a DatasetLoadReport.

        Args:
            directory_path: Path to directory containing PlanetLab trace files.
            max_files: Optional maximum number of trace files to load.
            sort_files: Whether to sort trace files deterministically by filename (default: True).
            recursive: Whether to search subdirectories recursively (default: False).

        Returns:
            DatasetLoadReport with valid_traces and skipped_files dictionary.

        Raises:
            FileNotFoundError: If the directory does not exist.
            NotADirectoryError: If the path is not a directory.
            ValueError: If max_files <= 0 or if no valid trace files were found.
        """
        dir_path = Path(directory_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: '{directory_path}'")
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: '{directory_path}'")

        if max_files is not None and max_files <= 0:
            raise ValueError(f"max_files must be greater than 0, got {max_files}")

        self.skipped_files.clear()
        valid_traces: List[VMTrace] = []

        if recursive:
            all_entries = [p for p in dir_path.rglob("*") if p.is_file()]
        else:
            all_entries = [p for p in dir_path.iterdir() if p.is_file()]

        # Filter out hidden files, git files, etc.
        candidate_files = []
        for p in all_entries:
            if any(part.startswith(".") for part in p.parts):
                continue
            candidate_files.append(p)

        if sort_files:
            candidate_files.sort(key=lambda p: (p.name, str(p)))

        for file_path in candidate_files:
            if max_files is not None and len(valid_traces) >= max_files:
                break

            try:
                trace = self.parse_trace_file(file_path)
                valid_traces.append(trace)
            except (ValueError, OSError) as e:
                self.skipped_files[str(file_path)] = str(e)
                logger.warning("Skipped invalid trace file '%s': %s", file_path, e)

        if not valid_traces:
            subdirs = [p.name for p in dir_path.iterdir() if p.is_dir() and not p.name.startswith(".")]
            hint = ""
            if subdirs and not recursive:
                hint = (
                    f" Note: Found {len(subdirs)} subdirectories ({subdirs[:3]}...). "
                    f"Pass recursive=True or specify a subdirectory path."
                )
            raise ValueError(
                f"No valid trace files found in '{directory_path}'. "
                f"Skipped {len(self.skipped_files)} files.{hint}"
            )

        return DatasetLoadReport(
            valid_traces=valid_traces,
            skipped_files=dict(self.skipped_files),
        )

    def load_directory(
        self,
        directory_path: Union[str, Path],
        max_files: Optional[int] = None,
        sort_files: bool = True,
        recursive: bool = False,
    ) -> List[VMTrace]:
        """Loads all valid trace files from the specified directory.

        Args:
            directory_path: Path to directory containing PlanetLab trace files.
            max_files: Optional maximum number of trace files to load.
            sort_files: Whether to sort trace files deterministically by filename (default: True).
            recursive: Whether to search subdirectories recursively (default: False).

        Returns:
            List of VMTrace instances in deterministic order.

        Raises:
            FileNotFoundError: If the directory does not exist.
            NotADirectoryError: If the path is not a directory.
            ValueError: If no valid trace files were found.
        """
        report = self.load_with_report(
            directory_path=directory_path,
            max_files=max_files,
            sort_files=sort_files,
            recursive=recursive,
        )
        return report.valid_traces


def load_planetlab_directory(
    directory_path: Union[str, Path],
    max_files: Optional[int] = None,
    recursive: bool = False,
) -> List[VMTrace]:
    """Convenience helper to load VM traces from a PlanetLab directory."""
    loader = PlanetLabDatasetLoader()
    return loader.load_directory(directory_path, max_files=max_files, recursive=recursive)


def parse_planetlab_file(file_path: Union[str, Path]) -> VMTrace:
    """Convenience helper to parse a single PlanetLab trace file."""
    loader = PlanetLabDatasetLoader()
    return loader.parse_trace_file(file_path)
