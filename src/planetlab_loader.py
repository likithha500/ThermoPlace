"""PlanetLab CPU utilization trace loader for ThermoPlace.

Dataset source
--------------
PlanetLab workload traces from the GWA-T-12 collection hosted at:
    http://gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains

Citation:
    Beloglazov, A., & Buyya, R. (2012). Optimal Online Deterministic Algorithms
    and Adaptive Heuristics for Energy and Performance Efficient Dynamic
    Consolidation of Virtual Machines in Cloud Data Centers. Concurrency and
    Computation: Practice and Experience, 24(13), 1397-1420.

    Also cited as: Beloglazov & Buyya, "Autonomous Energy Efficient Data
    Centers" (2012).

File format
-----------
- Plain text files, one CPU utilization value (float) per line.
- No header row.
- Nominally 288 lines per file representing 5-minute intervals over 24 hours
  (288 × 5 min = 1440 min = 24 h).
- CPU values are measured utilization percentages in the range [0.0, 100.0].
- Values outside [0, 100] are clamped to the nearest bound.
- Empty lines and non-numeric lines are silently skipped.
- Files are organized in date-based subdirectories (e.g. ``20090320/``).

Memory note
-----------
RAM consumption is NOT measured in PlanetLab traces. It is instead assigned
from standard VM flavor profiles (e.g., [2, 4, 8, 16] GB) by
``TraceWorkloadGenerator`` at schedule-generation time, mirroring the approach
used for the Bitbrains dataset.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.dataset_loader import DatasetLoadReport, VMTrace

logger = logging.getLogger(__name__)


def _parse_planetlab_file(file_path: Path) -> VMTrace:
    """Parses a single PlanetLab plain-text trace file into a VMTrace object.

    Args:
        file_path: Path to a PlanetLab-format ``.txt`` trace file.

    Returns:
        VMTrace containing chronological CPU utilization samples in [0.0, 100.0].

    Raises:
        ValueError: If the file contains no valid numeric samples after filtering.
        OSError: If the file cannot be opened or read.
    """
    try:
        raw_text = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        raise OSError(f"Could not read '{file_path.name}': {exc}") from exc

    cpu_samples: List[float] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            # Skip blank lines
            continue
        try:
            value = float(stripped)
        except ValueError:
            # Skip non-numeric lines (headers, annotations, etc.)
            logger.debug("Skipping non-numeric line in '%s': %r", file_path.name, stripped)
            continue
        # Clamp to [0.0, 100.0]
        clamped = max(0.0, min(100.0, value))
        cpu_samples.append(clamped)

    if not cpu_samples:
        raise ValueError(
            f"No valid CPU utilization samples found in '{file_path.name}'. "
            "All lines were empty, non-numeric, or the file was empty."
        )

    return VMTrace(
        vm_id=file_path.stem,
        cpu_samples=cpu_samples,
        source_file=str(file_path),
    )


class PlanetLabLoader:
    """Loads VM CPU utilization traces from a PlanetLab dataset directory.

    PlanetLab traces are plain-text files with one CPU utilization percentage
    value per line, typically organized in date-based subdirectories:

        <dataset_dir>/
            20090320/
                planetlab1.txt
                planetlab2.txt
                ...
            20090321/
                ...

    Each file represents one VM's 24-hour CPU profile sampled at 5-minute
    intervals (288 samples per file).
    """

    def __init__(self) -> None:
        self._skipped: Dict[str, str] = {}

    def load_with_report(
        self,
        directory_path: Union[str, Path],
        max_traces: Optional[int] = None,
        sort_files: bool = True,
        recursive: bool = True,
    ) -> DatasetLoadReport:
        """Loads PlanetLab CPU traces from a directory tree.

        Args:
            directory_path: Root path to scan for ``.txt`` trace files.
            max_traces: Maximum number of valid traces to load (None = all).
            sort_files: Sort candidate files deterministically by name (default: True).
            recursive: Search subdirectories recursively (default: True).

        Returns:
            DatasetLoadReport containing valid VMTrace objects and skipped file mapping.

        Raises:
            FileNotFoundError: If the directory does not exist.
            NotADirectoryError: If the path exists but is not a directory.
            ValueError: If no valid traces are found.
        """
        dir_path = Path(directory_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: '{directory_path}'")
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: '{directory_path}'")

        self._skipped.clear()
        valid_traces: List[VMTrace] = []

        # Gather candidate .txt files
        glob = dir_path.rglob("*.txt") if recursive else dir_path.glob("*.txt")
        candidates: List[Path] = [
            p for p in glob
            if p.is_file() and not any(part.startswith(".") for part in p.parts)
        ]

        if sort_files:
            # Sort lexicographically by full path for determinism
            candidates.sort(key=lambda p: str(p))

        for file_path in candidates:
            if max_traces is not None and len(valid_traces) >= max_traces:
                break
            try:
                trace = _parse_planetlab_file(file_path)
                valid_traces.append(trace)
            except (ValueError, OSError) as exc:
                self._skipped[str(file_path)] = str(exc)
                logger.debug("Skipped '%s': %s", file_path.name, exc)

        if not valid_traces:
            raise ValueError(
                f"No valid PlanetLab traces found under '{directory_path}'. "
                f"Skipped {len(self._skipped)} files."
            )

        logger.info(
            "PlanetLabLoader: loaded %d traces, skipped %d from '%s'.",
            len(valid_traces), len(self._skipped), directory_path,
        )
        return DatasetLoadReport(
            valid_traces=valid_traces,
            skipped_files=dict(self._skipped),
        )

    def load_traces(
        self,
        directory_path: Union[str, Path],
        max_traces: Optional[int] = None,
        sort_files: bool = True,
        recursive: bool = True,
    ) -> List[VMTrace]:
        """Convenience wrapper returning only the list of valid VMTrace objects."""
        return self.load_with_report(
            directory_path=directory_path,
            max_traces=max_traces,
            sort_files=sort_files,
            recursive=recursive,
        ).valid_traces


def load_planetlab_traces(
    directory_path: Union[str, Path],
    max_traces: Optional[int] = None,
    recursive: bool = True,
) -> List[VMTrace]:
    """Module-level convenience helper to load PlanetLab traces.

    Args:
        directory_path: Root directory containing PlanetLab ``.txt`` files.
        max_traces: Optional cap on number of loaded traces.
        recursive: Whether to search subdirectories (default: True).

    Returns:
        List of VMTrace instances sorted lexicographically by file path.
    """
    return PlanetLabLoader().load_traces(
        directory_path=directory_path,
        max_traces=max_traces,
        recursive=recursive,
    )
