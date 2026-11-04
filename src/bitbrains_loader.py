"""Bitbrains GWA-T-12 dataset loader for ThermoPlace.

Loads VM CPU utilization trace files from the Bitbrains fastStorage dataset.
Each CSV file represents one VM's monitoring trace sampled at 5-minute intervals.

CSV Format (semicolon + tab separated, header on row 1):
    Timestamp [ms]; \tCPU cores; \tCPU capacity provisioned [MHZ]; \tCPU usage [MHZ];
    \tCPU usage [%]; \tMemory capacity provisioned [KB]; \tMemory usage [KB]; ...

This loader extracts the ``CPU usage [%]`` column only, which represents the
percentage of provisioned CPU capacity in use at each sample interval.

Scientific note: RAM consumption is provisioned from standard VM flavor profiles
(e.g., [2, 4, 8, 16] GB) since memory measurements in the Bitbrains traces are
recorded in KB as absolute values, not as a percentage of a known host capacity.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from src.dataset_loader import DatasetLoadReport, VMTrace

logger = logging.getLogger(__name__)

# Column name after stripping leading/trailing whitespace from each header token
_CPU_USAGE_COLUMN = "CPU usage [%]"


def _parse_bitbrains_file(file_path: Path) -> VMTrace:
    """Parses a single Bitbrains CSV trace file into a VMTrace object.

    Args:
        file_path: Path to a Bitbrains-format CSV file.

    Returns:
        VMTrace containing chronological CPU utilization samples in [0.0, 100.0].

    Raises:
        ValueError: If the file is empty, missing the CPU usage column, or all
                    values are non-finite.
        OSError: If the file cannot be read.
    """
    try:
        # Bitbrains CSVs use "; \t" (semicolon followed by a tab) as the separator
        df = pd.read_csv(file_path, sep=";\t", engine="python")
    except Exception as exc:
        raise OSError(f"Could not parse '{file_path.name}': {exc}") from exc

    # Strip whitespace from column names to handle leading tabs
    df.columns = [str(c).strip() for c in df.columns]

    if _CPU_USAGE_COLUMN not in df.columns:
        raise ValueError(
            f"Expected column '{_CPU_USAGE_COLUMN}' not found in '{file_path.name}'. "
            f"Found: {df.columns.tolist()}"
        )

    raw_series = df[_CPU_USAGE_COLUMN]
    if raw_series.empty:
        raise ValueError(f"No data rows found in '{file_path.name}'.")

    # Drop NaN / infinite values and convert to float
    cleaned = pd.to_numeric(raw_series, errors="coerce").dropna()
    cleaned = cleaned[cleaned.between(0.0, 100.0)]

    if cleaned.empty:
        raise ValueError(
            f"All CPU usage values in '{file_path.name}' are invalid or out of range [0, 100]."
        )

    return VMTrace(
        vm_id=file_path.stem,
        cpu_samples=[float(v) for v in cleaned],
        source_file=str(file_path),
    )


class BitbrainsLoader:
    """Loads VM CPU utilization traces from a Bitbrains GWA-T-12 dataset directory.

    Typical directory layout:
        <dataset_dir>/
            <vm_id>.csv
            <vm_id>.csv
            ...
    or nested inside month subdirectories:
        <dataset_dir>/fastStorage/2013-8/<vm_id>.csv
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
        """Loads Bitbrains CSV traces from a directory tree.

        Args:
            directory_path: Root path to scan for ``.csv`` files.
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

        # Gather candidate CSV files
        glob = dir_path.rglob("*.csv") if recursive else dir_path.glob("*.csv")
        candidates: List[Path] = [
            p for p in glob
            if p.is_file() and not any(part.startswith(".") for part in p.parts)
        ]

        if sort_files:
            # Sort numerically by stem if stem is an integer, else lexicographically
            def sort_key(p: Path):
                try:
                    return (0, int(p.stem), str(p))
                except ValueError:
                    return (1, 0, str(p))

            candidates.sort(key=sort_key)

        for file_path in candidates:
            if max_traces is not None and len(valid_traces) >= max_traces:
                break
            try:
                trace = _parse_bitbrains_file(file_path)
                valid_traces.append(trace)
            except (ValueError, OSError) as exc:
                self._skipped[str(file_path)] = str(exc)
                logger.debug("Skipped '%s': %s", file_path.name, exc)

        if not valid_traces:
            raise ValueError(
                f"No valid Bitbrains CSV traces found under '{directory_path}'. "
                f"Skipped {len(self._skipped)} files."
            )

        logger.info(
            "BitbrainsLoader: loaded %d traces, skipped %d from '%s'.",
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


def load_bitbrains_traces(
    directory_path: Union[str, Path],
    max_traces: Optional[int] = None,
    recursive: bool = True,
) -> List[VMTrace]:
    """Module-level convenience helper to load Bitbrains traces.

    Args:
        directory_path: Root directory containing Bitbrains CSV files.
        max_traces: Optional cap on number of loaded traces.
        recursive: Whether to search subdirectories (default: True).

    Returns:
        List of VMTrace instances sorted numerically by VM ID.
    """
    return BitbrainsLoader().load_traces(
        directory_path=directory_path,
        max_traces=max_traces,
        recursive=recursive,
    )

