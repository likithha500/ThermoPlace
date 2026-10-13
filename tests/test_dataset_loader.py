"""Unit tests for the PlanetLab dataset loader.

Covers:
- Missing dataset directory (FileNotFoundError)
- Empty directory (ValueError)
- Malformed data (non-numeric tokens, NaN/Inf, empty files)
- Invalid CPU utilization values (< 0% or > 100%)
- Preservation of original sample order without fabrication or imputation
- Real PlanetLab trace validation
"""

from pathlib import Path
import tempfile
import unittest

from src.dataset_loader import (
    PlanetLabDatasetLoader,
    VMTrace,
    load_planetlab_directory,
    parse_planetlab_file,
)


class TestPlanetLabDatasetLoader(unittest.TestCase):
    def setUp(self):
        self.loader = PlanetLabDatasetLoader()

    # -------------------------------------------------------------------------
    # 1. Missing Directory & File Tests
    # -------------------------------------------------------------------------

    def test_missing_dataset_directory(self):
        """Loader must raise FileNotFoundError with a clear message if directory does not exist."""
        with self.assertRaises(FileNotFoundError) as ctx:
            self.loader.load_directory("non_existent_dataset_directory_xyz123")
        self.assertIn("Dataset directory not found", str(ctx.exception))

    def test_missing_trace_file(self):
        """Parser must raise FileNotFoundError if a specific trace file does not exist."""
        with self.assertRaises(FileNotFoundError) as ctx:
            self.loader.parse_trace_file("non_existent_file_trace_xyz123")
        self.assertIn("Trace file not found", str(ctx.exception))

    def test_path_is_not_a_directory(self):
        """load_directory must raise NotADirectoryError when given a regular file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "some_file.txt"
            file_path.write_text("50\n", encoding="utf-8")
            with self.assertRaises(NotADirectoryError) as ctx:
                self.loader.load_directory(file_path)
            self.assertIn("Path is not a directory", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 2. Empty Directory Tests
    # -------------------------------------------------------------------------

    def test_empty_directory_raises_value_error(self):
        """Loader must raise ValueError if the directory exists but contains no valid trace files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError) as ctx:
                self.loader.load_directory(tmp_dir)
            self.assertIn("No valid trace files found", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 3. Malformed Data Tests
    # -------------------------------------------------------------------------

    def test_malformed_data_non_numeric_token(self):
        """Parser must raise ValueError reporting line number and token if line is non-numeric."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            corrupt_file = Path(tmp_dir) / "corrupt_token_trace"
            corrupt_file.write_text("10\n20\nMALFORMED_VALUE\n30\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self.loader.parse_trace_file(corrupt_file)
            self.assertIn("Invalid non-numeric value 'MALFORMED_VALUE'", str(ctx.exception))
            self.assertIn("line 3", str(ctx.exception))

    def test_malformed_data_nan_and_inf_tokens(self):
        """Parser must reject non-finite float tokens (NaN, inf, -inf) as malformed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nan_file = Path(tmp_dir) / "nan_trace"
            nan_file.write_text("15\nNaN\n25\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self.loader.parse_trace_file(nan_file)
            self.assertIn("non-finite", str(ctx.exception))

            inf_file = Path(tmp_dir) / "inf_trace"
            inf_file.write_text("15\ninf\n25\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self.loader.parse_trace_file(inf_file)
            self.assertIn("non-finite", str(ctx.exception))

    def test_malformed_empty_file(self):
        """Parser must raise ValueError if trace file contains no samples (empty or whitespace)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            empty_file = Path(tmp_dir) / "empty_trace"
            empty_file.write_text("   \n\n  \n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self.loader.parse_trace_file(empty_file)
            self.assertIn("contains no valid CPU utilization samples", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 4. Invalid CPU Utilization Range Tests
    # -------------------------------------------------------------------------

    def test_invalid_cpu_negative_utilization(self):
        """Parser must raise ValueError if CPU utilization is negative (< 0.0)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            neg_file = Path(tmp_dir) / "neg_trace"
            neg_file.write_text("10\n-0.5\n20\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self.loader.parse_trace_file(neg_file)
            self.assertIn("out of valid range", str(ctx.exception))

    def test_invalid_cpu_utilization_exceeds_100_percent(self):
        """Parser must raise ValueError if CPU utilization exceeds 100.0%."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            over_file = Path(tmp_dir) / "over_trace"
            over_file.write_text("10\n100.5\n20\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self.loader.parse_trace_file(over_file)
            self.assertIn("out of valid range", str(ctx.exception))

    def test_custom_cpu_thresholds(self):
        """Loader thresholds can be customized and validate boundaries strictly."""
        custom_loader = PlanetLabDatasetLoader(min_allowed_cpu=10.0, max_allowed_cpu=80.0)
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "boundary_trace"
            file_path.write_text("5.0\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                custom_loader.parse_trace_file(file_path)
            self.assertIn("out of valid range [10.0, 80.0]", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 5. Order Preservation & Non-Fabrication of Missing Samples
    # -------------------------------------------------------------------------

    def test_preserves_original_sample_order(self):
        """Parser must strictly preserve the original chronological sequence of samples."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "ordered_trace"
            original_values = [14.0, 92.5, 3.0, 48.0, 0.0, 77.2, 5.0, 100.0]
            file_path.write_text("\n".join(str(v) for v in original_values) + "\n", encoding="utf-8")

            trace = self.loader.parse_trace_file(file_path)

            # Exactly matches original values and order without altering or sorting
            self.assertEqual(trace.cpu_samples, original_values)
            self.assertEqual(trace.total_samples, len(original_values))
            self.assertEqual(trace[0], 14.0)
            self.assertEqual(trace[1], 92.5)
            self.assertEqual(trace[-1], 100.0)
            self.assertEqual(trace.min_cpu, 0.0)
            self.assertEqual(trace.max_cpu, 100.0)
            self.assertAlmostEqual(trace.mean_cpu, sum(original_values) / len(original_values), places=4)

    def test_does_not_fabricate_missing_samples(self):
        """Parser must reject files with internal gaps instead of inventing/interpolating samples."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            gap_file = Path(tmp_dir) / "missing_epoch_trace"
            # Missing sample between line 2 (20) and line 4 (30)
            gap_file.write_text("10\n20\n\n30\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self.loader.parse_trace_file(gap_file)
            self.assertIn("Missing sample / internal empty line", str(ctx.exception))
            self.assertIn("Traces must be continuous without missing epochs", str(ctx.exception))

    def test_exact_sample_count_retained_without_synthetic_padding(self):
        """Parser must return the exact recorded count and never pad with synthetic samples."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "short_trace"
            sample_data = "10\n20\n30\n"
            file_path.write_text(sample_data, encoding="utf-8")

            trace = self.loader.parse_trace_file(file_path)
            # Exactly 3 samples — no synthetic padding to 288 or 100
            self.assertEqual(trace.total_samples, 3)
            self.assertEqual(len(trace), 3)
            self.assertEqual(trace.cpu_samples, [10.0, 20.0, 30.0])

    # -------------------------------------------------------------------------
    # 6. Skipped File Reporting & Directory Loading
    # -------------------------------------------------------------------------

    def test_load_directory_skips_invalid_files_and_reports(self):
        """Loader must skip invalid files, record them in the report, and load valid files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dir_path = Path(tmp_dir)
            (dir_path / "vm_good1").write_text("10\n20\n", encoding="utf-8")
            (dir_path / "vm_good2").write_text("30\n40\n", encoding="utf-8")
            (dir_path / "vm_bad_corrupt").write_text("10\nNON_NUMERIC\n", encoding="utf-8")
            (dir_path / "vm_bad_empty").write_text("", encoding="utf-8")
            (dir_path / "vm_bad_range").write_text("150\n", encoding="utf-8")

            report = self.loader.load_with_report(dir_path)

            self.assertEqual(report.total_loaded, 2)
            loaded_ids = [t.vm_id for t in report.valid_traces]
            self.assertEqual(loaded_ids, ["vm_good1", "vm_good2"])

            self.assertEqual(report.total_skipped, 3)
            skipped_filenames = [Path(p).name for p in report.skipped_files.keys()]
            self.assertIn("vm_bad_corrupt", skipped_filenames)
            self.assertIn("vm_bad_empty", skipped_filenames)
            self.assertIn("vm_bad_range", skipped_filenames)

    def test_load_directory_deterministic_sorting(self):
        """Loader sorts candidate files deterministically by filename."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dir_path = Path(tmp_dir)
            (dir_path / "vm_c").write_text("30\n", encoding="utf-8")
            (dir_path / "vm_a").write_text("10\n", encoding="utf-8")
            (dir_path / "vm_b").write_text("20\n", encoding="utf-8")

            traces = self.loader.load_directory(dir_path, sort_files=True)
            self.assertEqual([t.vm_id for t in traces], ["vm_a", "vm_b", "vm_c"])

    def test_load_directory_max_files_limit(self):
        """max_files limits the number of loaded valid traces."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dir_path = Path(tmp_dir)
            for i in range(5):
                (dir_path / f"vm_{i:02d}").write_text(f"{i*10}\n", encoding="utf-8")

            traces = self.loader.load_directory(dir_path, max_files=2)
            self.assertEqual(len(traces), 2)
            self.assertEqual([t.vm_id for t in traces], ["vm_00", "vm_01"])

    def test_load_directory_recursive_subfolders(self):
        """recursive=True discovers and loads traces from date subdirectories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_dir = Path(tmp_dir)
            sub1 = root_dir / "20110303"
            sub2 = root_dir / "20110306"
            sub1.mkdir()
            sub2.mkdir()
            (sub1 / "trace_a").write_text("10\n20\n", encoding="utf-8")
            (sub2 / "trace_b").write_text("30\n40\n", encoding="utf-8")

            traces = self.loader.load_directory(root_dir, recursive=True)
            self.assertEqual(len(traces), 2)
            vm_ids = [t.vm_id for t in traces]
            self.assertIn("trace_a", vm_ids)
            self.assertIn("trace_b", vm_ids)

    # -------------------------------------------------------------------------
    # 7. Real PlanetLab Dataset Integration Test
    # -------------------------------------------------------------------------

    def test_real_dataset_traces_if_present(self):
        """Validates that real PlanetLab trace files load with 288 samples in [0, 100]."""
        base_dir = Path("data/raw/planetlab")
        if base_dir.exists() and base_dir.is_dir():
            day_dirs = [d for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
            if day_dirs:
                sample_day = day_dirs[0]
                traces = self.loader.load_directory(sample_day, max_files=5)
                self.assertEqual(len(traces), 5)
                for trace in traces:
                    self.assertEqual(trace.total_samples, 288)
                    for sample in trace.cpu_samples:
                        self.assertGreaterEqual(sample, 0.0)
                        self.assertLessEqual(sample, 100.0)


if __name__ == "__main__":
    unittest.main()
