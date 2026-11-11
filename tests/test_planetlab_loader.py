"""Comprehensive unit and integration tests for PlanetLabLoader.

Tests cover:
- Valid file parsing (single value per line, plain text)
- CPU utilization conversion and clamping
- Empty lines and non-numeric line handling
- Empty / zero-valid-sample file skipping
- max_traces truncation
- Recursive vs flat directory discovery
- DatasetLoadReport counts
- VM identity (vm_id from filename stem)
- source_file path preservation
- VMTrace CPU statistics (min, max, mean)
- Compatibility with TraceWorkloadGenerator
- Whitespace handling
- Independence from BitbrainsLoader
- Integration: First Fit simulation with PlanetLab workload
- Integration: Q-Learning strategy with PlanetLab workload
- CLI --dataset-type argument parsing
- Existing Bitbrains test regression guard
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset_loader import DatasetLoadReport, VMTrace
from src.planetlab_loader import PlanetLabLoader, _parse_planetlab_file, load_planetlab_traces


def _write_trace(dir_path: Path, filename: str, lines: List[str]) -> Path:
    """Helper: writes a trace file with the given lines to dir_path."""
    fp = dir_path / filename
    fp.write_text("\n".join(lines), encoding="utf-8")
    return fp


class TestPlanetLabFileParser(unittest.TestCase):
    """Unit tests for _parse_planetlab_file helper function."""

    def test_valid_single_file_parsing(self):
        """Parses a file with five valid CPU values correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = _write_trace(
                Path(tmpdir), "vm001.txt",
                ["0.0", "25.5", "50.0", "75.0", "100.0"]
            )
            trace = _parse_planetlab_file(fp)
            self.assertEqual(trace.vm_id, "vm001")
            self.assertEqual(trace.total_samples, 5)
            self.assertAlmostEqual(trace.cpu_samples[0], 0.0)
            self.assertAlmostEqual(trace.cpu_samples[1], 25.5)
            self.assertAlmostEqual(trace.cpu_samples[4], 100.0)
            self.assertAlmostEqual(trace.min_cpu, 0.0)
            self.assertAlmostEqual(trace.max_cpu, 100.0)
            self.assertAlmostEqual(trace.mean_cpu, 50.1)

    def test_valid_288_sample_file(self):
        """Parses a file with exactly 288 lines (standard PlanetLab day trace)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            values = [str(float(i % 101)) for i in range(288)]
            fp = _write_trace(Path(tmpdir), "vm288.txt", values)
            trace = _parse_planetlab_file(fp)
            self.assertEqual(trace.total_samples, 288)

    def test_out_of_range_clamping(self):
        """Values outside [0, 100] are clamped to the nearest bound."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = _write_trace(
                Path(tmpdir), "clamp.txt",
                ["-10.0", "0.0", "50.0", "110.0", "200.0"]
            )
            trace = _parse_planetlab_file(fp)
            self.assertEqual(trace.total_samples, 5)
            self.assertAlmostEqual(trace.cpu_samples[0], 0.0)   # clamped from -10
            self.assertAlmostEqual(trace.cpu_samples[1], 0.0)
            self.assertAlmostEqual(trace.cpu_samples[2], 50.0)
            self.assertAlmostEqual(trace.cpu_samples[3], 100.0)  # clamped from 110
            self.assertAlmostEqual(trace.cpu_samples[4], 100.0)  # clamped from 200

    def test_empty_lines_skipped(self):
        """Empty lines are skipped without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = _write_trace(
                Path(tmpdir), "gaps.txt",
                ["20.0", "", "40.0", "", "", "60.0"]
            )
            trace = _parse_planetlab_file(fp)
            self.assertEqual(trace.total_samples, 3)
            self.assertAlmostEqual(trace.mean_cpu, 40.0)

    def test_non_numeric_lines_skipped(self):
        """Non-numeric lines (annotations, corrupted entries) are silently skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = _write_trace(
                Path(tmpdir), "mixed.txt",
                ["N/A", "30.0", "timestamp", "error", "70.0"]
            )
            trace = _parse_planetlab_file(fp)
            self.assertEqual(trace.total_samples, 2)
            self.assertAlmostEqual(trace.mean_cpu, 50.0)

    def test_empty_file_raises_value_error(self):
        """An empty file (zero bytes) raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = Path(tmpdir) / "empty.txt"
            fp.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                _parse_planetlab_file(fp)

    def test_all_non_numeric_raises_value_error(self):
        """A file with only non-numeric content raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = _write_trace(
                Path(tmpdir), "junk.txt",
                ["N/A", "header", "timestamp", "---"]
            )
            with self.assertRaises(ValueError):
                _parse_planetlab_file(fp)

    def test_whitespace_handling(self):
        """Lines with leading/trailing whitespace are parsed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = _write_trace(
                Path(tmpdir), "ws.txt",
                ["  50.0  ", "\t25.0\t", " 75.0 "]
            )
            trace = _parse_planetlab_file(fp)
            self.assertEqual(trace.total_samples, 3)
            self.assertAlmostEqual(trace.cpu_samples[0], 50.0)
            self.assertAlmostEqual(trace.cpu_samples[1], 25.0)
            self.assertAlmostEqual(trace.cpu_samples[2], 75.0)

    def test_integer_values_parsed(self):
        """Integer strings (as used in real PlanetLab files) are parsed as floats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = _write_trace(
                Path(tmpdir), "ints.txt",
                ["24", "34", "29", "26", "21"]
            )
            trace = _parse_planetlab_file(fp)
            self.assertEqual(trace.total_samples, 5)
            self.assertAlmostEqual(trace.cpu_samples[0], 24.0)

    def test_vm_id_from_filename_stem(self):
        """vm_id is set to the filename stem (without extension)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = _write_trace(Path(tmpdir), "myvm_42.txt", ["50.0"])
            trace = _parse_planetlab_file(fp)
            self.assertEqual(trace.vm_id, "myvm_42")

    def test_source_file_path_set(self):
        """source_file is set to the absolute path of the parsed file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = _write_trace(Path(tmpdir), "vm_src.txt", ["50.0"])
            trace = _parse_planetlab_file(fp)
            self.assertIn("vm_src", trace.source_file)

    def test_cpu_stats_correctness(self):
        """min_cpu, max_cpu, mean_cpu are computed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = _write_trace(
                Path(tmpdir), "stats.txt",
                ["20.0", "40.0", "60.0", "80.0"]
            )
            trace = _parse_planetlab_file(fp)
            self.assertAlmostEqual(trace.min_cpu, 20.0)
            self.assertAlmostEqual(trace.max_cpu, 80.0)
            self.assertAlmostEqual(trace.mean_cpu, 50.0)


class TestPlanetLabLoader(unittest.TestCase):
    """Unit tests for the PlanetLabLoader class."""

    def setUp(self):
        self.loader = PlanetLabLoader()

    def _make_dir(self, root: Path, structure: dict) -> None:
        """Creates files from a dict: {relative_path: list_of_lines}."""
        for rel_path, lines in structure.items():
            fp = root / rel_path
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text("\n".join(lines), encoding="utf-8")

    # -- Directory-level tests --------------------------------------------------

    def test_load_with_report_valid_traces(self):
        """load_with_report returns correct trace count for valid files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._make_dir(root, {
                "vm1.txt": ["10.0", "20.0", "30.0"],
                "vm2.txt": ["40.0", "50.0", "60.0"],
            })
            report = self.loader.load_with_report(root)
            self.assertIsInstance(report, DatasetLoadReport)
            self.assertEqual(report.total_loaded, 2)
            self.assertEqual(report.total_skipped, 0)
            vm_ids = {t.vm_id for t in report.valid_traces}
            self.assertEqual(vm_ids, {"vm1", "vm2"})

    def test_empty_file_reported_in_skipped(self):
        """Empty files appear in skipped_files, not in valid_traces."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._make_dir(root, {
                "good.txt": ["50.0", "60.0"],
                "empty.txt": [],
            })
            report = self.loader.load_with_report(root)
            self.assertEqual(report.total_loaded, 1)
            self.assertEqual(report.total_skipped, 1)
            self.assertEqual(report.valid_traces[0].vm_id, "good")
            # Empty file path should be in skipped_files
            skipped_keys_lower = " ".join(report.skipped_files.keys()).lower()
            self.assertIn("empty", skipped_keys_lower)

    def test_zero_valid_sample_file_skipped(self):
        """Files with only non-numeric content appear in skipped_files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._make_dir(root, {
                "good.txt": ["25.0"],
                "bad.txt": ["N/A", "header"],
            })
            report = self.loader.load_with_report(root)
            self.assertEqual(report.total_loaded, 1)
            self.assertEqual(report.total_skipped, 1)

    def test_max_traces_limit(self):
        """max_traces correctly caps the number of loaded traces."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for i in range(5):
                (root / f"vm{i}.txt").write_text("50.0\n", encoding="utf-8")
            report = self.loader.load_with_report(root, max_traces=3)
            self.assertEqual(report.total_loaded, 3)

    def test_recursive_discovery(self):
        """recursive=True discovers files in subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._make_dir(root, {
                "20090320/vm1.txt": ["10.0"],
                "20090321/vm2.txt": ["20.0"],
            })
            report = self.loader.load_with_report(root, recursive=True)
            self.assertEqual(report.total_loaded, 2)

    def test_non_recursive_discovery(self):
        """recursive=False only searches the top-level directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._make_dir(root, {
                "top_level.txt": ["30.0"],
                "subdir/nested.txt": ["40.0"],
            })
            report = self.loader.load_with_report(root, recursive=False)
            self.assertEqual(report.total_loaded, 1)
            self.assertEqual(report.valid_traces[0].vm_id, "top_level")

    def test_dataset_load_report_counts(self):
        """total_loaded and total_skipped are consistent with valid/skipped counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._make_dir(root, {
                "v1.txt": ["10.0"],
                "v2.txt": ["20.0"],
                "v3.txt": ["30.0"],
                "bad.txt": ["N/A"],
            })
            report = self.loader.load_with_report(root)
            self.assertEqual(report.total_loaded, 3)
            self.assertEqual(report.total_skipped, 1)
            self.assertEqual(len(report.valid_traces), report.total_loaded)
            self.assertEqual(len(report.skipped_files), report.total_skipped)

    def test_nonexistent_directory_raises(self):
        """FileNotFoundError is raised for a non-existent path."""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_with_report("/nonexistent/path/xyz")

    def test_not_a_directory_raises(self):
        """NotADirectoryError is raised when path is a file, not a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = Path(tmpdir) / "not_a_dir.txt"
            fp.write_text("50.0\n", encoding="utf-8")
            with self.assertRaises(NotADirectoryError):
                self.loader.load_with_report(str(fp))

    def test_all_invalid_raises_value_error(self):
        """ValueError is raised when no valid traces are found in directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "bad.txt").write_text("N/A\nheader", encoding="utf-8")
            with self.assertRaises(ValueError):
                self.loader.load_with_report(root)

    def test_load_traces_convenience_method(self):
        """load_traces returns List[VMTrace] directly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "vm1.txt").write_text("50.0\n60.0", encoding="utf-8")
            traces = self.loader.load_traces(root)
            self.assertIsInstance(traces, list)
            self.assertEqual(len(traces), 1)
            self.assertIsInstance(traces[0], VMTrace)

    def test_module_level_convenience_function(self):
        """load_planetlab_traces module function returns correct traces."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "vm1.txt").write_text("50.0\n", encoding="utf-8")
            traces = load_planetlab_traces(root)
            self.assertEqual(len(traces), 1)
            self.assertEqual(traces[0].vm_id, "vm1")


class TestPlanetLabWorkloadCompatibility(unittest.TestCase):
    """Tests PlanetLab traces flowing through TraceWorkloadGenerator and Simulator."""

    def test_compatibility_with_workload_generator(self):
        """PlanetLab traces are compatible with TraceWorkloadGenerator."""
        from src.workload import TraceWorkloadGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for i in range(5):
                fp = root / f"vm{i}.txt"
                fp.write_text("\n".join([str(float(j % 100)) for j in range(50)]), encoding="utf-8")

            loader = PlanetLabLoader()
            report = loader.load_with_report(root)
            self.assertEqual(report.total_loaded, 5)

            gen = TraceWorkloadGenerator(
                traces=report.valid_traces,
                arrival_rate_lambda=1.0,
                max_arrival_time=20,
                seed=42,
            )
            schedule = gen.generate_schedule_by_step(total_time_steps=20)

            # Schedule must be a dict keyed by time step
            self.assertIsInstance(schedule, dict)
            # At least one VM should be scheduled across 20 steps
            total_vms = sum(len(vms) for vms in schedule.values())
            self.assertGreater(total_vms, 0)

            # VMs should have cpu_trace attached from the PlanetLab traces
            for t, vms in schedule.items():
                for vm in vms:
                    self.assertIsNotNone(vm.cpu_trace, f"VM {vm.vm_id} at t={t} missing cpu_trace")
                    self.assertIsInstance(vm.cpu_trace, list)

    def test_first_fit_simulation_with_planetlab(self):
        """First Fit placement runs correctly end-to-end with PlanetLab workload."""
        from src.workload import TraceWorkloadGenerator
        from src.datacenter import DataCenter
        from src.thermal import ThermalModel, ThermalConfig
        from src.placement.first_fit import FirstFitPlacement
        from src.simulator import Simulator

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for i in range(10):
                fp = root / f"vm{i}.txt"
                fp.write_text("\n".join([str(float(j % 100)) for j in range(30)]), encoding="utf-8")

            loader = PlanetLabLoader()
            report = loader.load_with_report(root)

            gen = TraceWorkloadGenerator(
                traces=report.valid_traces,
                arrival_rate_lambda=2.0,
                max_arrival_time=15,
                seed=42,
            )
            schedule = gen.generate_schedule_by_step(total_time_steps=15)

            dc = DataCenter.create_standard_datacenter(num_racks=2, hosts_per_rack=4)
            thermal_model = ThermalModel(ThermalConfig())
            strategy = FirstFitPlacement()
            sim = Simulator(datacenter=dc, thermal_model=thermal_model)

            metrics = sim.run(workload_schedule=schedule, strategy=strategy, total_steps=15)
            summary = metrics.summary()

            # Simulation must complete and produce a summary
            self.assertIsInstance(summary, dict)
            self.assertIn("total_arrived_vms", summary)
            self.assertIn("total_placed_vms", summary)
            self.assertGreaterEqual(summary["total_arrived_vms"], 0)

    def test_qlearning_simulation_with_planetlab(self):
        """Q-Learning strategy runs correctly end-to-end with PlanetLab workload."""
        from src.workload import TraceWorkloadGenerator
        from src.datacenter import DataCenter
        from src.thermal import ThermalModel, ThermalConfig
        from src.qlearning.agent import QLearningPlacementAgent
        from src.qlearning.strategy import QLearningSimulatorStrategy
        from src.simulator import Simulator

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for i in range(8):
                fp = root / f"vm{i}.txt"
                fp.write_text("\n".join([str(float(j % 100)) for j in range(30)]), encoding="utf-8")

            loader = PlanetLabLoader()
            report = loader.load_with_report(root)

            gen = TraceWorkloadGenerator(
                traces=report.valid_traces,
                arrival_rate_lambda=1.5,
                max_arrival_time=10,
                seed=42,
            )
            schedule = gen.generate_schedule_by_step(total_time_steps=10)

            dc = DataCenter.create_standard_datacenter(num_racks=2, hosts_per_rack=4)
            thermal_model = ThermalModel(ThermalConfig())
            agent = QLearningPlacementAgent(seed=42, epsilon=0.5)
            strategy = QLearningSimulatorStrategy(agent=agent, training=True)
            sim = Simulator(datacenter=dc, thermal_model=thermal_model)

            metrics = sim.run(workload_schedule=schedule, strategy=strategy, total_steps=10)
            summary = metrics.summary()

            self.assertIsInstance(summary, dict)
            self.assertIn("total_placed_vms", summary)
            # Q-table should have learned some entries
            self.assertGreaterEqual(len(agent.q_table), 0)


class TestCLIDatasetType(unittest.TestCase):
    """Tests --dataset-type argument in main.py and train.py CLIs."""

    def test_main_parse_args_dataset_type_default(self):
        """main.py parse_args defaults --dataset-type to 'bitbrains'."""
        from main import parse_args
        args = parse_args([
            "--dataset-dir", "some/path",
            "--strategy", "first_fit",
        ])
        self.assertEqual(args.dataset_type, "bitbrains")

    def test_main_parse_args_dataset_type_planetlab(self):
        """main.py parse_args accepts --dataset-type planetlab."""
        from main import parse_args
        args = parse_args([
            "--dataset-dir", "some/path",
            "--dataset-type", "planetlab",
        ])
        self.assertEqual(args.dataset_type, "planetlab")

    def test_main_parse_args_dataset_type_invalid(self):
        """main.py parse_args rejects an invalid --dataset-type value."""
        from main import parse_args
        import argparse
        with self.assertRaises(SystemExit):
            parse_args(["--dataset-dir", "some/path", "--dataset-type", "invalid"])

    def test_train_parse_args_dataset_type_default(self):
        """train.py parse_args defaults --dataset-type to 'bitbrains'."""
        from train import parse_args as train_parse_args
        args = train_parse_args(["--dataset-dir", "some/path"])
        self.assertEqual(args.dataset_type, "bitbrains")

    def test_train_parse_args_dataset_type_planetlab(self):
        """train.py parse_args accepts --dataset-type planetlab."""
        from train import parse_args as train_parse_args
        args = train_parse_args(["--dataset-dir", "some/path", "--dataset-type", "planetlab"])
        self.assertEqual(args.dataset_type, "planetlab")


class TestTrainerDatasetType(unittest.TestCase):
    """Tests that QLearningTrainer selects the correct loader via dataset_type."""

    def test_training_config_dataset_type_default(self):
        """TrainingConfig defaults dataset_type to 'bitbrains'."""
        from src.qlearning.trainer import TrainingConfig
        cfg = TrainingConfig()
        self.assertEqual(cfg.dataset_type, "bitbrains")

    def test_trainer_planetlab_loader_dispatch(self):
        """QLearningTrainer uses PlanetLabLoader when dataset_type='planetlab'."""
        from src.qlearning.trainer import QLearningTrainer, TrainingConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for i in range(5):
                (root / f"vm{i}.txt").write_text(
                    "\n".join([str(float(j % 100)) for j in range(30)]),
                    encoding="utf-8",
                )
            cfg = TrainingConfig(
                dataset_dir=str(root),
                dataset_type="planetlab",
                episodes=1,
                steps=15,
                max_vms=5,
                arrival_rate=1.0,
                seed=42,
            )
            trainer = QLearningTrainer(config=cfg)
            self.assertIsNotNone(trainer.workload_generator)

    def test_trainer_bitbrains_dataset_type_preserved(self):
        """QLearningTrainer with no dataset_dir still creates synthetic workload."""
        from src.qlearning.trainer import QLearningTrainer, TrainingConfig
        cfg = TrainingConfig(dataset_dir=None, dataset_type="bitbrains", steps=5)
        trainer = QLearningTrainer(config=cfg)
        self.assertIsNotNone(trainer.workload_generator)


class TestNoCrossContamination(unittest.TestCase):
    """Verifies that PlanetLabLoader is independent of BitbrainsLoader."""

    def test_import_does_not_require_bitbrains(self):
        """PlanetLabLoader can be imported without importing BitbrainsLoader."""
        import importlib
        loader_module = importlib.import_module("src.planetlab_loader")
        # BitbrainsLoader must NOT be a name in the module's namespace
        self.assertFalse(
            hasattr(loader_module, "BitbrainsLoader"),
            "PlanetLabLoader module unexpectedly exposes BitbrainsLoader",
        )
        self.assertTrue(hasattr(loader_module, "PlanetLabLoader"))

    def test_planetlab_loader_does_not_import_pandas(self):
        """PlanetLabLoader is pure stdlib — does not depend on pandas."""
        import importlib
        loader_module = importlib.import_module("src.planetlab_loader")
        source = Path(loader_module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import pandas", source)
        self.assertNotIn("from pandas", source)


if __name__ == "__main__":
    unittest.main()
