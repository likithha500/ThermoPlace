"""Unit tests for dataset loader data structures and reporting models."""

import unittest
from src.dataset_loader import VMTrace, DatasetLoadReport


class TestDatasetLoaderModels(unittest.TestCase):
    def test_vmtrace_initialization_and_properties(self):
        """VMTrace correctly computes summary metrics across chronological samples."""
        samples = [10.0, 50.0, 90.0, 30.0]
        trace = VMTrace(
            vm_id="vm_test_01",
            cpu_samples=samples,
            source_file="/test/path/vm_test_01",
        )

        self.assertEqual(trace.vm_id, "vm_test_01")
        self.assertEqual(trace.source_file, "/test/path/vm_test_01")
        self.assertEqual(trace.total_samples, 4)
        self.assertEqual(len(trace), 4)
        self.assertEqual(trace.min_cpu, 10.0)
        self.assertEqual(trace.max_cpu, 90.0)
        self.assertAlmostEqual(trace.mean_cpu, 45.0)

        # Direct indexing, slicing, iteration
        self.assertEqual(trace[0], 10.0)
        self.assertEqual(trace[-1], 30.0)
        self.assertEqual(trace[1:3], [50.0, 90.0])
        self.assertEqual(list(trace), samples)

    def test_vmtrace_empty_samples(self):
        """VMTrace handles empty sample sequence gracefully."""
        trace = VMTrace(
            vm_id="vm_empty",
            cpu_samples=[],
            source_file="/test/path/empty",
        )
        self.assertEqual(trace.total_samples, 0)
        self.assertEqual(len(trace), 0)
        self.assertEqual(trace.min_cpu, 0.0)
        self.assertEqual(trace.max_cpu, 0.0)
        self.assertEqual(trace.mean_cpu, 0.0)

    def test_dataset_load_report_properties(self):
        """DatasetLoadReport correctly tracks loaded and skipped counts."""
        trace1 = VMTrace(vm_id="t1", cpu_samples=[20.0], source_file="f1")
        trace2 = VMTrace(vm_id="t2", cpu_samples=[40.0], source_file="f2")

        report = DatasetLoadReport(
            valid_traces=[trace1, trace2],
            skipped_files={"/path/corrupt": "Non-numeric token", "/path/empty": "No samples"},
        )

        self.assertEqual(report.total_loaded, 2)
        self.assertEqual(report.total_skipped, 2)
        self.assertEqual(len(report.valid_traces), 2)
        self.assertIn("/path/corrupt", report.skipped_files)


if __name__ == "__main__":
    unittest.main()
