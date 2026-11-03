"""Unit tests for main CLI configuration and dataset execution checks."""

from pathlib import Path
import tempfile
import unittest

from main import parse_args, main


class TestMain(unittest.TestCase):
    def test_parse_args_defaults(self):
        """parse_args provides expected default values."""
        args = parse_args([])
        self.assertIsNone(args.dataset_dir)
        self.assertEqual(args.max_vms, 50)
        self.assertEqual(args.steps, 100)
        self.assertEqual(args.racks, 2)
        self.assertEqual(args.hosts_per_rack, 4)
        self.assertEqual(args.cpu_per_host, 16)
        self.assertEqual(args.ram_per_host, 64)
        self.assertEqual(args.ambient_temp, 25.0)
        self.assertEqual(args.hotspot_threshold, 75.0)
        self.assertEqual(args.seed, 42)

    def test_parse_args_custom_values(self):
        """parse_args correctly parses custom CLI arguments."""
        custom_args = [
            "--dataset-dir", "custom/path",
            "--max-vms", "25",
            "--steps", "50",
            "--racks", "4",
            "--hosts-per-rack", "8",
            "--cpu-per-host", "32",
            "--ram-per-host", "128",
            "--ambient-temp", "20.0",
            "--hotspot-threshold", "80.0",
            "--seed", "99",
        ]
        args = parse_args(custom_args)
        self.assertEqual(args.dataset_dir, "custom/path")
        self.assertEqual(args.max_vms, 25)
        self.assertEqual(args.steps, 50)
        self.assertEqual(args.racks, 4)
        self.assertEqual(args.hosts_per_rack, 8)
        self.assertEqual(args.cpu_per_host, 32)
        self.assertEqual(args.ram_per_host, 128)
        self.assertEqual(args.ambient_temp, 20.0)
        self.assertEqual(args.hotspot_threshold, 80.0)
        self.assertEqual(args.seed, 99)

    def test_main_no_dataset_configured_exits(self):
        """main exits cleanly with code 1 if no dataset is configured."""
        with self.assertRaises(SystemExit) as ctx:
            main([])
        self.assertEqual(ctx.exception.code, 1)

    def test_main_missing_dataset_directory_exits(self):
        """main exits cleanly with code 1 if dataset directory does not exist."""
        with self.assertRaises(SystemExit) as ctx:
            main(["--dataset-dir", "non_existent_dir_98765"])
        self.assertEqual(ctx.exception.code, 1)

    def test_main_dataset_loader_pending_exits(self):
        """main exits cleanly with code 1 if dataset loader is not configured for directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(SystemExit) as ctx:
                main(["--dataset-dir", tmp_dir])
            self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
