"""Unit tests for telemetry and metric collection."""

import unittest
from src.datacenter import DataCenter
from src.thermal import ThermalModel, ThermalConfig
from src.metrics import MetricsCollector


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.dc = DataCenter.create_standard_datacenter(
            num_racks=1,
            hosts_per_rack=3,
            initial_temperature_c=25.0,
        )
        self.thermal_config = ThermalConfig(hotspot_threshold_c=75.0)
        self.thermal_model = ThermalModel(config=self.thermal_config)
        self.collector = MetricsCollector(
            algorithm_name="TestAlgorithm",
            thermal_model=self.thermal_model,
        )

    def test_metrics_step_recording(self):
        # Set artificial simulated temperatures
        self.dc.hosts[0].simulated_temperature_c = 40.0
        self.dc.hosts[1].simulated_temperature_c = 60.0
        self.dc.hosts[2].simulated_temperature_c = 80.0  # Hotspot!

        snapshot = self.collector.record_step(
            time_step=0,
            datacenter=self.dc,
            vms_arrived_this_step=3,
            vms_placed_this_step=2,
            vms_rejected_this_step=1,
        )

        self.assertEqual(snapshot.time_step, 0)
        self.assertEqual(snapshot.num_vms_arrived, 3)
        self.assertEqual(snapshot.num_vms_placed, 2)
        self.assertEqual(snapshot.num_vms_rejected, 1)
        self.assertEqual(snapshot.max_temperature_c, 80.0)
        self.assertEqual(snapshot.min_temperature_c, 40.0)
        self.assertEqual(snapshot.hotspot_count, 1)
        self.assertAlmostEqual(snapshot.thermal_imbalance_range, 40.0)
        self.assertGreater(snapshot.thermal_imbalance_std, 0.0)

    def test_metrics_dataframe_and_summary(self):
        self.dc.hosts[0].simulated_temperature_c = 50.0
        self.dc.hosts[1].simulated_temperature_c = 55.0
        self.dc.hosts[2].simulated_temperature_c = 60.0

        self.collector.record_step(0, self.dc, 2, 2, 0)
        self.collector.record_step(1, self.dc, 2, 1, 1)

        df = self.collector.to_dataframe()
        self.assertEqual(len(df), 2)
        self.assertIn("max_temp_c", df.columns)
        self.assertIn("thermal_imbalance_std", df.columns)

        summary = self.collector.summary()
        self.assertEqual(summary["total_steps"], 2)
        self.assertEqual(summary["total_arrived_vms"], 4)
        self.assertEqual(summary["total_placed_vms"], 3)
        self.assertEqual(summary["total_rejected_vms"], 1)
        self.assertAlmostEqual(summary["acceptance_rate"], 0.75)


if __name__ == "__main__":
    unittest.main()
