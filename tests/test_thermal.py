"""Unit tests for preliminary analytical thermal model."""

import unittest
from src.datacenter import Host, Rack, DataCenter
from src.thermal import ThermalModel, ThermalConfig
from src.workload import VM


class TestThermalModel(unittest.TestCase):
    def setUp(self):
        self.config = ThermalConfig(
            ambient_temperature_c=25.0,
            thermal_resistance_c_per_w=0.13,
            thermal_time_constant_tau=5.0,
            recirculation_alpha=0.002,
            slot_height_beta=0.5,
            hotspot_threshold_c=75.0,
        )
        self.model = ThermalModel(config=self.config)

    def test_host_power_scaling(self):
        host = Host(
            host_id="H0",
            rack_id=0,
            rack_slot=0,
            total_cpu=16,
            total_ram=64,
            idle_power_w=150.0,
            max_power_w=350.0,
        )
        # Idle (0% utilization)
        power_idle = self.model.estimate_host_power_w(host)
        self.assertAlmostEqual(power_idle, 150.0)

        # 50% utilization (8 cores allocated)
        vm_half = VM(vm_id=1, cpu_demand=8, ram_demand=32, arrival_time=0, duration=5)
        host.allocate(vm_half)
        power_half = self.model.estimate_host_power_w(host)
        self.assertAlmostEqual(power_half, 250.0)

        # 100% utilization (additional 8 cores)
        vm_full = VM(vm_id=2, cpu_demand=8, ram_demand=32, arrival_time=0, duration=5)
        host.allocate(vm_full)
        power_full = self.model.estimate_host_power_w(host)
        self.assertAlmostEqual(power_full, 350.0)

    def test_inlet_temperature_stratification(self):
        rack = Rack(rack_id=0)
        bottom_host = Host(host_id="H_bottom", rack_id=0, rack_slot=0, idle_power_w=150.0)
        top_host = Host(host_id="H_top", rack_id=0, rack_slot=3, idle_power_w=150.0)
        rack.add_host(bottom_host)
        rack.add_host(top_host)

        inlet_bottom = self.model.estimate_inlet_temperature_c(bottom_host, rack)
        inlet_top = self.model.estimate_inlet_temperature_c(top_host, rack)

        # Upper host should experience warmer inlet air due to slot_height_beta
        self.assertGreater(inlet_top, inlet_bottom)
        self.assertAlmostEqual(inlet_top - inlet_bottom, 3 * self.config.slot_height_beta)

    def test_transient_thermal_inertia(self):
        # Server starts at 25°C and is suddenly placed under 100% load
        host = Host(
            host_id="H0",
            rack_id=0,
            rack_slot=0,
            total_cpu=16,
            total_ram=64,
            idle_power_w=150.0,
            max_power_w=350.0,
            initial_temperature_c=25.0,
        )
        rack = Rack(rack_id=0, hosts=[host])
        vm = VM(vm_id=1, cpu_demand=16, ram_demand=64, arrival_time=0, duration=50)
        host.allocate(vm)

        t_target = self.model.estimate_target_temperature_c(host, rack)
        self.assertGreater(t_target, 60.0)

        # Step 1: temperature should rise smoothly, not jump instantaneously to t_target
        t_step1 = self.model.step_host_temperature(host, rack, dt=1.0)
        self.assertGreater(t_step1, 25.0)
        self.assertLess(t_step1, t_target)

        # Multiple steps: temperature monotonically approaches target
        prev_t = t_step1
        for _ in range(15):
            curr_t = self.model.step_host_temperature(host, rack, dt=1.0)
            self.assertGreaterEqual(curr_t, prev_t)
            prev_t = curr_t

        self.assertAlmostEqual(prev_t, t_target, delta=2.0)

    def test_hotspot_threshold_detection(self):
        host = Host(
            host_id="H0",
            rack_id=0,
            rack_slot=0,
            initial_temperature_c=74.9,
        )
        self.assertFalse(self.model.is_hotspot(host))

        host.simulated_temperature_c = 75.0
        self.assertTrue(self.model.is_hotspot(host))

        host.simulated_temperature_c = 82.5
        self.assertTrue(self.model.is_hotspot(host))


if __name__ == "__main__":
    unittest.main()
