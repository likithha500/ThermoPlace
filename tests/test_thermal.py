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

    def test_trace_driven_power_and_temperature_scaling(self):
        """Dynamic trace samples drive proportional changes in host utilization, power, and target temp."""
        host = Host(
            host_id="H_dynamic",
            rack_id=0,
            rack_slot=0,
            total_cpu=16,
            total_ram=64,
            idle_power_w=150.0,
            max_power_w=350.0,
            initial_temperature_c=25.0,
        )
        rack = Rack(rack_id=0, hosts=[host])

        # VM with 16 reserved cores, dynamic trace: 10% utilization then 90% utilization
        vm = VM(
            vm_id="vm_burst",
            cpu_demand=16,
            ram_demand=32,
            arrival_time=0,
            cpu_trace=[10.0, 90.0],
        )
        host.allocate(vm)

        # Timestep 0: Trace sample is 10%
        self.assertAlmostEqual(host.actual_used_cpu, 1.6)
        self.assertAlmostEqual(host.actual_cpu_utilization, 0.10)
        power_low = self.model.estimate_host_power_w(host)
        # 150 + 200 * 0.10 = 170 W
        self.assertAlmostEqual(power_low, 170.0)
        target_low = self.model.estimate_target_temperature_c(host, rack)

        # Step host temperature at low power
        t_after_low = self.model.step_host_temperature(host, rack, dt=1.0)
        self.assertGreater(t_after_low, 25.0)

        # Advance trace to Timestep 1: Trace sample is 90%
        vm.tick()
        self.assertAlmostEqual(host.actual_used_cpu, 14.4)
        self.assertAlmostEqual(host.actual_cpu_utilization, 0.90)
        power_high = self.model.estimate_host_power_w(host)
        # 150 + 200 * 0.90 = 330 W
        self.assertAlmostEqual(power_high, 330.0)
        target_high = self.model.estimate_target_temperature_c(host, rack)

        # High utilization must yield substantially higher power and target temp
        self.assertGreater(power_high, power_low)
        self.assertGreater(target_high, target_low)

        # Step temperature at high power: temperature rises toward target_high
        t_after_high = self.model.step_host_temperature(host, rack, dt=1.0)
        self.assertGreater(t_after_high, t_after_low)

    def test_multiple_vms_utilization_combination_and_capacity_capping(self):
        """Host aggregates multiple active VMs' actual CPU usage and prevents overflow beyond capacity."""
        host = Host(
            host_id="H_multi",
            rack_id=0,
            rack_slot=0,
            total_cpu=16,
            total_ram=64,
            idle_power_w=150.0,
            max_power_w=350.0,
        )
        # VM 1: 8 cores, 50% utilization -> 4 cores
        vm1 = VM(vm_id=1, cpu_demand=8, ram_demand=16, arrival_time=0, cpu_trace=[50.0])
        # VM 2: 8 cores, 75% utilization -> 6 cores
        vm2 = VM(vm_id=2, cpu_demand=8, ram_demand=16, arrival_time=0, cpu_trace=[75.0])

        host.allocate(vm1)
        host.allocate(vm2)

        # Total reserved = 16 cores (100% capacity)
        self.assertEqual(host.used_cpu, 16)
        self.assertAlmostEqual(host.cpu_utilization, 1.0)

        # Aggregate actual = 4 + 6 = 10 cores
        self.assertAlmostEqual(host.actual_used_cpu, 10.0)
        # Actual ratio = 10 / 16 = 0.625
        self.assertAlmostEqual(host.actual_cpu_utilization, 0.625)

        # Power = 150 + 200 * 0.625 = 275 W
        power = self.model.estimate_host_power_w(host)
        self.assertAlmostEqual(power, 275.0)

        # Ensure safety clamp: actual_cpu_utilization is guaranteed <= 1.0
        self.assertLessEqual(host.actual_cpu_utilization, 1.0)


if __name__ == "__main__":
    unittest.main()
