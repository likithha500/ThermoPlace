"""Unit tests for Host, Rack, and DataCenter capacity and state management."""

import unittest
from src.datacenter import Host, Rack, DataCenter
from src.workload import VM


class TestDataCenter(unittest.TestCase):
    def setUp(self):
        self.host = Host(
            host_id="R0_H0",
            rack_id=0,
            rack_slot=0,
            total_cpu=16,
            total_ram=64,
            idle_power_w=150.0,
            max_power_w=350.0,
            initial_temperature_c=25.0,
        )

    def test_host_capacity_enforcement(self):
        vm1 = VM(vm_id=1, cpu_demand=8, ram_demand=32, arrival_time=0, duration=10)
        vm2 = VM(vm_id=2, cpu_demand=8, ram_demand=32, arrival_time=0, duration=10)
        vm_oversubscribed = VM(vm_id=3, cpu_demand=4, ram_demand=8, arrival_time=0, duration=10)

        # Allocate first VM
        self.assertTrue(self.host.can_accommodate(vm1))
        self.assertTrue(self.host.allocate(vm1))
        self.assertEqual(self.host.used_cpu, 8)
        self.assertEqual(self.host.used_ram, 32)
        self.assertAlmostEqual(self.host.cpu_utilization, 0.5)
        self.assertAlmostEqual(self.host.ram_utilization, 0.5)

        # Allocate second VM (fills capacity exactly)
        self.assertTrue(self.host.can_accommodate(vm2))
        self.assertTrue(self.host.allocate(vm2))
        self.assertEqual(self.host.used_cpu, 16)
        self.assertEqual(self.host.used_ram, 64)
        self.assertAlmostEqual(self.host.cpu_utilization, 1.0)
        self.assertAlmostEqual(self.host.ram_utilization, 1.0)

        # Attempt to over-allocate
        self.assertFalse(self.host.can_accommodate(vm_oversubscribed))
        self.assertFalse(self.host.allocate(vm_oversubscribed))
        # Ensure state remained unchanged
        self.assertEqual(self.host.used_cpu, 16)
        self.assertEqual(self.host.used_ram, 64)

    def test_host_vm_release(self):
        vm = VM(vm_id=10, cpu_demand=4, ram_demand=16, arrival_time=0, duration=5)
        self.host.allocate(vm)
        self.assertEqual(self.host.used_cpu, 4)
        self.assertEqual(self.host.used_ram, 16)

        released = self.host.release(10)
        self.assertEqual(released, vm)
        self.assertEqual(self.host.used_cpu, 0)
        self.assertEqual(self.host.used_ram, 0)
        self.assertEqual(len(self.host.allocated_vms), 0)

        # Releasing non-existent VM returns None safely
        self.assertIsNone(self.host.release(999))

    def test_host_reset(self):
        vm = VM(vm_id=1, cpu_demand=8, ram_demand=16, arrival_time=0, duration=5)
        self.host.allocate(vm)
        self.host.simulated_temperature_c = 68.5

        self.host.reset(initial_temp=25.0)
        self.assertEqual(self.host.used_cpu, 0)
        self.assertEqual(self.host.used_ram, 0)
        self.assertEqual(len(self.host.allocated_vms), 0)
        self.assertEqual(self.host.simulated_temperature_c, 25.0)

    def test_datacenter_creation_and_reset(self):
        dc = DataCenter.create_standard_datacenter(
            num_racks=2,
            hosts_per_rack=3,
            cpu_per_host=16,
            ram_per_host=64,
            initial_temperature_c=22.0,
        )
        self.assertEqual(len(dc.racks), 2)
        self.assertEqual(len(dc.hosts), 6)

        # Allocate VM to first host
        vm = VM(vm_id=100, cpu_demand=8, ram_demand=32, arrival_time=0, duration=10)
        first_host = dc.hosts[0]
        first_host.allocate(vm)
        first_host.simulated_temperature_c = 55.0

        self.assertEqual(first_host.used_cpu, 8)

        # Global reset
        dc.reset(initial_temp=22.0)
        for h in dc.hosts:
            self.assertEqual(h.used_cpu, 0)
            self.assertEqual(h.used_ram, 0)
            self.assertEqual(len(h.allocated_vms), 0)
            self.assertEqual(h.simulated_temperature_c, 22.0)

    def test_host_actual_used_cpu_with_planetlab_traces(self):
        """Host computes actual_used_cpu dynamically from VM traces while enforcing reserved capacity."""
        # VM 1: Reserved 8 cores, currently utilizing 50% = 4.0 cores
        vm1 = VM(
            vm_id="vm_planetlab_alpha",
            cpu_demand=8,
            ram_demand=16,
            arrival_time=0,
            cpu_trace=[50.0, 75.0],
        )
        # VM 2: Reserved 4 cores, currently utilizing 25% = 1.0 core
        vm2 = VM(
            vm_id="vm_planetlab_beta",
            cpu_demand=4,
            ram_demand=8,
            arrival_time=0,
            cpu_trace=[25.0, 50.0],
        )

        self.assertTrue(self.host.allocate(vm1))
        self.assertTrue(self.host.allocate(vm2))

        # Reserved capacity allocated: 8 + 4 = 12 cores
        self.assertEqual(self.host.used_cpu, 12)
        # Dynamic actual CPU utilized: (8 * 0.50) + (4 * 0.25) = 4.0 + 1.0 = 5.0 cores
        self.assertAlmostEqual(self.host.actual_used_cpu, 5.0)
        self.assertAlmostEqual(self.host.actual_cpu_utilization, 5.0 / 16.0)

        # Advance one step
        vm1.tick()
        vm2.tick()
        # Reserved capacity remains 12 cores
        self.assertEqual(self.host.used_cpu, 12)
        # Dynamic actual CPU utilized: (8 * 0.75) + (4 * 0.50) = 6.0 + 2.0 = 8.0 cores
        self.assertAlmostEqual(self.host.actual_used_cpu, 8.0)

        # Release string-identified VM
        released = self.host.release("vm_planetlab_alpha")
        self.assertEqual(released, vm1)
        self.assertEqual(self.host.used_cpu, 4)
        self.assertAlmostEqual(self.host.actual_used_cpu, 2.0)


if __name__ == "__main__":
    unittest.main()
