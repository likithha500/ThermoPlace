"""Unit tests for First Fit placement strategy."""

import unittest
from src.datacenter import DataCenter
from src.placement.first_fit import FirstFitPlacement
from src.workload import VM


class TestFirstFit(unittest.TestCase):
    def setUp(self):
        self.dc = DataCenter.create_standard_datacenter(
            num_racks=1,
            hosts_per_rack=3,
            cpu_per_host=8,
            ram_per_host=16,
        )
        self.strategy = FirstFitPlacement()

    def test_first_fit_sequential_packing(self):
        # VM 1 should go to Host 0
        vm1 = VM(vm_id=1, cpu_demand=4, ram_demand=8, arrival_time=0, duration=10)
        h1 = self.strategy.select_host(vm1, self.dc)
        self.assertIsNotNone(h1)
        self.assertEqual(h1.host_id, "R0_H0")
        h1.allocate(vm1)

        # VM 2 still fits on Host 0 (remaining: 4 CPU, 8 RAM)
        vm2 = VM(vm_id=2, cpu_demand=4, ram_demand=8, arrival_time=0, duration=10)
        h2 = self.strategy.select_host(vm2, self.dc)
        self.assertIsNotNone(h2)
        self.assertEqual(h2.host_id, "R0_H0")
        h2.allocate(vm2)

        # Host 0 is now full. VM 3 should spill over to Host 1 ("R0_H1")
        vm3 = VM(vm_id=3, cpu_demand=4, ram_demand=8, arrival_time=0, duration=10)
        h3 = self.strategy.select_host(vm3, self.dc)
        self.assertIsNotNone(h3)
        self.assertEqual(h3.host_id, "R0_H1")
        h3.allocate(vm3)

    def test_first_fit_rejection_when_full(self):
        # Fill all 3 hosts
        for idx, host in enumerate(self.dc.hosts):
            vm = VM(vm_id=idx + 10, cpu_demand=8, ram_demand=16, arrival_time=0, duration=10)
            host.allocate(vm)

        # Arriving VM cannot be placed
        vm_overflow = VM(vm_id=99, cpu_demand=2, ram_demand=4, arrival_time=0, duration=10)
        selected = self.strategy.select_host(vm_overflow, self.dc)
        self.assertIsNone(selected)


if __name__ == "__main__":
    unittest.main()
