"""Unit tests for workload generation and VM lifecycle."""

import unittest
from src.workload import VM, WorkloadGenerator


class TestWorkload(unittest.TestCase):
    def test_vm_lifecycle(self):
        vm = VM(
            vm_id=1,
            cpu_demand=4,
            ram_demand=8,
            arrival_time=0,
            duration=3,
        )
        self.assertEqual(vm.remaining_duration, 3)
        self.assertFalse(vm.is_finished())

        self.assertTrue(vm.tick())  # remaining: 2
        self.assertEqual(vm.remaining_duration, 2)
        self.assertTrue(vm.tick())  # remaining: 1
        self.assertFalse(vm.tick())  # remaining: 0, returns False
        self.assertTrue(vm.is_finished())

    def test_workload_reproducibility_with_seed(self):
        gen1 = WorkloadGenerator(seed=123, total_time_steps=20)
        gen2 = WorkloadGenerator(seed=123, total_time_steps=20)

        vms1 = gen1.generate_workload()
        vms2 = gen2.generate_workload()

        self.assertEqual(len(vms1), len(vms2))
        for v1, v2 in zip(vms1, vms2):
            self.assertEqual(v1.vm_id, v2.vm_id)
            self.assertEqual(v1.cpu_demand, v2.cpu_demand)
            self.assertEqual(v1.ram_demand, v2.ram_demand)
            self.assertEqual(v1.arrival_time, v2.arrival_time)
            self.assertEqual(v1.duration, v2.duration)

    def test_different_seeds_produce_different_workloads(self):
        gen1 = WorkloadGenerator(seed=101, total_time_steps=50)
        gen2 = WorkloadGenerator(seed=202, total_time_steps=50)

        vms1 = gen1.generate_workload()
        vms2 = gen2.generate_workload()

        # Highly unlikely to be identical with different seeds
        vms1_attrs = [(v.cpu_demand, v.ram_demand, v.duration) for v in vms1]
        vms2_attrs = [(v.cpu_demand, v.ram_demand, v.duration) for v in vms2]
        self.assertNotEqual(vms1_attrs, vms2_attrs)

    def test_schedule_by_step_structure(self):
        gen = WorkloadGenerator(seed=42, total_time_steps=10)
        schedule = gen.generate_schedule_by_step()

        self.assertEqual(len(schedule), 10)
        for t in range(10):
            self.assertIn(t, schedule)
            for vm in schedule[t]:
                self.assertEqual(vm.arrival_time, t)


if __name__ == "__main__":
    unittest.main()
