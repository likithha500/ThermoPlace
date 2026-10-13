"""Unit tests for workload generation, VM lifecycle, and PlanetLab trace integration."""

import unittest
from src.workload import VM, WorkloadGenerator, PlanetLabWorkloadGenerator
from src.dataset_loader import VMTrace


class TestWorkload(unittest.TestCase):
    # -------------------------------------------------------------------------
    # 1. Existing VM Lifecycle & Synthetic Workload Tests
    # -------------------------------------------------------------------------

    def test_vm_lifecycle(self):
        """Preserves original lifecycle behavior for synthetic VMs without traces."""
        vm = VM(
            vm_id=1,
            cpu_demand=4,
            ram_demand=8,
            arrival_time=0,
            duration=3,
        )
        self.assertEqual(vm.remaining_duration, 3)
        self.assertFalse(vm.is_finished())
        self.assertFalse(vm.has_trace)
        self.assertIsNone(vm.current_cpu_utilization)
        # Without trace, actual utilized CPU equals reserved demand
        self.assertEqual(vm.current_actual_cpu, 4.0)

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

    # -------------------------------------------------------------------------
    # 2. PlanetLab Trace Integration Tests: Sample Progression
    # -------------------------------------------------------------------------

    def test_vm_with_trace_sample_progression(self):
        """Advances through CPU trace samples in chronological order without alteration."""
        samples = [12.0, 48.5, 80.0, 24.0]
        vm = VM(
            vm_id="vm_trace_01",
            cpu_demand=4,
            ram_demand=8,
            arrival_time=0,
            cpu_trace=samples,
        )

        # Inferred duration from trace length
        self.assertEqual(vm.duration, 4)
        self.assertEqual(vm.remaining_duration, 4)
        self.assertTrue(vm.has_trace)
        self.assertEqual(vm.total_trace_samples, 4)

        # Step 0: Initial sample
        self.assertEqual(vm.current_sample_index, 0)
        self.assertEqual(vm.current_cpu_utilization, 12.0)
        self.assertEqual(vm.remaining_trace_samples, 4)

        # Step 1: Advance sample index
        self.assertTrue(vm.tick())
        self.assertEqual(vm.current_sample_index, 1)
        self.assertEqual(vm.current_cpu_utilization, 48.5)
        self.assertEqual(vm.remaining_trace_samples, 3)

        # Step 2
        self.assertTrue(vm.tick())
        self.assertEqual(vm.current_sample_index, 2)
        self.assertEqual(vm.current_cpu_utilization, 80.0)
        self.assertEqual(vm.remaining_trace_samples, 2)

        # Step 3
        self.assertTrue(vm.tick())
        self.assertEqual(vm.current_sample_index, 3)
        self.assertEqual(vm.current_cpu_utilization, 24.0)
        self.assertEqual(vm.remaining_trace_samples, 1)

    # -------------------------------------------------------------------------
    # 3. Trace Completion & Non-Fabrication Tests
    # -------------------------------------------------------------------------

    def test_vm_trace_completion(self):
        """When all samples are consumed, marks trace finished and does not fabricate data."""
        samples = [25.0, 75.0]
        vm = VM(
            vm_id="vm_short",
            cpu_demand=2,
            ram_demand=4,
            arrival_time=0,
            cpu_trace=samples,
        )

        # Sample 0
        self.assertEqual(vm.current_cpu_utilization, 25.0)
        self.assertFalse(vm.is_trace_finished)
        self.assertFalse(vm.is_finished())

        # Sample 1
        self.assertTrue(vm.tick())
        self.assertEqual(vm.current_cpu_utilization, 75.0)
        self.assertFalse(vm.is_trace_finished)
        self.assertFalse(vm.is_finished())

        # Trace exhausted
        self.assertFalse(vm.tick())
        self.assertEqual(vm.current_sample_index, 2)
        self.assertTrue(vm.is_trace_finished)
        self.assertTrue(vm.is_finished())

        # Crucial: Does NOT invent, interpolate, or fabricate samples beyond the end
        self.assertIsNone(vm.current_cpu_utilization)
        self.assertEqual(vm.remaining_trace_samples, 0)

        # Subsequent tick calls continue to report finished
        self.assertFalse(vm.tick())

    # -------------------------------------------------------------------------
    # 4. Separation of Reserved CPU Capacity vs. Actual Trace Utilization
    # -------------------------------------------------------------------------

    def test_reserved_cpu_demand_vs_actual_cpu_utilization(self):
        """Preserves reserved cpu_demand for placement while tracking actual utilized cores."""
        samples = [50.0, 25.0, 100.0, 0.0]
        vm = VM(
            vm_id="vm_split",
            cpu_demand=8,  # Reserved capacity: 8 cores
            ram_demand=16,
            arrival_time=0,
            cpu_trace=samples,
        )

        # Placement reservation remains strictly 8 cores throughout
        self.assertEqual(vm.cpu_demand, 8)
        # Sample 0 (50% of 8 cores = 4.0 cores)
        self.assertEqual(vm.current_actual_cpu, 4.0)

        vm.tick()
        self.assertEqual(vm.cpu_demand, 8)
        # Sample 1 (25% of 8 cores = 2.0 cores)
        self.assertEqual(vm.current_actual_cpu, 2.0)

        vm.tick()
        self.assertEqual(vm.cpu_demand, 8)
        # Sample 2 (100% of 8 cores = 8.0 cores)
        self.assertEqual(vm.current_actual_cpu, 8.0)

        vm.tick()
        self.assertEqual(vm.cpu_demand, 8)
        # Sample 3 (0% of 8 cores = 0.0 cores)
        self.assertEqual(vm.current_actual_cpu, 0.0)

    # -------------------------------------------------------------------------
    # 5. VM.from_trace Factory & VMTrace Integration
    # -------------------------------------------------------------------------

    def test_vm_from_trace_factory_with_vmtrace_object(self):
        """Instantiates VM directly from VMTrace instance preserving metadata."""
        trace = VMTrace(
            vm_id="host_146-179_slice",
            cpu_samples=[10.0, 20.0, 30.0],
            source_file="/fake/data/trace",
        )

        vm = VM.from_trace(
            trace=trace,
            cpu_demand=4,
            ram_demand=8,
            arrival_time=5,
        )

        self.assertEqual(vm.vm_id, "host_146-179_slice")
        self.assertEqual(vm.trace_id, "host_146-179_slice")
        self.assertEqual(vm.cpu_demand, 4)
        self.assertEqual(vm.ram_demand, 8)
        self.assertEqual(vm.arrival_time, 5)
        self.assertEqual(vm.duration, 3)
        self.assertEqual(vm.remaining_duration, 3)
        self.assertEqual(vm.cpu_trace, [10.0, 20.0, 30.0])
        self.assertEqual(vm.current_cpu_utilization, 10.0)

    # -------------------------------------------------------------------------
    # 6. RAM Provisioning & Documentation
    # -------------------------------------------------------------------------

    def test_ram_demand_provisioning_without_fabricated_measurements(self):
        """RAM demand is provisioned for placement; no dynamic RAM time-series is invented."""
        vm = VM.from_trace(
            trace=[30.0, 60.0],
            ram_demand=16,
        )
        self.assertEqual(vm.ram_demand, 16)
        # Verify VM does not contain or generate fabricated dynamic RAM time series
        self.assertFalse(hasattr(vm, "ram_trace"))

    # -------------------------------------------------------------------------
    # 7. State Reset for Experiment Reproducibility
    # -------------------------------------------------------------------------

    def test_vm_reset_restores_initial_state(self):
        """reset() restores initial remaining_duration and current_sample_index."""
        vm = VM(
            vm_id="vm_resettable",
            cpu_demand=2,
            ram_demand=4,
            arrival_time=0,
            cpu_trace=[10.0, 20.0, 30.0],
        )
        vm.tick()
        vm.tick()
        self.assertEqual(vm.current_sample_index, 2)
        self.assertEqual(vm.remaining_duration, 1)

        vm.reset()
        self.assertEqual(vm.current_sample_index, 0)
        self.assertEqual(vm.remaining_duration, 3)
        self.assertEqual(vm.current_cpu_utilization, 10.0)
        self.assertFalse(vm.is_finished())

    # -------------------------------------------------------------------------
    # 8. PlanetLabWorkloadGenerator Tests
    # -------------------------------------------------------------------------

    def test_planetlab_workload_generator_schedule(self):
        """Generates reproducible simulation schedule from real PlanetLab traces."""
        traces = [
            VMTrace(vm_id=f"trace_{i}", cpu_samples=[10.0 + i] * 5, source_file="f")
            for i in range(10)
        ]

        gen = PlanetLabWorkloadGenerator(
            traces=traces,
            cpu_options=[2, 4],
            ram_options=[4, 8],
            arrival_rate_lambda=2.0,
            max_arrival_time=5,
            seed=42,
        )

        schedule = gen.generate_schedule_by_step(total_time_steps=6)
        total_scheduled = sum(len(vms) for vms in schedule.values())
        self.assertEqual(total_scheduled, 10)

        for t, vms in schedule.items():
            for vm in vms:
                self.assertEqual(vm.arrival_time, t)
                self.assertTrue(vm.has_trace)
                self.assertIn(vm.cpu_demand, [2, 4])
                self.assertIn(vm.ram_demand, [4, 8])


if __name__ == "__main__":
    unittest.main()
