"""Unit tests for the discrete-time simulator and reset reproducibility."""

import unittest
from src.datacenter import DataCenter
from src.workload import VM, WorkloadGenerator
from src.thermal import ThermalModel, ThermalConfig
from src.placement.first_fit import FirstFitPlacement
from src.simulator import Simulator


class TestSimulator(unittest.TestCase):
    def test_end_to_end_simulation_and_reset_reproducibility(self):
        # 1. Setup deterministic workload
        gen = WorkloadGenerator(seed=777, total_time_steps=20, arrival_rate_lambda=2.0)
        schedule = gen.generate_schedule_by_step()

        # 2. Setup datacenter and simulator
        dc = DataCenter.create_standard_datacenter(
            num_racks=2,
            hosts_per_rack=2,
            cpu_per_host=16,
            ram_per_host=64,
            initial_temperature_c=25.0,
        )
        sim = Simulator(datacenter=dc)
        strategy = FirstFitPlacement()

        # Run 1
        metrics_run1 = sim.run(workload_schedule=schedule, strategy=strategy, total_steps=20)
        summary1 = metrics_run1.summary()

        # Reset datacenter and simulator
        sim.reset(initial_temp=25.0)

        # Regnerate fresh VM objects with identical schedule for Run 2
        # (VM objects mutate remaining_duration during simulation, so generate_schedule_by_step gives fresh instances)
        gen2 = WorkloadGenerator(seed=777, total_time_steps=20, arrival_rate_lambda=2.0)
        schedule2 = gen2.generate_schedule_by_step()

        metrics_run2 = sim.run(workload_schedule=schedule2, strategy=strategy, total_steps=20)
        summary2 = metrics_run2.summary()

        # Ensure Run 1 and Run 2 are exactly identical across all metrics
        self.assertEqual(summary1["total_arrived_vms"], summary2["total_arrived_vms"])
        self.assertEqual(summary1["total_placed_vms"], summary2["total_placed_vms"])
        self.assertEqual(summary1["total_rejected_vms"], summary2["total_rejected_vms"])
        self.assertAlmostEqual(summary1["peak_temperature_c"], summary2["peak_temperature_c"], places=5)
        self.assertAlmostEqual(
            summary1["average_thermal_imbalance_std"],
            summary2["average_thermal_imbalance_std"],
            places=5,
        )
        self.assertEqual(
            summary1["cumulative_hotspot_events"],
            summary2["cumulative_hotspot_events"],
        )

    def test_capacity_constraints_never_violated(self):
        # Intense workload on small datacenter to force saturation
        gen = WorkloadGenerator(seed=42, total_time_steps=15, arrival_rate_lambda=5.0)
        schedule = gen.generate_schedule_by_step()

        dc = DataCenter.create_standard_datacenter(
            num_racks=1,
            hosts_per_rack=2,
            cpu_per_host=8,
            ram_per_host=16,
        )
        sim = Simulator(datacenter=dc)
        sim.run(workload_schedule=schedule, strategy=FirstFitPlacement(), total_steps=15)

        # Verify that no host ever exceeded CPU or RAM at end of simulation
        for host in dc.hosts:
            self.assertLessEqual(host.used_cpu, host.total_cpu)
            self.assertLessEqual(host.used_ram, host.total_ram)

    def test_simulation_with_planetlab_trace_workload(self):
        """Simulator drives active VMs through trace samples and updates thermal state dynamically."""
        dc = DataCenter.create_standard_datacenter(
            num_racks=1,
            hosts_per_rack=2,
            cpu_per_host=16,
            ram_per_host=32,
            initial_temperature_c=25.0,
        )
        sim = Simulator(datacenter=dc)

        # VM 1: Arrives at t=0, 4 cores, trace=[10%, 80%, 20%]
        vm1 = VM(
            vm_id="vm_pl_1",
            cpu_demand=4,
            ram_demand=8,
            arrival_time=0,
            cpu_trace=[10.0, 80.0, 20.0],
        )
        # VM 2: Arrives at t=1, 8 cores, trace=[50%, 50%]
        vm2 = VM(
            vm_id="vm_pl_2",
            cpu_demand=8,
            ram_demand=16,
            arrival_time=1,
            cpu_trace=[50.0, 50.0],
        )

        schedule = {0: [vm1], 1: [vm2]}
        metrics = sim.run(workload_schedule=schedule, strategy=FirstFitPlacement(), total_steps=3)

        self.assertEqual(len(metrics.snapshots), 3)

        # Step 0: Only vm1 is placed, sample 0 (10% of 4 = 0.4 cores)
        snap0 = metrics.snapshots[0]
        self.assertEqual(snap0.active_vm_count, 1)
        host0 = dc.hosts[0]
        self.assertIn("vm_pl_1", host0.allocated_vms)

        # Step 1: vm1 ticked to sample 1 (80% of 4 = 3.2 cores), vm2 arrived with sample 0 (50% of 8 = 4.0 cores)
        # Both on host0: actual used = 3.2 + 4.0 = 7.2 cores out of 16
        snap1 = metrics.snapshots[1]
        self.assertEqual(snap1.active_vm_count, 2)
        self.assertGreater(snap1.total_power_w, snap0.total_power_w)

        # Step 2: vm1 ticked to sample 2 (20% of 4 = 0.8 cores), vm2 ticked to sample 1 (50% of 8 = 4.0 cores)
        # Power decreases as vm1 usage drops from 80% to 20%
        snap2 = metrics.snapshots[2]
        self.assertEqual(snap2.active_vm_count, 2)
        self.assertLess(snap2.total_power_w, snap1.total_power_w)

    def test_trace_completion_departure_reclaims_capacity(self):
        """When a VM finishes its trace, simulator departures reclaim host capacity."""
        dc = DataCenter.create_standard_datacenter(
            num_racks=1,
            hosts_per_rack=1,
            cpu_per_host=8,
            ram_per_host=16,
        )
        sim = Simulator(datacenter=dc)

        # VM with only 2 samples [50%, 50%], arrives at t=0
        short_vm = VM(
            vm_id="short_vm",
            cpu_demand=8,
            ram_demand=16,
            arrival_time=0,
            cpu_trace=[50.0, 50.0],
        )
        # Large VM arriving at t=2 requiring full host capacity
        large_vm = VM(
            vm_id="large_vm",
            cpu_demand=8,
            ram_demand=16,
            arrival_time=2,
            cpu_trace=[30.0],
        )

        schedule = {0: [short_vm], 2: [large_vm]}
        metrics = sim.run(workload_schedule=schedule, strategy=FirstFitPlacement(), total_steps=3)

        # At t=2, short_vm should have completed and departed, allowing large_vm to be placed!
        self.assertEqual(metrics.total_arrived_vms, 2)
        self.assertEqual(metrics.total_placed_vms, 2)
        self.assertEqual(metrics.total_rejected_vms, 0)


if __name__ == "__main__":
    unittest.main()
