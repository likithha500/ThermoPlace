"""Comprehensive test suite for Phase 5: CPU + RAM + Disk-Aware VM Placement.

Verifies:
1. Dataset & Trace tests (disk read/write extraction, total disk I/O, missing values, timestamps).
2. Datacenter tests (disk capacity initialization, utilization, allocation, release, reset).
3. First-Fit placement tests (multi-resource checking, first eligible selection, rejection, backwards compatibility).
4. Q-Learning agent tests (extended state discretization, action eligibility, updates, reward extension, table compatibility).
5. Integration & Simulation tests (First-Fit vs Q-Learning with CPU+RAM+Disk, evaluation mode immutability, leak prevention).
"""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.dataset_loader import VMTrace, DatasetLoadReport
from src.bitbrains_loader import BitbrainsLoader, _parse_bitbrains_file
from src.workload import VM, WorkloadGenerator, TraceWorkloadGenerator
from src.datacenter import Host, Rack, DataCenter
from src.placement.first_fit import FirstFitPlacement
from src.thermal import ThermalModel, ThermalConfig
from src.simulator import Simulator
from src.metrics import MetricsCollector
from src.qlearning.agent import QLearningPlacementAgent, StateDiscretizerConfig
from src.qlearning.reward import ThermalRewardCalculator, RewardConfig, RewardBreakdown
from src.qlearning.strategy import QLearningSimulatorStrategy
from src.qlearning.trainer import QLearningTrainer, TrainingConfig, EpisodeRecord


class TestPhase5Dataset(unittest.TestCase):
    """1. Dataset and Workload Trace Tests."""

    def test_disk_read_and_write_values_extracted_correctly(self):
        csv_content = (
            "Timestamp [ms];\tCPU cores;\tCPU capacity provisioned [MHZ];"
            "\tCPU usage [MHZ];\tCPU usage [%];\tMemory capacity provisioned [KB];"
            "\tMemory usage [KB];\tDisk read throughput [KB/s];"
            "\tDisk write throughput [KB/s];\tNetwork received throughput [KB/s];"
            "\tNetwork transmitted throughput [KB/s]\n"
            "1376314846;\t2;\t2000.0;\t500.0;\t25.0;\t4096.0;\t2048.0;\t120.5;\t340.2;\t0.0;\t0.0\n"
            "1376315146;\t2;\t2000.0;\t1000.0;\t50.0;\t4096.0;\t2048.0;\t200.0;\t500.0;\t0.0;\t0.0\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = Path(f.name)

        try:
            trace = _parse_bitbrains_file(temp_path, load_disk=True)
            self.assertEqual(trace.vm_id, temp_path.stem)
            self.assertEqual(len(trace.cpu_samples), 2)
            self.assertIsNotNone(trace.disk_read_samples)
            self.assertIsNotNone(trace.disk_write_samples)
            self.assertEqual(trace.disk_read_samples, [120.5, 200.0])
            self.assertEqual(trace.disk_write_samples, [340.2, 500.0])
            self.assertAlmostEqual(trace.mean_disk_read, 160.25)
            self.assertEqual(trace.max_disk_read, 200.0)
            self.assertAlmostEqual(trace.mean_disk_write, 420.1)
            self.assertEqual(trace.max_disk_write, 500.0)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_total_disk_io_calculated_correctly(self):
        trace = VMTrace(
            vm_id="vm_calc",
            cpu_samples=[10.0, 20.0],
            source_file="mock.csv",
            disk_read_samples=[100.0, 250.0],
            disk_write_samples=[50.0, 150.0],
        )
        total_io = trace.total_disk_io_samples
        self.assertIsNotNone(total_io)
        self.assertEqual(total_io, [150.0, 400.0])
        self.assertAlmostEqual(trace.mean_disk_io, 275.0)
        self.assertEqual(trace.max_disk_io, 400.0)

    def test_missing_or_invalid_disk_values_handled_safely(self):
        # Row with invalid (negative or non-numeric) disk values
        csv_content = (
            "Timestamp [ms];\tCPU cores;\tCPU capacity provisioned [MHZ];"
            "\tCPU usage [MHZ];\tCPU usage [%];\tMemory capacity provisioned [KB];"
            "\tMemory usage [KB];\tDisk read throughput [KB/s];"
            "\tDisk write throughput [KB/s];\tNetwork received throughput [KB/s];"
            "\tNetwork transmitted throughput [KB/s]\n"
            "1376314846;\t2;\t2000.0;\t500.0;\t25.0;\t4096.0;\t2048.0;\t10.0;\t20.0;\t0.0;\t0.0\n"
            "1376315146;\t2;\t2000.0;\t500.0;\t25.0;\t4096.0;\t2048.0;\tNaN;\t-5.0;\t0.0;\t0.0\n"
            "1376315446;\t2;\t2000.0;\t500.0;\t30.0;\t4096.0;\t2048.0;\t15.0;\t25.0;\t0.0;\t0.0\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = Path(f.name)

        try:
            trace = _parse_bitbrains_file(temp_path, load_disk=True)
            # The invalid middle row is dropped cleanly, leaving 2 synchronized valid rows
            self.assertEqual(len(trace.cpu_samples), 2)
            self.assertEqual(trace.cpu_samples, [25.0, 30.0])
            self.assertEqual(trace.disk_read_samples, [10.0, 15.0])
            self.assertEqual(trace.disk_write_samples, [20.0, 25.0])
            self.assertEqual(trace.timestamps, [1376314846, 1376315446])
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_timestamp_progression_preserved(self):
        trace = VMTrace(
            vm_id="vm_ts",
            cpu_samples=[10.0, 20.0, 30.0],
            source_file="mock.csv",
            disk_read_samples=[5.0, 10.0, 15.0],
            disk_write_samples=[20.0, 30.0, 40.0],
            timestamps=[1000, 2000, 3000],
        )
        vm = VM.from_trace(trace=trace)
        self.assertEqual(vm.current_timestamp, 1000)
        self.assertEqual(vm.current_disk_read, 5.0)
        self.assertEqual(vm.current_disk_write, 20.0)
        self.assertEqual(vm.current_total_disk_io, 25.0)

        vm.tick()
        self.assertEqual(vm.current_timestamp, 2000)
        self.assertEqual(vm.current_disk_read, 10.0)
        self.assertEqual(vm.current_disk_write, 30.0)
        self.assertEqual(vm.current_total_disk_io, 40.0)

        vm.tick()
        self.assertEqual(vm.current_timestamp, 3000)
        self.assertEqual(vm.current_disk_read, 15.0)

    def test_existing_cpu_and_ram_parsing_remains_unchanged(self):
        trace = VMTrace(
            vm_id="vm_legacy",
            cpu_samples=[30.0, 60.0],
            source_file="mock.csv",
        )
        self.assertIsNone(trace.disk_read_samples)
        self.assertIsNone(trace.disk_write_samples)
        self.assertFalse(trace.has_disk_trace)
        self.assertAlmostEqual(trace.mean_cpu, 45.0)


class TestPhase5Datacenter(unittest.TestCase):
    """2. Datacenter Resource Model Tests with Disk I/O."""

    def setUp(self):
        self.host = Host(
            host_id="H0",
            rack_id=0,
            rack_slot=0,
            total_cpu=16,
            total_ram=64,
            total_disk_io=10000.0,  # 10,000 KB/s capacity
            feature_set="cpu_ram_disk",
        )

    def test_disk_capacity_initialization_and_utilization(self):
        self.assertEqual(self.host.total_disk_io, 10000.0)
        self.assertEqual(self.host.used_disk_io, 0.0)
        self.assertEqual(self.host.available_disk_io, 10000.0)
        self.assertEqual(self.host.disk_io_utilization, 0.0)

    def test_disk_capacity_enforced_in_can_accommodate(self):
        # Fits: 4 CPU, 16 RAM, 4000 KB/s Disk
        vm_fit = VM(
            vm_id="vm1",
            cpu_demand=4,
            ram_demand=16,
            arrival_time=0,
            duration=5,
            disk_io_demand=4000.0,
        )
        self.assertTrue(self.host.can_accommodate(vm_fit))

        # Exceeds Disk: 4 CPU, 16 RAM, 12000 KB/s Disk (> 10000 limit)
        vm_overflow = VM(
            vm_id="vm_over",
            cpu_demand=4,
            ram_demand=16,
            arrival_time=0,
            duration=5,
            disk_io_demand=12000.0,
        )
        self.assertFalse(self.host.can_accommodate(vm_overflow))

    def test_disk_usage_increases_after_placement_and_releases(self):
        vm = VM(
            vm_id="vm1",
            cpu_demand=4,
            ram_demand=16,
            arrival_time=0,
            duration=5,
            disk_read_demand=2000.0,
            disk_write_demand=3000.0,
            disk_io_demand=5000.0,
        )
        self.assertTrue(self.host.allocate(vm))
        self.assertEqual(self.host.used_cpu, 4)
        self.assertEqual(self.host.used_ram, 16)
        self.assertEqual(self.host.used_disk_read, 2000.0)
        self.assertEqual(self.host.used_disk_write, 3000.0)
        self.assertEqual(self.host.used_disk_io, 5000.0)
        self.assertEqual(self.host.available_disk_io, 5000.0)
        self.assertAlmostEqual(self.host.disk_io_utilization, 0.5)

        # Release
        released = self.host.release("vm1")
        self.assertIsNotNone(released)
        self.assertEqual(self.host.used_cpu, 0)
        self.assertEqual(self.host.used_ram, 0)
        self.assertEqual(self.host.used_disk_read, 0.0)
        self.assertEqual(self.host.used_disk_write, 0.0)
        self.assertEqual(self.host.used_disk_io, 0.0)
        self.assertEqual(self.host.available_disk_io, 10000.0)
        self.assertEqual(self.host.disk_io_utilization, 0.0)

    def test_disk_usage_resets_correctly_between_episodes(self):
        vm = VM(vm_id="vm_tmp", cpu_demand=2, ram_demand=4, arrival_time=0, duration=10, disk_io_demand=3000.0)
        self.host.allocate(vm)
        self.assertEqual(self.host.used_disk_io, 3000.0)

        self.host.reset()
        self.assertEqual(self.host.used_disk_io, 0.0)
        self.assertEqual(self.host.used_disk_read, 0.0)
        self.assertEqual(self.host.used_disk_write, 0.0)
        self.assertEqual(len(self.host.allocated_vms), 0)

    def test_cpu_and_ram_capacity_enforcement_continues_working(self):
        # CPU exceed
        vm_cpu_over = VM(vm_id="cpu_over", cpu_demand=20, ram_demand=8, arrival_time=0, duration=5, disk_io_demand=100.0)
        self.assertFalse(self.host.can_accommodate(vm_cpu_over))

        # RAM exceed
        vm_ram_over = VM(vm_id="ram_over", cpu_demand=4, ram_demand=128, arrival_time=0, duration=5, disk_io_demand=100.0)
        self.assertFalse(self.host.can_accommodate(vm_ram_over))


class TestPhase5FirstFit(unittest.TestCase):
    """3. First-Fit Multi-Resource Placement Tests."""

    def setUp(self):
        self.dc = DataCenter.create_standard_datacenter(
            num_racks=1,
            hosts_per_rack=3,
            cpu_per_host=8,
            ram_per_host=16,
            disk_io_per_host=10000.0,
            feature_set="cpu_ram_disk",
        )
        self.strategy = FirstFitPlacement(feature_set="cpu_ram_disk")

    def test_first_fit_checks_cpu_ram_and_disk(self):
        # Host 0 fits VM1
        vm1 = VM(vm_id=1, cpu_demand=4, ram_demand=8, arrival_time=0, duration=10, disk_io_demand=6000.0)
        h1 = self.strategy.select_host(vm1, self.dc)
        self.assertIsNotNone(h1)
        self.assertEqual(h1.host_id, "R0_H0")
        h1.allocate(vm1, feature_set="cpu_ram_disk")

        # VM2 needs 4 CPU, 8 RAM, 6000 Disk.
        # Host 0 has 4 CPU and 8 RAM left, but only 4000 Disk left (< 6000 required).
        # First-Fit must skip Host 0 and place VM2 on Host 1!
        vm2 = VM(vm_id=2, cpu_demand=4, ram_demand=8, arrival_time=0, duration=10, disk_io_demand=6000.0)
        h2 = self.strategy.select_host(vm2, self.dc)
        self.assertIsNotNone(h2)
        self.assertEqual(h2.host_id, "R0_H1")

    def test_first_fit_rejects_vm_when_no_host_satisfies_all_three(self):
        # VM requires 15000 Disk IO (all hosts only have 10000)
        vm_huge_disk = VM(vm_id=99, cpu_demand=2, ram_demand=4, arrival_time=0, duration=5, disk_io_demand=15000.0)
        selected = self.strategy.select_host(vm_huge_disk, self.dc)
        self.assertIsNone(selected)

    def test_cpu_ram_mode_backward_compatibility(self):
        ff_legacy = FirstFitPlacement(feature_set="cpu_ram")
        # In cpu_ram mode, disk demand is NOT enforced
        vm_big_disk = VM(vm_id=1, cpu_demand=4, ram_demand=8, arrival_time=0, duration=10, disk_io_demand=50000.0)
        selected = ff_legacy.select_host(vm_big_disk, self.dc)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.host_id, "R0_H0")


class TestPhase5QLearning(unittest.TestCase):
    """4. Q-Learning State, Action Eligibility, and Reward Tests."""

    def setUp(self):
        self.dc = DataCenter.create_standard_datacenter(
            num_racks=1,
            hosts_per_rack=3,
            cpu_per_host=8,
            ram_per_host=16,
            disk_io_per_host=10000.0,
            feature_set="cpu_ram_disk",
        )
        self.agent = QLearningPlacementAgent(
            seed=42,
            feature_set="cpu_ram_disk",
        )

    def test_extended_state_representation_with_disk(self):
        # In cpu_ram_disk mode, state is a 7-tuple containing host disk bin
        state = self.agent.discretize_state(
            vm_cpu=2.0,
            vm_ram=4.0,
            host_cpu=0.20,
            host_ram=0.30,
            host_disk=0.85,  # High disk
            host_temp=40.0,
            dc_thermal="COOL",
        )
        self.assertEqual(len(state), 7)
        self.assertEqual(state[0], "SMALL")   # VM CPU
        self.assertEqual(state[1], "SMALL")   # VM RAM
        self.assertEqual(state[2], "LOW")     # Host CPU
        self.assertEqual(state[3], "LOW")     # Host RAM
        self.assertEqual(state[4], "HIGH")    # Host Disk (>= 0.70)
        self.assertEqual(state[5], "COOL")    # Host Temp
        self.assertEqual(state[6], "COOL")    # DC Thermal

    def test_cpu_ram_baseline_state_remains_6_tuple(self):
        agent_baseline = QLearningPlacementAgent(feature_set="cpu_ram")
        state = agent_baseline.discretize_state(
            vm_cpu=2.0,
            vm_ram=4.0,
            host_cpu=0.20,
            host_ram=0.30,
            host_temp=40.0,
            dc_thermal="COOL",
        )
        self.assertEqual(len(state), 6)

    def test_action_eligibility_filters_disk_constrained_hosts(self):
        host0 = self.dc.hosts[0]
        # Allocate 8000 Disk on host0
        vm_fill = VM(vm_id="f0", cpu_demand=2, ram_demand=4, arrival_time=0, duration=10, disk_io_demand=8000.0)
        host0.allocate(vm_fill, feature_set="cpu_ram_disk")

        # New VM needs 5000 Disk. Host 0 only has 2000 remaining.
        vm_new = VM(vm_id="n1", cpu_demand=2, ram_demand=4, arrival_time=0, duration=10, disk_io_demand=5000.0)
        eligible = self.agent.get_eligible_hosts(vm_new, self.dc)
        self.assertEqual(len(eligible), 2)
        self.assertNotIn(host0, eligible)

    def test_epsilon_greedy_selects_only_eligible_actions(self):
        # Set epsilon = 1.0 (pure exploration)
        self.agent.epsilon = 1.0
        host0 = self.dc.hosts[0]
        host1 = self.dc.hosts[1]
        # Max out host0 and host1 disk
        host0.allocate(VM(vm_id="f0", cpu_demand=2, ram_demand=4, arrival_time=0, duration=10, disk_io_demand=10000.0), feature_set="cpu_ram_disk")
        host1.allocate(VM(vm_id="f1", cpu_demand=2, ram_demand=4, arrival_time=0, duration=10, disk_io_demand=10000.0), feature_set="cpu_ram_disk")

        # Only Host 2 has capacity left
        vm_new = VM(vm_id="n1", cpu_demand=2, ram_demand=4, arrival_time=0, duration=10, disk_io_demand=5000.0)
        for _ in range(10):
            selected = self.agent.select_host(vm_new, self.dc)
            self.assertEqual(selected.host_id, "R0_H2")

    def test_q_table_persistence_and_incompatible_feature_rejection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_disk_qtable.json"
            self.agent.set_q_value(("SMALL", "SMALL", "LOW", "LOW", "LOW", "COOL", "COOL"), "R0_H0", 12.34)
            self.agent.save(model_path)

            # Compatible agent loads successfully
            agent_comp = QLearningPlacementAgent(feature_set="cpu_ram_disk")
            agent_comp.load(model_path, validate_config=True)
            self.assertAlmostEqual(agent_comp.get_q_value(("SMALL", "SMALL", "LOW", "LOW", "LOW", "COOL", "COOL"), "R0_H0"), 12.34)

            # Incompatible agent (cpu_ram) rejects disk Q-table
            agent_incomp = QLearningPlacementAgent(feature_set="cpu_ram")
            with self.assertRaises(ValueError):
                agent_incomp.load(model_path, validate_config=True)

    def test_reward_function_with_disk_penalty(self):
        reward_calc = ThermalRewardCalculator(
            RewardConfig(
                w_disk_penalty=5.0,
                disk_saturation_threshold=0.80,
            )
        )
        # Moderate disk (0.50 <= 0.80) -> no penalty
        r_mod = reward_calc.compute_reward(
            placement_success=True,
            host_temp_c=30.0,
            host_util=0.5,
            host_disk_util=0.50,
        )
        self.assertEqual(r_mod.disk_penalty, 0.0)

        # High disk (0.90 > 0.80) -> penalized: 5.0 * (0.9 - 0.8)/(1 - 0.8) = 5.0 * 0.5 = 2.5
        r_high = reward_calc.compute_reward(
            placement_success=True,
            host_temp_c=30.0,
            host_util=0.5,
            host_disk_util=0.90,
        )
        self.assertAlmostEqual(r_high.disk_penalty, 2.5)
        self.assertAlmostEqual(r_mod.total_reward - r_high.total_reward, 2.5)


class TestPhase5Integration(unittest.TestCase):
    """5. End-to-end Integration and Simulation Tests."""

    def test_first_fit_and_qlearning_run_simulation_with_disk(self):
        dc_ff = DataCenter.create_standard_datacenter(
            num_racks=1, hosts_per_rack=2, cpu_per_host=8, ram_per_host=16, disk_io_per_host=5000.0, feature_set="cpu_ram_disk"
        )
        dc_ql = DataCenter.create_standard_datacenter(
            num_racks=1, hosts_per_rack=2, cpu_per_host=8, ram_per_host=16, disk_io_per_host=5000.0, feature_set="cpu_ram_disk"
        )
        sim_ff = Simulator(datacenter=dc_ff)
        sim_ql = Simulator(datacenter=dc_ql)

        vms = [
            VM(vm_id=1, cpu_demand=2, ram_demand=4, arrival_time=0, duration=3, disk_io_demand=2000.0),
            VM(vm_id=2, cpu_demand=2, ram_demand=4, arrival_time=0, duration=3, disk_io_demand=4000.0),
            VM(vm_id=3, cpu_demand=2, ram_demand=4, arrival_time=1, duration=3, disk_io_demand=5000.0),
        ]
        schedule_ff = {0: [vms[0], vms[1]], 1: [vms[2]]}
        schedule_ql = copy.deepcopy(schedule_ff)

        strat_ff = FirstFitPlacement(feature_set="cpu_ram_disk")
        agent_ql = QLearningPlacementAgent(seed=42, feature_set="cpu_ram_disk")
        strat_ql = QLearningSimulatorStrategy(agent=agent_ql, training=True)

        m_ff = sim_ff.run(schedule_ff, strategy=strat_ff, total_steps=3)
        m_ql = sim_ql.run(schedule_ql, strategy=strat_ql, total_steps=3)

        sum_ff = m_ff.summary()
        sum_ql = m_ql.summary()

        self.assertIn("average_disk_io_utilization", sum_ff)
        self.assertIn("average_disk_io_utilization", sum_ql)
        self.assertEqual(sum_ff["disk_capacity_violations"], 0)
        self.assertEqual(sum_ql["disk_capacity_violations"], 0)

    def test_evaluation_mode_freezes_q_table_and_reproducible(self):
        agent = QLearningPlacementAgent(seed=42, feature_set="cpu_ram_disk")
        # Set some Q-values
        state_key = ("SMALL", "SMALL", "LOW", "LOW", "LOW", "COOL", "COOL")
        agent.set_q_value(state_key, "R0_H0", 5.0)

        strat_eval = QLearningSimulatorStrategy(agent=agent, training=False)
        self.assertEqual(agent.epsilon, 0.0)

        dc = DataCenter.create_standard_datacenter(
            num_racks=1, hosts_per_rack=2, cpu_per_host=8, ram_per_host=16, disk_io_per_host=10000.0, feature_set="cpu_ram_disk"
        )
        sim = Simulator(datacenter=dc)
        vm = VM(vm_id="eval1", cpu_demand=2, ram_demand=4, arrival_time=0, duration=2, disk_io_demand=1000.0)
        sim.run({0: [vm]}, strategy=strat_eval, total_steps=2)

        # Q-table value must be identical (not modified during evaluation)
        self.assertEqual(agent.get_q_value(state_key, "R0_H0"), 5.0)
        self.assertEqual(agent.epsilon, 0.0)

    def test_no_resource_leakage_between_episodes(self):
        config = TrainingConfig(
            episodes=2,
            steps=5,
            racks=1,
            hosts_per_rack=2,
            features="cpu_ram_disk",
            disk_io_per_host=10000.0,
            save_plots=False,
            save_model=False,
        )
        trainer = QLearningTrainer(config=config)
        trainer.train(episodes=2, verbose=False)

        # After training ends, datacenter should have 0 allocated VMs and 0 disk usage
        trainer.simulator.reset(initial_temp=config.ambient_temp)
        for host in trainer.datacenter.hosts:
            self.assertEqual(host.used_cpu, 0)
            self.assertEqual(host.used_ram, 0)
            self.assertEqual(host.used_disk_io, 0.0)
            self.assertEqual(len(host.allocated_vms), 0)


if __name__ == "__main__":
    unittest.main()

