"""Unit and integration tests for Experiment B2: CPU + Memory Thermal-Aware Q-Learning.

Validates the 11 core requirements for Experiment B2:
1. B2 state contains CPU, memory, and thermal features (6-tuple: VM_CPU, VM_RAM, HOST_CPU, HOST_RAM, HOST_TEMP, DATACENTER_THERMAL_STATE).
2. Changing CPU usage [%] changes the discrete CPU state bin.
3. Changing Memory usage [KB] changes the discrete memory state bin.
4. Host RAM utilization ratio influences the state (HOST_RAM bin).
5. Disk and network features do not influence the state.
6. CPU capacity filtering works correctly.
7. RAM capacity filtering works correctly.
8. Memory unit conversions between KB and GB are exact (1 GB = 1,048,576 KB).
9. Thermal-aware reward remains operational (penalties for hotspots/imbalance, rewards for safe placement).
10. Identical random seeds produce strictly reproducible simulation results.
11. First Fit and Q-Learning execute successfully under identical Bitbrains workload conditions.
"""

import copy
import unittest
from pathlib import Path

from src.bitbrains_loader import BitbrainsLoader
from src.datacenter import DataCenter, Host
from src.metrics import MetricsCollector
from src.placement.first_fit import FirstFitPlacement
from src.qlearning.agent import QLearningPlacementAgent, StateDiscretizerConfig
from src.qlearning.reward import RewardConfig, ThermalRewardCalculator
from src.qlearning.strategy import QLearningSimulatorStrategy
from src.simulator import Simulator
from src.thermal import ThermalConfig, ThermalModel
from src.workload import VM, TraceWorkloadGenerator


class TestExperimentB2(unittest.TestCase):
    """Test suite validating Experiment B2 requirements."""

    def setUp(self):
        self.state_config = StateDiscretizerConfig(
            vm_cpu_thresholds=(2.0, 4.0),
            vm_cpu_util_thresholds=(33.33, 66.67),
            vm_ram_thresholds=(4.0, 16.0),
            vm_ram_kb_thresholds=(4194304.0, 16777216.0),
            host_cpu_thresholds=(0.40, 0.70),
            host_ram_thresholds=(0.40, 0.70),
            host_temp_thresholds=(55.0, 65.0, 75.0),
            hotspot_threshold_c=75.0,
            warm_threshold_c=65.0,
        )
        self.agent_b2 = QLearningPlacementAgent(
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            min_epsilon=0.05,
            epsilon_decay=0.995,
            seed=42,
            state_config=self.state_config,
            state_mode="B2",
        )
        self.dc = DataCenter.create_standard_datacenter(
            num_racks=2,
            hosts_per_rack=2,
            cpu_per_host=16,
            ram_per_host=64,
            initial_temperature_c=25.0,
        )

    # -------------------------------------------------------------------------
    # 1. B2 State Representation (CPU + Memory + Thermal Features)
    # -------------------------------------------------------------------------

    def test_1_b2_state_contains_cpu_memory_and_thermal_features(self):
        """1. Verifies B2 state is a 6-tuple: (VM_CPU, VM_RAM, HOST_CPU, HOST_RAM, HOST_TEMP, DC_THERMAL)."""
        vm = VM(
            vm_id=1,
            cpu_demand=4,
            ram_demand=16,
            arrival_time=0,
            duration=10,
            cpu_trace=[55.0],
            memory_usage_kb=8388608.0,  # 8 GB
            disk_read_kbps=1200.0,
            disk_write_kbps=800.0,
            network_rx_kbps=500.0,
            network_tx_kbps=300.0,
        )
        host = self.dc.hosts[0]
        host.used_cpu = 8  # 50% CPU
        host.used_ram = 32  # 50% RAM
        host.simulated_temperature_c = 58.0

        state = self.agent_b2.discretize_state(vm, host, self.dc)

        self.assertIsInstance(state, tuple)
        self.assertEqual(len(state), 6, f"B2 state must be exactly a 6-tuple, got length {len(state)}: {state}")
        vm_cpu_bin, vm_ram_bin, host_cpu_bin, host_ram_bin, host_temp_bin, dc_thermal_bin = state

        # Verify categorical discrete bins
        self.assertIn(vm_cpu_bin, ["LOW", "MEDIUM", "HIGH"])
        self.assertIn(vm_ram_bin, ["SMALL", "MEDIUM", "LARGE"])
        self.assertIn(host_cpu_bin, ["LOW", "MEDIUM", "HIGH"])
        self.assertIn(host_ram_bin, ["LOW", "MEDIUM", "HIGH"])
        self.assertIn(host_temp_bin, ["COOL", "NORMAL", "WARM", "HOT"])
        self.assertIn(dc_thermal_bin, ["COOL", "MODERATE", "CRITICAL"])

        # Specifically for 55% VM CPU, 8 GB VM RAM, 50% host CPU, 50% host RAM, 58°C temp
        self.assertEqual(vm_cpu_bin, "MEDIUM")
        self.assertEqual(vm_ram_bin, "MEDIUM")
        self.assertEqual(host_cpu_bin, "MEDIUM")
        self.assertEqual(host_ram_bin, "MEDIUM")
        self.assertEqual(host_temp_bin, "NORMAL")
        self.assertEqual(dc_thermal_bin, "COOL")

    # -------------------------------------------------------------------------
    # 2. Changing CPU usage [%] Changes CPU State
    # -------------------------------------------------------------------------

    def test_2_changing_cpu_usage_percent_changes_cpu_state(self):
        """2. Verifies that varying CPU usage [%] across thresholds transitions the VM_CPU bin."""
        host = self.dc.hosts[0]

        # Low CPU utilization (<= 33.33%)
        vm_low = VM(vm_id=101, cpu_demand=4, ram_demand=8, arrival_time=0, cpu_trace=[15.0], memory_usage_kb=8388608.0)
        state_low = self.agent_b2.discretize_state(vm_low, host, self.dc)
        self.assertEqual(state_low[0], "LOW")

        # Medium CPU utilization (33.33% < u <= 66.67%)
        vm_med = VM(vm_id=102, cpu_demand=4, ram_demand=8, arrival_time=0, cpu_trace=[50.0], memory_usage_kb=8388608.0)
        state_med = self.agent_b2.discretize_state(vm_med, host, self.dc)
        self.assertEqual(state_med[0], "MEDIUM")

        # High CPU utilization (> 66.67%)
        vm_high = VM(vm_id=103, cpu_demand=4, ram_demand=8, arrival_time=0, cpu_trace=[88.0], memory_usage_kb=8388608.0)
        state_high = self.agent_b2.discretize_state(vm_high, host, self.dc)
        self.assertEqual(state_high[0], "HIGH")

        # Verify distinct states are observed
        self.assertNotEqual(state_low[0], state_med[0])
        self.assertNotEqual(state_med[0], state_high[0])

    # -------------------------------------------------------------------------
    # 3. Changing Memory Usage [KB] Changes Memory State
    # -------------------------------------------------------------------------

    def test_3_changing_memory_usage_changes_memory_state(self):
        """3. Verifies varying Memory usage [KB] across thresholds transitions the VM_RAM bin."""
        host = self.dc.hosts[0]

        # Small RAM (<= 4 GB = 4,194,304 KB): 1 GB = 1,048,576 KB
        vm_small_mem = VM(vm_id=201, cpu_demand=4, ram_demand=4, arrival_time=0, cpu_trace=[40.0], memory_usage_kb=1048576.0)
        state_small = self.agent_b2.discretize_state(vm_small_mem, host, self.dc)
        self.assertEqual(state_small[1], "SMALL")

        # Medium RAM (4 GB < m <= 16 GB = 16,777,216 KB): 8 GB = 8,388,608 KB
        vm_med_mem = VM(vm_id=202, cpu_demand=4, ram_demand=8, arrival_time=0, cpu_trace=[40.0], memory_usage_kb=8388608.0)
        state_med = self.agent_b2.discretize_state(vm_med_mem, host, self.dc)
        self.assertEqual(state_med[1], "MEDIUM")

        # Large RAM (> 16 GB): 32 GB = 33,554,432 KB
        vm_lrg_mem = VM(vm_id=203, cpu_demand=4, ram_demand=32, arrival_time=0, cpu_trace=[40.0], memory_usage_kb=33554432.0)
        state_lrg = self.agent_b2.discretize_state(vm_lrg_mem, host, self.dc)
        self.assertEqual(state_lrg[1], "LARGE")

        # Verify distinct memory state bins
        self.assertNotEqual(state_small[1], state_med[1])
        self.assertNotEqual(state_med[1], state_lrg[1])

    # -------------------------------------------------------------------------
    # 4. Host RAM Utilization Influences State
    # -------------------------------------------------------------------------

    def test_4_host_ram_utilization_influences_state(self):
        """4. Verifies varying host RAM utilization ratio transitions the HOST_RAM state bin."""
        vm = VM(vm_id=301, cpu_demand=2, ram_demand=4, arrival_time=0, cpu_trace=[30.0], memory_usage_kb=2097152.0)
        host = self.dc.hosts[0]

        # Low host RAM utilization (< 0.40): 10/64 = 15.6%
        host.used_ram = 10
        state_low = self.agent_b2.discretize_state(vm, host, self.dc)
        self.assertEqual(state_low[3], "LOW")

        # Medium host RAM utilization (0.40 <= u < 0.70): 32/64 = 50.0%
        host.used_ram = 32
        state_med = self.agent_b2.discretize_state(vm, host, self.dc)
        self.assertEqual(state_med[3], "MEDIUM")

        # High host RAM utilization (>= 0.70): 55/64 = 85.9%
        host.used_ram = 55
        state_high = self.agent_b2.discretize_state(vm, host, self.dc)
        self.assertEqual(state_high[3], "HIGH")

        self.assertNotEqual(state_low[3], state_med[3])
        self.assertNotEqual(state_med[3], state_high[3])

    # -------------------------------------------------------------------------
    # 5. Disk and Network Features Do Not Influence State
    # -------------------------------------------------------------------------

    def test_5_disk_and_network_features_do_not_influence_state(self):
        """5. Verifies disk throughput and network traffic variations leave the B2 state unchanged."""
        host = self.dc.hosts[0]

        vm_idle_io = VM(
            vm_id=401,
            cpu_demand=4,
            ram_demand=8,
            arrival_time=0,
            cpu_trace=[40.0],
            memory_usage_kb=4194304.0,
            disk_read_kbps=0.0,
            disk_write_kbps=0.0,
            network_rx_kbps=0.0,
            network_tx_kbps=0.0,
        )
        vm_heavy_io = VM(
            vm_id=402,
            cpu_demand=4,
            ram_demand=8,
            arrival_time=0,
            cpu_trace=[40.0],
            memory_usage_kb=4194304.0,
            disk_read_kbps=600000.0,
            disk_write_kbps=450000.0,
            network_rx_kbps=1500000.0,
            network_tx_kbps=900000.0,
        )

        state_idle = self.agent_b2.discretize_state(vm_idle_io, host, self.dc)
        state_heavy = self.agent_b2.discretize_state(vm_heavy_io, host, self.dc)

        self.assertEqual(
            state_idle,
            state_heavy,
            f"B2 state changed when altering disk/network I/O: {state_idle} vs {state_heavy}",
        )

    # -------------------------------------------------------------------------
    # 6. CPU Capacity Filtering Works Correctly
    # -------------------------------------------------------------------------

    def test_6_cpu_capacity_filtering_works_correctly(self):
        """6. Verifies host with insufficient CPU capacity rejects VM placement."""
        host = self.dc.hosts[0]
        # Host limit: 16 cores. Used: 14 cores (2 available)
        host.used_cpu = 14
        host.used_ram = 10

        vm_fits = VM(vm_id=501, cpu_demand=2, ram_demand=4, arrival_time=0)
        vm_exceeds = VM(vm_id=502, cpu_demand=4, ram_demand=4, arrival_time=0)

        # 14 + 2 <= 16 -> True
        self.assertTrue(host.can_accommodate(vm_fits))
        # 14 + 4 = 18 > 16 -> False
        self.assertFalse(host.can_accommodate(vm_exceeds))

        eligible = self.agent_b2.get_eligible_hosts(vm_exceeds, self.dc)
        self.assertNotIn(host, eligible)
        self.assertFalse(host.allocate(vm_exceeds))

    # -------------------------------------------------------------------------
    # 7. RAM Capacity Filtering Works Correctly
    # -------------------------------------------------------------------------

    def test_7_ram_capacity_filtering_works_correctly(self):
        """7. Verifies host with insufficient RAM capacity rejects VM placement."""
        host = self.dc.hosts[0]
        # Host limit: 16 CPU cores, 64 GB RAM.
        # CPU is free (0/16 used), but RAM is full (60/64 used, 4 GB available)
        host.used_cpu = 0
        host.used_ram = 60

        vm_fits_ram = VM(vm_id=601, cpu_demand=2, ram_demand=4, arrival_time=0)
        vm_exceeds_ram = VM(vm_id=602, cpu_demand=2, ram_demand=8, arrival_time=0)

        # 60 + 4 <= 64 -> True
        self.assertTrue(host.can_accommodate(vm_fits_ram))
        # 60 + 8 = 68 > 64 -> False
        self.assertFalse(host.can_accommodate(vm_exceeds_ram))

        eligible = self.agent_b2.get_eligible_hosts(vm_exceeds_ram, self.dc)
        self.assertNotIn(host, eligible)
        self.assertFalse(host.allocate(vm_exceeds_ram))

    # -------------------------------------------------------------------------
    # 8. Memory Unit Conversions Are Correct
    # -------------------------------------------------------------------------

    def test_8_memory_unit_conversions_are_correct(self):
        """8. Validates exact numerical conversion between Kilobytes (KB) and Gigabytes (GB)."""
        kb_per_gb = 1024 * 1024  # 1,048,576 KB

        # Test known standard sizing conversions
        test_cases = [
            (1048576.0, 1.0),
            (2097152.0, 2.0),
            (4194304.0, 4.0),
            (8388608.0, 8.0),
            (16777216.0, 16.0),
            (33554432.0, 32.0),
            (67108864.0, 64.0),
        ]
        for kb_val, expected_gb in test_cases:
            converted_gb = kb_val / kb_per_gb
            self.assertEqual(converted_gb, expected_gb)

        # Verify VM helper properties
        vm = VM(
            vm_id=701,
            cpu_demand=2,
            ram_demand=4,
            arrival_time=0,
            memory_usage_kb=4194304.0,
            memory_provisioned_kb=16777216.0,
        )
        self.assertEqual(vm.current_memory_usage_gb, 4.0)
        self.assertEqual(vm.current_memory_provisioned_gb, 16.0)

    # -------------------------------------------------------------------------
    # 9. Thermal-Aware Reward Remains Operational
    # -------------------------------------------------------------------------

    def test_9_thermal_aware_reward_remains_operational(self):
        """9. Verifies the thermal reward calculator assesses hotspot penalties and thermal imbalance."""
        reward_calc = ThermalRewardCalculator(RewardConfig(
            hotspot_threshold_c=75.0,
            w_success=10.0,
            w_reject=5.0,
            w_new_hotspot=15.0,
            w_imbalance=2.0,
        ))

        # Case A: Normal placement, low temp, no hotspots, no imbalance delta
        bd_normal = reward_calc.compute_reward(
            placement_success=True,
            host_temp_c=50.0,
            host_util=0.5,
            is_new_hotspot=False,
            is_existing_hotspot=False,
            imbalance_before=1.0,
            imbalance_after=1.0,
        )
        self.assertGreater(bd_normal.total_reward, 0.0)

        # Case B: Critical hotspot placement (78°C >= 75°C, new hotspot)
        bd_hotspot = reward_calc.compute_reward(
            placement_success=True,
            host_temp_c=78.0,
            host_util=0.5,
            is_new_hotspot=True,
            is_existing_hotspot=False,
            imbalance_before=1.0,
            imbalance_after=1.0,
        )
        self.assertLess(bd_hotspot.total_reward, bd_normal.total_reward)
        self.assertLess(bd_hotspot.total_reward, 0.0)

        # Case C: Placement worsens thermal imbalance
        bd_imbalance = reward_calc.compute_reward(
            placement_success=True,
            host_temp_c=55.0,
            host_util=0.5,
            is_new_hotspot=False,
            is_existing_hotspot=False,
            imbalance_before=1.0,
            imbalance_after=6.0,  # delta = +5.0
        )
        self.assertLess(bd_imbalance.total_reward, bd_normal.total_reward)

        # Case D: Rejection penalty
        bd_reject = reward_calc.compute_reward(placement_success=False)
        self.assertEqual(bd_reject.total_reward, -5.0)

    # -------------------------------------------------------------------------
    # 10. Identical Seeds Produce Reproducible Results
    # -------------------------------------------------------------------------

    def test_10_identical_seeds_produce_reproducible_results(self):
        """10. Verifies identical random seeds yield identical simulation trajectories."""
        traces_dir = Path("dataset/bitbrains/fastStorage/2013-8")
        if not traces_dir.exists():
            self.skipTest(f"Dataset directory not found: {traces_dir}")

        loader = BitbrainsLoader()
        report = loader.load_with_report(directory_path=traces_dir, max_traces=10, recursive=True)
        self.assertGreater(report.total_loaded, 0)

        def run_sim_with_seed(seed: int):
            gen = TraceWorkloadGenerator(
                traces=report.valid_traces,
                arrival_rate_lambda=2.0,
                max_arrival_time=20,
                seed=seed,
            )
            schedule = gen.generate_schedule_by_step(total_time_steps=20)

            dc = DataCenter.create_standard_datacenter(
                num_racks=1,
                hosts_per_rack=2,
                cpu_per_host=16,
                ram_per_host=64,
                initial_temperature_c=25.0,
            )
            thermal_model = ThermalModel(ThermalConfig(ambient_temperature_c=25.0, hotspot_threshold_c=75.0))

            agent = QLearningPlacementAgent(seed=seed, state_mode="B2")
            strategy = QLearningSimulatorStrategy(agent=agent, training=True)

            sim = Simulator(datacenter=dc, thermal_model=thermal_model)
            metrics = sim.run(workload_schedule=schedule, strategy=strategy, total_steps=20)
            return metrics.summary(), strategy.stats.cumulative_reward

        summary1, reward1 = run_sim_with_seed(42)
        summary2, reward2 = run_sim_with_seed(42)

        self.assertEqual(summary1["total_placed_vms"], summary2["total_placed_vms"])
        self.assertEqual(summary1["total_rejected_vms"], summary2["total_rejected_vms"])
        self.assertAlmostEqual(summary1["peak_temperature_c"], summary2["peak_temperature_c"], places=5)
        self.assertAlmostEqual(reward1, reward2, places=5)

    # -------------------------------------------------------------------------
    # 11. First Fit and Q-Learning Execute Successfully
    # -------------------------------------------------------------------------

    def test_11_first_fit_and_qlearning_execute_successfully(self):
        """11. Verifies First Fit and Q-Learning (B2) both execute successfully on identical Bitbrains schedule."""
        traces_dir = Path("dataset/bitbrains/fastStorage/2013-8")
        if not traces_dir.exists():
            self.skipTest(f"Dataset directory not found: {traces_dir}")

        loader = BitbrainsLoader()
        report = loader.load_with_report(directory_path=traces_dir, max_traces=15, recursive=True)
        self.assertGreater(report.total_loaded, 0)

        # Generate single deterministic schedule used by both
        gen = TraceWorkloadGenerator(
            traces=report.valid_traces,
            arrival_rate_lambda=2.0,
            max_arrival_time=30,
            seed=42,
        )
        schedule = gen.generate_schedule_by_step(total_time_steps=30)

        # Run First Fit
        dc_ff = DataCenter.create_standard_datacenter(
            num_racks=2,
            hosts_per_rack=2,
            cpu_per_host=16,
            ram_per_host=64,
            initial_temperature_c=25.0,
        )
        sim_ff = Simulator(datacenter=dc_ff, thermal_model=ThermalModel(ThermalConfig()))
        metrics_ff = sim_ff.run(
            workload_schedule=copy.deepcopy(schedule),
            strategy=FirstFitPlacement(),
            total_steps=30,
        )
        summary_ff = metrics_ff.summary()

        # Run Q-Learning (B2)
        dc_ql = DataCenter.create_standard_datacenter(
            num_racks=2,
            hosts_per_rack=2,
            cpu_per_host=16,
            ram_per_host=64,
            initial_temperature_c=25.0,
        )
        agent_ql = QLearningPlacementAgent(seed=42, state_mode="B2")
        strategy_ql = QLearningSimulatorStrategy(agent=agent_ql, training=True)
        sim_ql = Simulator(datacenter=dc_ql, thermal_model=ThermalModel(ThermalConfig()))
        metrics_ql = sim_ql.run(
            workload_schedule=copy.deepcopy(schedule),
            strategy=strategy_ql,
            total_steps=30,
        )
        summary_ql = metrics_ql.summary()

        # Verify both produced valid summaries
        for summary in [summary_ff, summary_ql]:
            self.assertIn("total_placed_vms", summary)
            self.assertIn("total_rejected_vms", summary)
            self.assertIn("average_cpu_utilization", summary)
            self.assertIn("average_ram_utilization", summary)
            self.assertIn("steps_with_hotspots", summary)
            self.assertIn("peak_temperature_c", summary)
            self.assertIn("average_thermal_imbalance_std", summary)
            self.assertGreaterEqual(summary["total_placed_vms"], 0)
            self.assertGreaterEqual(summary["total_rejected_vms"], 0)

        self.assertIsInstance(strategy_ql.stats.cumulative_reward, float)


if __name__ == "__main__":
    unittest.main()

