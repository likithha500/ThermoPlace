"""Unit and integration tests for Experiment B1: CPU-Only Thermal-Aware Q-Learning.

Validates the 9 core requirements for Experiment B1:
1. B1 state includes CPU and thermal features only (4-tuple: VM_CPU, HOST_CPU, HOST_TEMP, DATACENTER_THERMAL_STATE).
2. Changing CPU usage [%] changes the discrete CPU state.
3. Changing VM memory usage [KB] does not change the agent state.
4. Changing disk or network measurements does not change the agent state.
5. CPU capacity constraints work properly.
6. RAM physical feasibility checks remain functional as safety constraints.
7. Thermal-aware reward remains operational (penalties for hotspots/imbalance, rewards for safe placement).
8. Identical random seeds produce strictly reproducible simulation results.
9. First Fit and Q-Learning execute successfully under identical Bitbrains workload conditions.
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


class TestExperimentB1(unittest.TestCase):
    """Test suite validating Experiment B1 requirements."""

    def setUp(self):
        self.state_config = StateDiscretizerConfig(
            vm_cpu_thresholds=(2.0, 4.0),
            vm_cpu_util_thresholds=(33.33, 66.67),
            host_cpu_thresholds=(0.40, 0.70),
            host_ram_thresholds=(0.40, 0.70),
            host_temp_thresholds=(55.0, 65.0, 75.0),
            hotspot_threshold_c=75.0,
            warm_threshold_c=65.0,
        )
        self.agent_b1 = QLearningPlacementAgent(
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            min_epsilon=0.05,
            epsilon_decay=0.995,
            seed=42,
            state_config=self.state_config,
            state_mode="B1",
        )
        self.dc = DataCenter.create_standard_datacenter(
            num_racks=2,
            hosts_per_rack=2,
            cpu_per_host=16,
            ram_per_host=64,
            initial_temperature_c=25.0,
        )

    # -------------------------------------------------------------------------
    # 1. B1 State Representation (CPU & Thermal Features Only)
    # -------------------------------------------------------------------------

    def test_1_b1_state_includes_cpu_and_thermal_features_only(self):
        """1. Verifies B1 state is a 4-tuple: (VM_CPU, HOST_CPU, HOST_TEMP, DATACENTER_THERMAL_STATE)."""
        vm = VM(
            vm_id=1,
            cpu_demand=4,
            ram_demand=16,
            arrival_time=0,
            duration=10,
            cpu_trace=[55.0],
            memory_usage_kb=4194304.0,
            disk_read_kbps=1200.0,
            disk_write_kbps=800.0,
            network_rx_kbps=500.0,
            network_tx_kbps=300.0,
        )
        host = self.dc.hosts[0]
        host.used_cpu = 8  # 50% CPU
        host.used_ram = 32  # 50% RAM
        host.simulated_temperature_c = 58.0

        state = self.agent_b1.discretize_state(vm, host, self.dc)

        self.assertIsInstance(state, tuple)
        self.assertEqual(len(state), 4, f"B1 state must be exactly a 4-tuple, got length {len(state)}: {state}")
        vm_cpu_bin, host_cpu_bin, host_temp_bin, dc_thermal_bin = state

        # Verify categorical discrete bins
        self.assertIn(vm_cpu_bin, ["LOW", "MEDIUM", "HIGH"])
        self.assertIn(host_cpu_bin, ["LOW", "MEDIUM", "HIGH"])
        self.assertIn(host_temp_bin, ["COOL", "NORMAL", "WARM", "HOT"])
        self.assertIn(dc_thermal_bin, ["COOL", "MODERATE", "CRITICAL"])

        # Specifically for 55% VM CPU, 50% host CPU, 58°C temp
        self.assertEqual(vm_cpu_bin, "MEDIUM")
        self.assertEqual(host_cpu_bin, "MEDIUM")
        self.assertEqual(host_temp_bin, "NORMAL")
        self.assertEqual(dc_thermal_bin, "COOL")

    # -------------------------------------------------------------------------
    # 2. Changing CPU usage [%] Changes CPU State
    # -------------------------------------------------------------------------

    def test_2_changing_cpu_usage_percent_changes_cpu_state(self):
        """2. Verifies that varying CPU usage [%] across thresholds transitions the VM_CPU bin."""
        host = self.dc.hosts[0]

        # Low CPU utilization (<= 33.33%)
        vm_low = VM(vm_id=101, cpu_demand=4, ram_demand=16, arrival_time=0, cpu_trace=[15.0])
        state_low = self.agent_b1.discretize_state(vm_low, host, self.dc)
        self.assertEqual(state_low[0], "LOW")

        # Medium CPU utilization (33.33% < u <= 66.67%)
        vm_med = VM(vm_id=102, cpu_demand=4, ram_demand=16, arrival_time=0, cpu_trace=[50.0])
        state_med = self.agent_b1.discretize_state(vm_med, host, self.dc)
        self.assertEqual(state_med[0], "MEDIUM")

        # High CPU utilization (> 66.67%)
        vm_high = VM(vm_id=103, cpu_demand=4, ram_demand=16, arrival_time=0, cpu_trace=[88.0])
        state_high = self.agent_b1.discretize_state(vm_high, host, self.dc)
        self.assertEqual(state_high[0], "HIGH")

        # Verify distinct states are observed
        self.assertNotEqual(state_low[0], state_med[0])
        self.assertNotEqual(state_med[0], state_high[0])

    # -------------------------------------------------------------------------
    # 3. Changing VM Memory Usage [KB] Does Not Change Agent State
    # -------------------------------------------------------------------------

    def test_3_changing_vm_memory_usage_does_not_change_agent_state(self):
        """3. Verifies that changing VM memory usage [KB] or RAM allocation leaves B1 state unchanged."""
        host = self.dc.hosts[0]

        vm_base = VM(
            vm_id=201,
            cpu_demand=4,
            ram_demand=8,
            arrival_time=0,
            cpu_trace=[45.0],
            memory_usage_kb=1048576.0,  # 1 GB
        )
        vm_huge_mem = VM(
            vm_id=202,
            cpu_demand=4,
            ram_demand=48,  # very different RAM demand
            arrival_time=0,
            cpu_trace=[45.0],
            memory_usage_kb=50331648.0,  # 48 GB
        )

        state_base = self.agent_b1.discretize_state(vm_base, host, self.dc)
        state_huge_mem = self.agent_b1.discretize_state(vm_huge_mem, host, self.dc)

        self.assertEqual(
            state_base,
            state_huge_mem,
            f"B1 state changed when altering memory: {state_base} vs {state_huge_mem}",
        )

    # -------------------------------------------------------------------------
    # 4. Changing Disk or Network Measurements Does Not Change Agent State
    # -------------------------------------------------------------------------

    def test_4_changing_disk_or_network_measurements_does_not_change_agent_state(self):
        """4. Verifies disk read/write and network rx/tx variations do not alter B1 state."""
        host = self.dc.hosts[0]

        vm_idle_io = VM(
            vm_id=301,
            cpu_demand=4,
            ram_demand=16,
            arrival_time=0,
            cpu_trace=[40.0],
            disk_read_kbps=0.0,
            disk_write_kbps=0.0,
            network_rx_kbps=0.0,
            network_tx_kbps=0.0,
        )
        vm_heavy_io = VM(
            vm_id=302,
            cpu_demand=4,
            ram_demand=16,
            arrival_time=0,
            cpu_trace=[40.0],
            disk_read_kbps=500000.0,
            disk_write_kbps=350000.0,
            network_rx_kbps=1000000.0,
            network_tx_kbps=800000.0,
        )

        state_idle = self.agent_b1.discretize_state(vm_idle_io, host, self.dc)
        state_heavy = self.agent_b1.discretize_state(vm_heavy_io, host, self.dc)

        self.assertEqual(
            state_idle,
            state_heavy,
            f"B1 state changed when altering disk/network I/O: {state_idle} vs {state_heavy}",
        )

    # -------------------------------------------------------------------------
    # 5. CPU Capacity Constraints Work
    # -------------------------------------------------------------------------

    def test_5_cpu_capacity_constraints_work(self):
        """5. Verifies physical CPU capacity constraint prevents placement onto oversubscribed host."""
        host = self.dc.hosts[0]
        # Host limit: 16 CPU cores. Fill with 14 cores used.
        host.used_cpu = 14
        host.used_ram = 10

        vm_fits = VM(vm_id=401, cpu_demand=2, ram_demand=4, arrival_time=0)
        vm_exceeds = VM(vm_id=402, cpu_demand=4, ram_demand=4, arrival_time=0)

        # 14 + 2 <= 16 -> fits
        self.assertTrue(host.can_accommodate(vm_fits))
        # 14 + 4 = 18 > 16 -> exceeds CPU limit
        self.assertFalse(host.can_accommodate(vm_exceeds))

        # Check agent filtering
        eligible_for_exceeds = self.agent_b1.get_eligible_hosts(vm_exceeds, self.dc)
        self.assertNotIn(host, eligible_for_exceeds)

        # Allocation must fail
        allocated = host.allocate(vm_exceeds)
        self.assertFalse(allocated)

    # -------------------------------------------------------------------------
    # 6. RAM Physical Feasibility Checks Remain Functional
    # -------------------------------------------------------------------------

    def test_6_ram_physical_feasibility_checks_remain_functional(self):
        """6. Verifies physical RAM constraints are enforced to prevent invalid placements in B1."""
        host = self.dc.hosts[0]
        # Host limit: 16 CPU cores, 64 GB RAM.
        # CPU is free (0/16 used), but RAM is nearly full (60/64 used).
        host.used_cpu = 0
        host.used_ram = 60

        vm_oversized_ram = VM(vm_id=501, cpu_demand=2, ram_demand=8, arrival_time=0)

        # CPU has room (2 <= 16), but RAM exceeds (60 + 8 = 68 > 64)
        self.assertFalse(host.can_accommodate(vm_oversized_ram))

        # Agent must filter out host despite CPU availability
        eligible = self.agent_b1.get_eligible_hosts(vm_oversized_ram, self.dc)
        self.assertNotIn(host, eligible)

        # Host.allocate must reject
        allocated = host.allocate(vm_oversized_ram)
        self.assertFalse(allocated)

    # -------------------------------------------------------------------------
    # 7. Thermal-Aware Reward Remains Operational
    # -------------------------------------------------------------------------

    def test_7_thermal_aware_reward_remains_operational(self):
        """7. Verifies the thermal reward calculator assesses hotspot penalties and thermal imbalance."""
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
        self.assertLess(bd_hotspot.total_reward, 0.0)  # Strong penalty for hotspot

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
    # 8. Identical Seeds Produce Reproducible Results
    # -------------------------------------------------------------------------

    def test_8_identical_seeds_produce_reproducible_results(self):
        """8. Verifies identical random seeds yield identical simulation trajectories."""
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

            agent = QLearningPlacementAgent(seed=seed, state_mode="B1")
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
    # 9. First Fit and Q-Learning Execute Successfully
    # -------------------------------------------------------------------------

    def test_9_first_fit_and_qlearning_execute_successfully(self):
        """9. Verifies First Fit and Q-Learning (B1) both execute successfully on identical Bitbrains schedule."""
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

        # Run Q-Learning (B1)
        dc_ql = DataCenter.create_standard_datacenter(
            num_racks=2,
            hosts_per_rack=2,
            cpu_per_host=16,
            ram_per_host=64,
            initial_temperature_c=25.0,
        )
        agent_ql = QLearningPlacementAgent(seed=42, state_mode="B1")
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
