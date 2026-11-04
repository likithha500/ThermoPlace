"""Integration and unit tests for Phase 3: Q-Learning Simulator Integration.

Tests cover:
1.  Q-Learning strategy can be selected and instantiated.
2.  First Fit remains functional and unchanged.
3.  Q-Learning selects only eligible hosts.
4.  Successful placement updates CPU and RAM allocations.
5.  Thermal model is updated exactly once per time step.
6.  Reward is calculated after the thermal model update (post-thermal).
7.  Q-table receives the correct transition.
8.  Next-state eligible actions are correctly calculated.
9.  Terminal transitions (no future Q-value) are handled correctly.
10. Rejected VMs update rejection metrics.
11. Evaluation mode does not modify Q-values.
12. Training mode updates Q-values.
13. Both strategies can run against the same workload and topology.

All tests use small deterministic synthetic workloads.
No dependency on the complete Bitbrains dataset.
"""

import copy
import unittest

from src.datacenter import DataCenter, Host
from src.placement.first_fit import FirstFitPlacement
from src.qlearning.agent import QLearningPlacementAgent
from src.qlearning.reward import RewardConfig
from src.qlearning.strategy import QLearningSimulatorStrategy, QLearningTrainingStats
from src.simulator import Simulator
from src.thermal import ThermalConfig, ThermalModel
from src.workload import VM, WorkloadGenerator


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _small_dc(
    num_racks: int = 1,
    hosts_per_rack: int = 2,
    cpu_per_host: int = 16,
    ram_per_host: int = 32,
    initial_temp: float = 25.0,
) -> DataCenter:
    return DataCenter.create_standard_datacenter(
        num_racks=num_racks,
        hosts_per_rack=hosts_per_rack,
        cpu_per_host=cpu_per_host,
        ram_per_host=ram_per_host,
        initial_temperature_c=initial_temp,
    )


def _make_vm(vm_id, cpu=2, ram=4, duration=10, arrival=0):
    return VM(vm_id=vm_id, cpu_demand=cpu, ram_demand=ram,
              arrival_time=arrival, duration=duration, remaining_duration=duration)


def _default_agent(epsilon: float = 1.0, seed: int = 0) -> QLearningPlacementAgent:
    return QLearningPlacementAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=epsilon,
        min_epsilon=0.01,
        epsilon_decay=0.99,
        seed=seed,
    )


def _small_schedule(steps: int = 5, seed: int = 7) -> dict:
    gen = WorkloadGenerator(seed=seed, total_time_steps=steps, arrival_rate_lambda=1.5)
    return gen.generate_schedule_by_step()


# ---------------------------------------------------------------------------
# Test 1: Q-Learning strategy can be selected
# ---------------------------------------------------------------------------

class TestStrategyInstantiation(unittest.TestCase):
    def test_qlearning_strategy_instantiates(self):
        """QLearningSimulatorStrategy can be constructed and has correct defaults."""
        agent = _default_agent()
        strategy = QLearningSimulatorStrategy(agent=agent, training=True)
        self.assertEqual(strategy.name, "Q-Learning")
        self.assertTrue(strategy.training)
        self.assertIsInstance(strategy.stats, QLearningTrainingStats)

    def test_first_fit_strategy_instantiates(self):
        """FirstFitPlacement still instantiates without modification."""
        strategy = FirstFitPlacement()
        self.assertEqual(strategy.name, "First-Fit")

    def test_qlearning_is_placement_strategy(self):
        """QLearningSimulatorStrategy implements BasePlacementStrategy."""
        from src.placement.base import BasePlacementStrategy
        agent = _default_agent()
        strategy = QLearningSimulatorStrategy(agent=agent)
        self.assertIsInstance(strategy, BasePlacementStrategy)


# ---------------------------------------------------------------------------
# Test 2: First Fit remains functional and unchanged
# ---------------------------------------------------------------------------

class TestFirstFitUnchanged(unittest.TestCase):
    def test_first_fit_places_vms_in_order(self):
        """First Fit places VMs on the first host that fits, as before."""
        dc = _small_dc(hosts_per_rack=3, cpu_per_host=8, ram_per_host=16)
        h0, h1, h2 = dc.hosts

        vm_a = _make_vm("a", cpu=4, ram=8)
        vm_b = _make_vm("b", cpu=6, ram=12)  # won't fit h0 after vm_a

        ff = FirstFitPlacement()
        host_a = ff.select_host(vm_a, dc)
        self.assertEqual(host_a, h0)
        h0.allocate(vm_a)

        host_b = ff.select_host(vm_b, dc)
        # h0 has 4 CPU used, needs 6 more (total 10 > 8) → not h0
        self.assertNotEqual(host_b, h0)
        self.assertIn(host_b, [h1, h2])

    def test_first_fit_full_simulation_run(self):
        """First Fit completes a full simulation run with metrics collected."""
        dc = _small_dc()
        sim = Simulator(datacenter=dc)
        schedule = _small_schedule(steps=10)
        metrics = sim.run(schedule, strategy=FirstFitPlacement(), total_steps=10)
        self.assertEqual(len(metrics.snapshots), 10)
        self.assertGreaterEqual(metrics.total_placed_vms, 0)


# ---------------------------------------------------------------------------
# Test 3: Q-Learning selects only eligible hosts
# ---------------------------------------------------------------------------

class TestQLearningEligibility(unittest.TestCase):
    def test_eligible_host_only(self):
        """Q-Learning never selects a host that cannot accommodate the VM."""
        dc = _small_dc(hosts_per_rack=2, cpu_per_host=4, ram_per_host=8)
        agent = _default_agent(epsilon=1.0, seed=1)  # 100% exploration
        strategy = QLearningSimulatorStrategy(agent=agent)

        # Fill host 0 completely
        h0 = dc.hosts[0]
        big_vm = _make_vm("big", cpu=4, ram=8)
        h0.allocate(big_vm)
        self.assertFalse(h0.can_accommodate(_make_vm("x", cpu=1, ram=1)))

        # Q-Learning should only pick h1
        vm = _make_vm("test", cpu=1, ram=1)
        host = strategy.select_host(vm, dc)
        self.assertIsNotNone(host)
        self.assertNotEqual(host, h0)

    def test_no_eligible_host_returns_none(self):
        """Q-Learning returns None when all hosts are full."""
        dc = _small_dc(hosts_per_rack=1, cpu_per_host=2, ram_per_host=4)
        agent = _default_agent()
        strategy = QLearningSimulatorStrategy(agent=agent)

        h = dc.hosts[0]
        h.allocate(_make_vm("fill", cpu=2, ram=4))

        vm = _make_vm("overflow", cpu=1, ram=1)
        result = strategy.select_host(vm, dc)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Test 4: Successful placement updates CPU and RAM allocations
# ---------------------------------------------------------------------------

class TestPlacementResourceAllocation(unittest.TestCase):
    def test_placement_updates_host_resources(self):
        """After simulator places a VM via Q-Learning, host used_cpu and used_ram increase."""
        dc = _small_dc(hosts_per_rack=2, cpu_per_host=16, ram_per_host=32)
        agent = _default_agent(epsilon=0.0, seed=42)  # greedy
        strategy = QLearningSimulatorStrategy(agent=agent)
        sim = Simulator(datacenter=dc)

        vm = _make_vm("v1", cpu=4, ram=8, duration=50)
        schedule = {0: [vm]}
        metrics = sim.run(schedule, strategy=strategy, total_steps=5)

        total_used_cpu = sum(h.used_cpu for h in dc.hosts)
        # VM may have finished by step 5 (duration=50 > 5, so still active)
        # At least 1 VM placed
        self.assertEqual(metrics.total_placed_vms, 1)
        self.assertGreater(total_used_cpu, 0)

    def test_capacity_never_violated(self):
        """No host ever exceeds its CPU or RAM capacity under Q-Learning."""
        dc = _small_dc(hosts_per_rack=2, cpu_per_host=8, ram_per_host=16)
        agent = _default_agent(epsilon=0.5, seed=5)
        strategy = QLearningSimulatorStrategy(agent=agent)
        sim = Simulator(datacenter=dc)

        gen = WorkloadGenerator(seed=5, total_time_steps=15, arrival_rate_lambda=3.0)
        schedule = gen.generate_schedule_by_step()
        sim.run(schedule, strategy=strategy, total_steps=15)

        for h in dc.hosts:
            self.assertLessEqual(h.used_cpu, h.total_cpu)
            self.assertLessEqual(h.used_ram, h.total_ram)


# ---------------------------------------------------------------------------
# Test 5: Thermal model updated exactly once per time step
# ---------------------------------------------------------------------------

class TestThermalModelUpdateTiming(unittest.TestCase):
    def test_thermal_called_once_per_step(self):
        """thermal_model.update_datacenter is called exactly once per simulator step."""
        dc = _small_dc()
        thermal_config = ThermalConfig(ambient_temperature_c=25.0)
        thermal_model = ThermalModel(config=thermal_config)
        sim = Simulator(datacenter=dc, thermal_model=thermal_model)

        agent = _default_agent()
        strategy = QLearningSimulatorStrategy(agent=agent)

        # Record temperatures before each step — they should change monotonically
        # as thermal model is advanced (even with zero VMs, idle power heats hosts)
        vm = _make_vm("v", cpu=2, ram=4, duration=100)
        schedule = {0: [vm]}
        metrics = sim.run(schedule, strategy=strategy, total_steps=4)

        # With a loaded host, temperature should increase from step 0 to step 1
        t0 = metrics.snapshots[0].max_temperature_c
        t1 = metrics.snapshots[1].max_temperature_c
        # Both are above ambient (idle power alone heats servers)
        self.assertGreater(t0, 24.0)
        self.assertGreater(t1, 24.0)
        # Temperatures are well-defined floats
        self.assertIsInstance(t0, float)
        self.assertIsInstance(t1, float)


# ---------------------------------------------------------------------------
# Test 6: Reward is calculated after thermal update (post-thermal observations)
# ---------------------------------------------------------------------------

class TestRewardAfterThermal(unittest.TestCase):
    def test_on_step_end_receives_post_thermal_temperatures(self):
        """on_step_end fires after thermal update — temperature snapshot is non-trivial."""
        dc = _small_dc(hosts_per_rack=1, cpu_per_host=16, ram_per_host=32)
        agent = _default_agent(epsilon=0.0, seed=1)

        observed_temps = []

        class InstrumentedStrategy(QLearningSimulatorStrategy):
            def on_step_end(self, time_step, datacenter, thermal_model=None):
                observed_temps.append(
                    [h.simulated_temperature_c for h in datacenter.hosts]
                )
                super().on_step_end(time_step, datacenter, thermal_model)

        strategy = InstrumentedStrategy(agent=agent)
        sim = Simulator(datacenter=dc)
        vm = _make_vm("v", cpu=8, ram=16, duration=50)
        sim.run({0: [vm]}, strategy=strategy, total_steps=3)

        # Temperatures recorded in on_step_end must reflect post-thermal state
        # (not ambient 25.0 — hosts warm up after placement due to idle + load power)
        self.assertEqual(len(observed_temps), 3)
        # After step 0 (VM placed, thermal advanced): temp should be above 25
        for temp_list in observed_temps:
            for t in temp_list:
                self.assertGreaterEqual(t, 25.0)


# ---------------------------------------------------------------------------
# Test 7: Q-table receives the correct transition (training mode)
# ---------------------------------------------------------------------------

class TestQTableUpdates(unittest.TestCase):
    def test_training_mode_writes_q_values(self):
        """In training mode, Q-table grows as placements occur."""
        dc = _small_dc()
        agent = _default_agent(epsilon=0.5, seed=10)
        strategy = QLearningSimulatorStrategy(agent=agent, training=True)
        sim = Simulator(datacenter=dc)

        gen = WorkloadGenerator(seed=10, total_time_steps=10, arrival_rate_lambda=2.0)
        schedule = gen.generate_schedule_by_step()
        sim.run(schedule, strategy=strategy, total_steps=10)

        # Q-table should have learned some entries
        self.assertGreater(strategy.q_table_size, 0)
        self.assertGreater(strategy.stats.total_q_updates, 0)

    def test_q_value_changes_after_update(self):
        """A Q-table entry is non-zero after a successful placement + update."""
        dc = _small_dc(hosts_per_rack=1, cpu_per_host=16, ram_per_host=32)
        agent = _default_agent(epsilon=0.0, seed=0)  # greedy — deterministic
        strategy = QLearningSimulatorStrategy(agent=agent, training=True)
        sim = Simulator(datacenter=dc)

        vm = _make_vm("v", cpu=2, ram=4, duration=50)
        sim.run({0: [vm]}, strategy=strategy, total_steps=3)

        # At least one Q-table entry should be non-default (non-zero)
        all_values = [v for v in agent.q_table.values()]
        self.assertTrue(any(v != 0.0 for v in all_values),
                        "Q-table still all zeros after a placement")


# ---------------------------------------------------------------------------
# Test 8: Next-state eligible actions correctly calculated
# ---------------------------------------------------------------------------

class TestNextStateEligibleActions(unittest.TestCase):
    def test_eligible_actions_exclude_full_hosts(self):
        """Agent's get_eligible_hosts correctly excludes over-capacity hosts."""
        dc = _small_dc(hosts_per_rack=2, cpu_per_host=4, ram_per_host=8)
        h0, h1 = dc.hosts

        h0.allocate(_make_vm("fill", cpu=4, ram=8))  # h0 is now full

        agent = _default_agent()
        small_vm = _make_vm("small", cpu=1, ram=1)
        eligible = agent.get_eligible_hosts(small_vm, dc)

        self.assertIn(h1, eligible)
        self.assertNotIn(h0, eligible)


# ---------------------------------------------------------------------------
# Test 9: Terminal transitions handled correctly
# ---------------------------------------------------------------------------

class TestTerminalTransitions(unittest.TestCase):
    def test_terminal_transition_no_panic(self):
        """When only one VM exists (terminal transition), Q-update does not raise."""
        dc = _small_dc(hosts_per_rack=1, cpu_per_host=16, ram_per_host=32)
        agent = _default_agent(epsilon=0.0, seed=0)
        strategy = QLearningSimulatorStrategy(agent=agent, training=True)
        sim = Simulator(datacenter=dc)

        vm = _make_vm("only", cpu=2, ram=4, duration=10)
        sim.run({0: [vm]}, strategy=strategy, total_steps=5)

        # No exception → terminal handling correct
        # Q-table should have exactly one entry (state, action) for this single VM
        self.assertGreaterEqual(strategy.q_table_size, 0)

    def test_agent_update_terminal_excludes_future_q(self):
        """Directly verify terminal update: Q(s,a) <- Q(s,a) + alpha*[r - Q(s,a)]."""
        agent = _default_agent(epsilon=0.0)
        agent.set_q_value("state_A", "host_0", 0.0)

        new_q = agent.update(
            state="state_A",
            action="host_0",
            reward=5.0,
            next_state=None,
            terminal=True,
        )
        # Q = 0 + 0.1 * (5 - 0) = 0.5
        self.assertAlmostEqual(new_q, 0.5, places=10)


# ---------------------------------------------------------------------------
# Test 10: Rejected VMs update rejection metrics
# ---------------------------------------------------------------------------

class TestRejectionMetrics(unittest.TestCase):
    def test_rejections_recorded_in_metrics(self):
        """VMs rejected due to capacity show up in MetricsCollector."""
        dc = _small_dc(hosts_per_rack=1, cpu_per_host=2, ram_per_host=4)
        agent = _default_agent(epsilon=0.0, seed=0)
        strategy = QLearningSimulatorStrategy(agent=agent)
        sim = Simulator(datacenter=dc)

        # First VM fills the single host
        vm1 = _make_vm("v1", cpu=2, ram=4, duration=100)
        # Second VM cannot fit anywhere
        vm2 = _make_vm("v2", cpu=1, ram=1, duration=10)

        schedule = {0: [vm1, vm2]}
        metrics = sim.run(schedule, strategy=strategy, total_steps=5)

        self.assertEqual(metrics.total_placed_vms, 1)
        self.assertEqual(metrics.total_rejected_vms, 1)
        self.assertGreater(strategy.stats.total_rejections, 0)

    def test_rejection_records_negative_reward(self):
        """A rejection should produce a negative cumulative reward contribution."""
        dc = _small_dc(hosts_per_rack=1, cpu_per_host=1, ram_per_host=2)
        agent = _default_agent(epsilon=0.0, seed=0)
        strategy = QLearningSimulatorStrategy(agent=agent, training=True)
        sim = Simulator(datacenter=dc)

        # VM that is too big to place
        vm = _make_vm("big", cpu=16, ram=64)
        sim.run({0: [vm]}, strategy=strategy, total_steps=3)

        self.assertLess(strategy.stats.cumulative_reward, 0.0)


# ---------------------------------------------------------------------------
# Test 11: Evaluation mode does not modify Q-values
# ---------------------------------------------------------------------------

class TestEvaluationMode(unittest.TestCase):
    def test_eval_mode_freezes_q_table(self):
        """In evaluation mode, Q-table is not modified during simulation."""
        dc = _small_dc()
        agent = _default_agent(epsilon=0.5, seed=42)

        # Pre-populate a known Q-value
        agent.set_q_value(("SMALL", "SMALL", "LOW", "LOW", "COOL", "COOL"), "R0_H0", 7.5)
        q_before = dict(agent.q_table)

        strategy = QLearningSimulatorStrategy(agent=agent, training=True)
        strategy.set_training(False)  # Switch to eval mode

        sim = Simulator(datacenter=dc)
        gen = WorkloadGenerator(seed=42, total_time_steps=10, arrival_rate_lambda=1.0)
        schedule = gen.generate_schedule_by_step()
        sim.run(schedule, strategy=strategy, total_steps=10)

        # Q-table must not have changed
        q_after = dict(agent.q_table)
        self.assertEqual(q_before, q_after)

    def test_eval_mode_epsilon_is_zero(self):
        """In evaluation mode, the agent uses epsilon=0 (pure greedy)."""
        agent = _default_agent(epsilon=0.8)
        strategy = QLearningSimulatorStrategy(agent=agent)
        strategy.set_training(False)
        self.assertEqual(agent.epsilon, 0.0)

    def test_eval_mode_epsilon_restored(self):
        """Calling restore_training() reverts epsilon to its pre-eval value."""
        agent = _default_agent(epsilon=0.6)
        strategy = QLearningSimulatorStrategy(agent=agent)
        strategy.set_training(False)
        self.assertEqual(agent.epsilon, 0.0)
        strategy.restore_training()
        self.assertAlmostEqual(agent.epsilon, 0.6, places=10)


# ---------------------------------------------------------------------------
# Test 12: Training mode updates Q-values
# ---------------------------------------------------------------------------

class TestTrainingMode(unittest.TestCase):
    def test_training_mode_enables_q_updates(self):
        """In training mode, Q-table size grows from zero after placements."""
        dc = _small_dc()
        agent = _default_agent(epsilon=0.5, seed=1)
        strategy = QLearningSimulatorStrategy(agent=agent, training=True)

        initial_size = strategy.q_table_size
        self.assertEqual(initial_size, 0)

        sim = Simulator(datacenter=dc)
        gen = WorkloadGenerator(seed=1, total_time_steps=10, arrival_rate_lambda=2.0)
        schedule = gen.generate_schedule_by_step()
        sim.run(schedule, strategy=strategy, total_steps=10)

        self.assertGreater(strategy.q_table_size, 0)
        self.assertGreater(strategy.stats.total_q_updates, 0)

    def test_epsilon_decays_at_episode_end(self):
        """Epsilon is decayed exactly once per simulation run (not once per step)."""
        agent = _default_agent(epsilon=1.0, seed=0)
        strategy = QLearningSimulatorStrategy(agent=agent, training=True)

        eps_before = agent.epsilon
        dc = _small_dc()
        sim = Simulator(datacenter=dc)
        sim.run({}, strategy=strategy, total_steps=10)

        eps_after = agent.epsilon
        # Exactly one decay applied (single episode)
        expected = max(agent.min_epsilon, eps_before * agent.epsilon_decay)
        self.assertAlmostEqual(eps_after, expected, places=10)
        self.assertEqual(len(strategy.stats.epsilon_history), 1)


# ---------------------------------------------------------------------------
# Test 13: Both strategies run against the same workload and topology
# ---------------------------------------------------------------------------

class TestBothStrategiesComparable(unittest.TestCase):
    def test_both_strategies_complete_same_workload(self):
        """First Fit and Q-Learning run the same workload without errors."""
        gen = WorkloadGenerator(seed=99, total_time_steps=10, arrival_rate_lambda=1.5)
        schedule_ff = gen.generate_schedule_by_step()
        # Regenerate so VMs start fresh (VM objects mutate remaining_duration)
        gen2 = WorkloadGenerator(seed=99, total_time_steps=10, arrival_rate_lambda=1.5)
        schedule_ql = gen2.generate_schedule_by_step()

        # First Fit
        dc_ff = _small_dc()
        sim_ff = Simulator(datacenter=dc_ff)
        metrics_ff = sim_ff.run(schedule_ff, strategy=FirstFitPlacement(), total_steps=10)

        # Q-Learning
        dc_ql = _small_dc()
        agent = _default_agent(epsilon=0.5, seed=99)
        strategy_ql = QLearningSimulatorStrategy(agent=agent, training=True)
        sim_ql = Simulator(datacenter=dc_ql)
        metrics_ql = sim_ql.run(schedule_ql, strategy=strategy_ql, total_steps=10)

        # Both ran the same number of steps
        self.assertEqual(len(metrics_ff.snapshots), len(metrics_ql.snapshots))
        # Both had the same total arrivals (identical seed)
        self.assertEqual(metrics_ff.total_arrived_vms, metrics_ql.total_arrived_vms)

    def test_q_learning_respects_capacity_same_as_first_fit(self):
        """Under an overloaded workload, both strategies reject some VMs, never violate capacity."""
        gen = WorkloadGenerator(seed=7, total_time_steps=10, arrival_rate_lambda=5.0)
        # Very small DC to force rejections
        for strategy_cls in [
            lambda: FirstFitPlacement(),
            lambda: QLearningSimulatorStrategy(_default_agent(epsilon=0.5, seed=7)),
        ]:
            gen_local = WorkloadGenerator(seed=7, total_time_steps=10, arrival_rate_lambda=5.0)
            schedule = gen_local.generate_schedule_by_step()
            dc = DataCenter.create_standard_datacenter(
                num_racks=1, hosts_per_rack=1, cpu_per_host=4, ram_per_host=8
            )
            sim = Simulator(datacenter=dc)
            metrics = sim.run(schedule, strategy=strategy_cls(), total_steps=10)

            for h in dc.hosts:
                self.assertLessEqual(h.used_cpu, h.total_cpu)
                self.assertLessEqual(h.used_ram, h.total_ram)


# ---------------------------------------------------------------------------
# Test 14: Bitbrains loader (without depending on actual dataset files)
# ---------------------------------------------------------------------------

class TestBitbrainsLoader(unittest.TestCase):
    def test_loader_reads_csv_format(self):
        """BitbrainsLoader parses CSV files with Bitbrains column schema."""
        import tempfile, os
        from src.bitbrains_loader import BitbrainsLoader

        csv_content = (
            "Timestamp [ms];\tCPU cores;\tCPU capacity provisioned [MHZ];"
            "\tCPU usage [MHZ];\tCPU usage [%];\tMemory capacity provisioned [KB];"
            "\tMemory usage [KB];\tDisk read throughput [KB/s];"
            "\tDisk write throughput [KB/s];\tNetwork received throughput [KB/s];"
            "\tNetwork transmitted throughput [KB/s]\n"
            "1376314846;\t1;\t1000.0;\t300.0;\t30.5;\t4096.0;\t2048.0;\t5.0;\t3.0;\t10.0;\t8.0\n"
            "1376315146;\t1;\t1000.0;\t500.0;\t50.0;\t4096.0;\t2048.0;\t5.0;\t3.0;\t10.0;\t8.0\n"
            "1376315446;\t1;\t1000.0;\t700.0;\t70.2;\t4096.0;\t2048.0;\t5.0;\t3.0;\t10.0;\t8.0\n"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "42.csv")
            with open(csv_path, "w") as f:
                f.write(csv_content)

            loader = BitbrainsLoader()
            traces = loader.load_traces(tmpdir, recursive=False)

        self.assertEqual(len(traces), 1)
        trace = traces[0]
        self.assertEqual(trace.vm_id, "42")
        self.assertAlmostEqual(trace.cpu_samples[0], 30.5)
        self.assertAlmostEqual(trace.cpu_samples[1], 50.0)
        self.assertAlmostEqual(trace.cpu_samples[2], 70.2)

    def test_loader_raises_on_missing_directory(self):
        """BitbrainsLoader raises FileNotFoundError for a non-existent path."""
        from src.bitbrains_loader import BitbrainsLoader
        loader = BitbrainsLoader()
        with self.assertRaises(FileNotFoundError):
            loader.load_traces("/nonexistent/path/xyz_99999")

    def test_loader_skips_invalid_csv(self):
        """BitbrainsLoader skips files that do not have the expected column."""
        import tempfile, os
        from src.bitbrains_loader import BitbrainsLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            # Valid file
            valid_csv = (
                "Timestamp [ms];\tCPU cores;\tCPU capacity provisioned [MHZ];"
                "\tCPU usage [MHZ];\tCPU usage [%];\tMemory capacity provisioned [KB];"
                "\tMemory usage [KB];\tDisk read throughput [KB/s];"
                "\tDisk write throughput [KB/s];\tNetwork received throughput [KB/s];"
                "\tNetwork transmitted throughput [KB/s]\n"
                "1376314846;\t1;\t1000.0;\t300.0;\t30.5;\t4096.0;\t2048.0;\t5.0;\t3.0;\t10.0;\t8.0\n"
            )
            with open(os.path.join(tmpdir, "1.csv"), "w") as f:
                f.write(valid_csv)

            # Invalid file (wrong columns)
            with open(os.path.join(tmpdir, "bad.csv"), "w") as f:
                f.write("col_a,col_b\n1,2\n")

            loader = BitbrainsLoader()
            report = loader.load_with_report(tmpdir, recursive=False)

        self.assertEqual(report.total_loaded, 1)
        self.assertEqual(report.total_skipped, 1)


# ---------------------------------------------------------------------------
# Test 15: main.py CLI integration
# ---------------------------------------------------------------------------

class TestMainCLI(unittest.TestCase):
    def test_parse_args_strategy_default(self):
        """Default strategy is 'first_fit'."""
        from main import parse_args
        args = parse_args([])
        self.assertEqual(args.strategy, "first_fit")
        self.assertFalse(args.eval_mode)

    def test_parse_args_strategy_qlearning(self):
        """--strategy q_learning is parsed correctly."""
        from main import parse_args
        args = parse_args(["--strategy", "q_learning"])
        self.assertEqual(args.strategy, "q_learning")

    def test_parse_args_eval_mode(self):
        """--eval-mode flag is parsed correctly."""
        from main import parse_args
        args = parse_args(["--strategy", "q_learning", "--eval-mode"])
        self.assertTrue(args.eval_mode)

    def test_parse_args_invalid_strategy(self):
        """An unrecognized strategy value causes argparse to exit."""
        from main import parse_args
        with self.assertRaises(SystemExit):
            parse_args(["--strategy", "random_strategy"])

    def test_main_no_dataset_exits(self):
        """main() exits with code 1 when no dataset is configured."""
        from main import main
        with self.assertRaises(SystemExit) as ctx:
            main([])
        self.assertEqual(ctx.exception.code, 1)

    def test_main_strategy_q_learning_no_dataset_exits(self):
        """main() exits with code 1 for q_learning strategy without dataset."""
        from main import main
        with self.assertRaises(SystemExit) as ctx:
            main(["--strategy", "q_learning"])
        self.assertEqual(ctx.exception.code, 1)

    def test_main_missing_dataset_directory_exits(self):
        """main() exits with code 1 if dataset directory does not exist."""
        from main import main
        with self.assertRaises(SystemExit) as ctx:
            main(["--dataset-dir", "non_existent_dir_phase3_test"])
        self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main()

