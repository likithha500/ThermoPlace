"""Integration and unit tests for Phase 4: Q-Learning Agent Training.

Tests cover:
1.  Multiple training episodes execute successfully.
2.  Datacenter resets between episodes.
3.  Q-table persists between episodes (not cleared).
4.  Epsilon decays correctly after each episode.
5.  Epsilon never falls below its configured minimum.
6.  Workload traces are replayed consistently across episodes.
7.  Training statistics are recorded correctly (CSV generation and schema).
8.  Q-table can be saved to and loaded from JSON.
9.  Loaded Q-table produces consistent greedy decisions.
10. Evaluation mode does not modify Q-values or decay epsilon.
11. Repeated training with the same seed is reproducible under deterministic settings.
12. No resource allocations or thermal states leak between episodes.
13. Incompatible state configuration thresholds raise ValueError on load.
14. Non-overwriting behavior creates timestamped backup files.
15. Matplotlib progress plots generate all required charts and dashboard.

All tests utilize small deterministic synthetic workloads for fast, reliable execution.
"""

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from src.datacenter import DataCenter, Host
from src.qlearning.agent import QLearningPlacementAgent, StateDiscretizerConfig
from src.qlearning.reward import RewardConfig
from src.qlearning.strategy import QLearningSimulatorStrategy
from src.qlearning.trainer import EpisodeRecord, QLearningTrainer, TrainingConfig
from src.simulator import Simulator
from src.thermal import ThermalConfig, ThermalModel
from src.workload import VM, WorkloadGenerator


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def _small_dc(initial_temp: float = 25.0) -> DataCenter:
    """Creates a compact 1-rack x 2-host datacenter for unit tests."""
    return DataCenter.create_standard_datacenter(
        num_racks=1,
        hosts_per_rack=2,
        cpu_per_host=16,
        ram_per_host=32,
        initial_temperature_c=initial_temp,
    )


def _small_config(episodes: int = 5, steps: int = 10, seed: int = 42, tmp_dir: str = "") -> TrainingConfig:
    """Creates a fast, small TrainingConfig."""
    return TrainingConfig(
        episodes=episodes,
        learning_rate=0.1,
        discount_factor=0.95,
        initial_epsilon=1.0,
        min_epsilon=0.05,
        epsilon_decay=0.9,
        seed=seed,
        steps=steps,
        racks=1,
        hosts_per_rack=2,
        cpu_per_host=16,
        ram_per_host=32,
        arrival_rate=1.5,
        ambient_temp=25.0,
        hotspot_threshold=75.0,
        output_dir=tmp_dir or tempfile.gettempdir(),
        plots_dir=str(Path(tmp_dir or tempfile.gettempdir()) / "plots"),
        model_path=str(Path(tmp_dir or tempfile.gettempdir()) / "q_table.json"),
        overwrite=True,
        save_plots=False,
        save_model=False,
    )


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

class TestPhase4Training(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.config = _small_config(episodes=5, steps=10, seed=42, tmp_dir=self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    # -----------------------------------------------------------------------
    # Requirement 1: Multiple training episodes execute successfully
    # -----------------------------------------------------------------------
    def test_multiple_training_episodes_execute_successfully(self):
        """Multi-episode training loop runs from episode 1 to N without error."""
        trainer = QLearningTrainer(config=self.config)
        history = trainer.train(episodes=5, verbose=False)

        self.assertEqual(len(history), 5)
        self.assertEqual(history[0].episode, 1)
        self.assertEqual(history[-1].episode, 5)
        for record in history:
            self.assertIsInstance(record, EpisodeRecord)
            self.assertGreaterEqual(record.successful_placements, 0)
            self.assertGreaterEqual(record.rejected_vms, 0)

    # -----------------------------------------------------------------------
    # Requirement 2: Datacenter resets between episodes
    # -----------------------------------------------------------------------
    def test_datacenter_resets_between_episodes(self):
        """Datacenter hosts have zero allocation and ambient temperature at each episode start."""
        trainer = QLearningTrainer(config=self.config)

        # Run 1 episode
        trainer.run_episode(1)
        dc = trainer.datacenter

        # Simulate dirtying datacenter state
        dc.hosts[0].used_cpu = 10
        dc.hosts[0].used_ram = 20
        dc.hosts[0].simulated_temperature_c = 68.0

        # Now reset simulator for episode 2 as done in run_episode
        trainer.simulator.reset(initial_temp=self.config.ambient_temp)

        for host in dc.hosts:
            self.assertEqual(host.used_cpu, 0, f"Host {host.host_id} CPU not reset")
            self.assertEqual(host.used_ram, 0, f"Host {host.host_id} RAM not reset")
            self.assertEqual(len(host.allocated_vms), 0, f"Host {host.host_id} has lingering VMs")
            self.assertAlmostEqual(host.simulated_temperature_c, 25.0, places=2)

    # -----------------------------------------------------------------------
    # Requirement 3: Q-table persists between episodes
    # -----------------------------------------------------------------------
    def test_q_table_persists_between_episodes(self):
        """Learned Q-table is NOT cleared between episodes; it accumulates knowledge."""
        trainer = QLearningTrainer(config=self.config)

        trainer.run_episode(1)
        size_ep1 = len(trainer.agent.q_table)

        trainer.run_episode(2)
        size_ep2 = len(trainer.agent.q_table)

        # Q-table entries should persist and grow or remain non-zero
        self.assertGreater(size_ep1, 0, "Q-table should have learned entries in episode 1")
        self.assertGreaterEqual(size_ep2, size_ep1, "Q-table entries should persist across episodes")

    # -----------------------------------------------------------------------
    # Requirement 4: Epsilon decays correctly
    # -----------------------------------------------------------------------
    def test_epsilon_decays_correctly(self):
        """Epsilon decays by epsilon_decay factor at each episode boundary."""
        decay_rate = 0.9
        cfg = _small_config(episodes=3, seed=123, tmp_dir=self.tmp_dir.name)
        cfg.epsilon_decay = decay_rate
        cfg.initial_epsilon = 1.0
        cfg.min_epsilon = 0.01

        trainer = QLearningTrainer(config=cfg)

        self.assertAlmostEqual(trainer.agent.epsilon, 1.0)
        trainer.run_episode(1)
        self.assertAlmostEqual(trainer.agent.epsilon, 1.0 * decay_rate, places=4)
        trainer.run_episode(2)
        self.assertAlmostEqual(trainer.agent.epsilon, 1.0 * (decay_rate ** 2), places=4)
        trainer.run_episode(3)
        self.assertAlmostEqual(trainer.agent.epsilon, 1.0 * (decay_rate ** 3), places=4)

    # -----------------------------------------------------------------------
    # Requirement 5: Epsilon never falls below its minimum
    # -----------------------------------------------------------------------
    def test_epsilon_never_below_minimum(self):
        """Epsilon decay is strictly bounded below by min_epsilon."""
        cfg = _small_config(episodes=10, seed=123, tmp_dir=self.tmp_dir.name)
        cfg.initial_epsilon = 0.1
        cfg.epsilon_decay = 0.5  # aggressive decay
        cfg.min_epsilon = 0.05

        trainer = QLearningTrainer(config=cfg)
        trainer.train(episodes=10, verbose=False)

        self.assertGreaterEqual(trainer.agent.epsilon, cfg.min_epsilon)
        self.assertAlmostEqual(trainer.agent.epsilon, cfg.min_epsilon, places=5)

    # -----------------------------------------------------------------------
    # Requirement 6: Workload traces are replayed consistently
    # -----------------------------------------------------------------------
    def test_workload_traces_replayed_consistently(self):
        """Workload schedule VMs have identical arrival times and reset durations each episode."""
        trainer = QLearningTrainer(config=self.config)

        # Get schedule for episode 1
        sched1 = trainer.get_episode_schedule()
        vms_ep1_meta = [
            (vm.vm_id, vm.arrival_time, vm.cpu_demand, vm.ram_demand, vm.duration, vm.remaining_duration)
            for vms in sched1.values() for vm in vms
        ]

        # Run episode 1 (VMs will be mutated during simulation)
        trainer.simulator.run(sched1, strategy=trainer.strategy, total_steps=self.config.steps)

        # Get schedule for episode 2
        sched2 = trainer.get_episode_schedule()
        vms_ep2_meta = [
            (vm.vm_id, vm.arrival_time, vm.cpu_demand, vm.ram_demand, vm.duration, vm.remaining_duration)
            for vms in sched2.values() for vm in vms
        ]

        self.assertEqual(len(vms_ep1_meta), len(vms_ep2_meta))
        self.assertEqual(vms_ep1_meta, vms_ep2_meta, "Workload VM schedule must be identical across episodes")

        # Also verify that all VMs in sched2 have remaining_duration == duration (not exhausted)
        for vms in sched2.values():
            for vm in vms:
                self.assertEqual(vm.remaining_duration, vm.duration)
                self.assertEqual(vm.current_sample_index, 0)

    # -----------------------------------------------------------------------
    # Requirement 7: Training statistics are recorded correctly
    # -----------------------------------------------------------------------
    def test_training_statistics_recorded_and_saved_to_csv(self):
        """Training statistics contain all 13 required fields and are written to CSV."""
        csv_path = Path(self.tmp_dir.name) / "test_history.csv"
        cfg = _small_config(episodes=3, seed=42, tmp_dir=self.tmp_dir.name)
        cfg.csv_filename = "test_history.csv"

        trainer = QLearningTrainer(config=cfg)
        trainer.train(episodes=3, verbose=False)
        saved_csv = trainer.save_history_csv(output_path=csv_path, overwrite=True)

        self.assertTrue(saved_csv.exists())

        required_fields = [
            "episode",
            "cumulative_reward",
            "avg_reward_per_decision",
            "successful_placements",
            "rejected_vms",
            "total_hotspots",
            "max_host_temperature",
            "avg_host_temperature",
            "thermal_imbalance",
            "avg_cpu_utilization",
            "avg_ram_utilization",
            "epsilon",
            "q_table_entries",
        ]

        with open(saved_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, required_fields)
            rows = list(reader)
            self.assertEqual(len(rows), 3)

            for i, row in enumerate(rows, start=1):
                self.assertEqual(int(row["episode"]), i)
                self.assertIsInstance(float(row["cumulative_reward"]), float)
                self.assertIsInstance(float(row["avg_reward_per_decision"]), float)
                self.assertGreaterEqual(int(row["successful_placements"]), 0)
                self.assertGreaterEqual(int(row["rejected_vms"]), 0)
                self.assertGreaterEqual(int(row["total_hotspots"]), 0)
                self.assertGreaterEqual(float(row["max_host_temperature"]), 25.0)
                self.assertGreaterEqual(float(row["avg_host_temperature"]), 25.0)
                self.assertGreaterEqual(float(row["thermal_imbalance"]), 0.0)
                self.assertGreaterEqual(float(row["epsilon"]), 0.0)
                self.assertGreater(int(row["q_table_entries"]), 0)

    # -----------------------------------------------------------------------
    # Requirement 8: Q-table can be saved and loaded
    # -----------------------------------------------------------------------
    def test_q_table_saving_and_loading(self):
        """Q-table saves to JSON with metadata and loads back with exact state-action values."""
        agent = QLearningPlacementAgent(learning_rate=0.1, discount_factor=0.95, seed=42)
        state_a = ("SMALL", "SMALL", "LOW", "LOW", "COOL", "COOL")
        state_b = ("LARGE", "LARGE", "HIGH", "HIGH", "HOT", "CRITICAL")

        agent.set_q_value(state_a, "R0_H0", 12.345)
        agent.set_q_value(state_b, "R0_H1", -8.765)

        model_path = Path(self.tmp_dir.name) / "test_model.json"
        agent.save(model_path, metadata={"test_run": "unit_test_v1"})

        self.assertTrue(model_path.exists())

        # Verify JSON format is valid and safe
        with open(model_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertIn("version", data)
            self.assertIn("hyperparameters", data)
            self.assertIn("state_config", data)
            self.assertIn("entries", data)
            self.assertEqual(data["training_metadata"]["test_run"], "unit_test_v1")

        # Load into a clean agent
        new_agent = QLearningPlacementAgent()
        new_agent.load(model_path, validate_config=True)

        self.assertAlmostEqual(new_agent.get_q_value(state_a, "R0_H0"), 12.345, places=3)
        self.assertAlmostEqual(new_agent.get_q_value(state_b, "R0_H1"), -8.765, places=3)
        self.assertEqual(new_agent.get_q_value(state_a, "unseen_action"), 0.0)

        # Test classmethod load_from_file
        loaded_agent = QLearningPlacementAgent.load_from_file(model_path)
        self.assertAlmostEqual(loaded_agent.get_q_value(state_a, "R0_H0"), 12.345, places=3)
        self.assertAlmostEqual(loaded_agent.get_q_value(state_b, "R0_H1"), -8.765, places=3)

    # -----------------------------------------------------------------------
    # Requirement 9: Loaded Q-table produces consistent greedy decisions
    # -----------------------------------------------------------------------
    def test_loaded_q_table_produces_consistent_greedy_decisions(self):
        """Loaded agent with epsilon=0 selects the exact same host as source agent."""
        dc = _small_dc()
        vm = VM(vm_id=1, cpu_demand=2, ram_demand=4, arrival_time=0)

        source_agent = QLearningPlacementAgent(epsilon=0.0, seed=42)
        state_h0 = source_agent.discretize_state(vm, dc.hosts[0], dc)
        state_h1 = source_agent.discretize_state(vm, dc.hosts[1], dc)

        # Explicitly favor Host 1
        source_agent.set_q_value(state_h0, "R0_H0", 1.0)
        source_agent.set_q_value(state_h1, "R0_H1", 20.0)

        model_path = Path(self.tmp_dir.name) / "greedy_test.json"
        source_agent.save(model_path)

        # Load into another agent
        eval_agent = QLearningPlacementAgent.load_from_file(model_path)
        eval_agent.epsilon = 0.0  # pure exploitation

        choice_source = source_agent.select_host(vm, dc)
        choice_loaded = eval_agent.select_host(vm, dc)

        self.assertEqual(choice_source.host_id, "R0_H1")
        self.assertEqual(choice_loaded.host_id, "R0_H1")

    # -----------------------------------------------------------------------
    # Requirement 10: Evaluation mode does not modify Q-values
    # -----------------------------------------------------------------------
    def test_evaluation_mode_does_not_modify_q_values(self):
        """In evaluation mode, Q-table remains completely frozen and epsilon does not decay."""
        trainer = QLearningTrainer(config=self.config)
        trainer.train(episodes=3, verbose=False)

        q_table_snapshot = copy.deepcopy(dict(trainer.agent.q_table))
        eps_before = trainer.agent.epsilon

        # Run pure evaluation
        metrics = trainer.evaluate()

        # Verify Q-table is unmodified
        self.assertEqual(dict(trainer.agent.q_table), q_table_snapshot)
        # Verify epsilon is restored
        self.assertAlmostEqual(trainer.agent.epsilon, eps_before)
        # Verify evaluation actually ran and collected metrics
        self.assertGreater(metrics.total_placed_vms, 0)

    # -----------------------------------------------------------------------
    # Requirement 11: Reproducibility with same seed
    # -----------------------------------------------------------------------
    def test_reproducibility_with_same_seed(self):
        """Two identical training runs with the same seed yield identical history and Q-values."""
        cfg1 = _small_config(episodes=3, seed=999, tmp_dir=self.tmp_dir.name)
        trainer1 = QLearningTrainer(config=cfg1)
        history1 = trainer1.train(episodes=3, verbose=False)

        cfg2 = _small_config(episodes=3, seed=999, tmp_dir=self.tmp_dir.name)
        trainer2 = QLearningTrainer(config=cfg2)
        history2 = trainer2.train(episodes=3, verbose=False)

        for r1, r2 in zip(history1, history2):
            self.assertAlmostEqual(r1.cumulative_reward, r2.cumulative_reward, places=3)
            self.assertEqual(r1.successful_placements, r2.successful_placements)
            self.assertEqual(r1.rejected_vms, r2.rejected_vms)
            self.assertAlmostEqual(r1.epsilon, r2.epsilon, places=4)

        self.assertEqual(dict(trainer1.agent.q_table), dict(trainer2.agent.q_table))

    # -----------------------------------------------------------------------
    # Requirement 12: No resource allocations or thermal states leak
    # -----------------------------------------------------------------------
    def test_no_resource_or_thermal_leakage_between_episodes(self):
        """Residual allocations and elevated temperatures do not leak into next episode."""
        trainer = QLearningTrainer(config=self.config)

        # Run episode 1
        trainer.run_episode(1)

        # Manually verify that simulator.reset wipes datacenter hosts completely
        trainer.simulator.reset(initial_temp=self.config.ambient_temp)

        total_active_vms = sum(len(h.allocated_vms) for h in trainer.datacenter.hosts)
        total_used_cpu = sum(h.used_cpu for h in trainer.datacenter.hosts)
        total_used_ram = sum(h.used_ram for h in trainer.datacenter.hosts)
        temps = [h.simulated_temperature_c for h in trainer.datacenter.hosts]

        self.assertEqual(total_active_vms, 0)
        self.assertEqual(total_used_cpu, 0)
        self.assertEqual(total_used_ram, 0)
        self.assertTrue(all(abs(t - self.config.ambient_temp) < 1e-4 for t in temps))

    # -----------------------------------------------------------------------
    # Requirement 13: Incompatible state config validation
    # -----------------------------------------------------------------------
    def test_incompatible_state_config_raises_value_error(self):
        """Loading a Q-table with incompatible state discretization thresholds raises ValueError."""
        model_path = Path(self.tmp_dir.name) / "incompat_model.json"
        agent = QLearningPlacementAgent()
        agent.save(model_path)

        # Alter saved file's hotspot threshold in metadata
        with open(model_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["state_config"]["hotspot_threshold_c"] = 99.9  # incompatible
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        # Loading into agent with default 75.0°C should raise ValueError
        new_agent = QLearningPlacementAgent()
        with self.assertRaises(ValueError) as ctx:
            new_agent.load(model_path, validate_config=True)
        self.assertIn("hotspot_threshold_c", str(ctx.exception))

    # -----------------------------------------------------------------------
    # Requirement 14: Non-overwriting behavior preserves previous CSV files
    # -----------------------------------------------------------------------
    def test_non_overwriting_behavior_creates_timestamped_file(self):
        """When overwrite=False and target CSV exists, trainer creates timestamped alternate."""
        cfg = _small_config(episodes=2, tmp_dir=self.tmp_dir.name)
        cfg.overwrite = False
        target_csv = Path(self.tmp_dir.name) / "history.csv"

        # Pre-create the file
        target_csv.write_text("existing content", encoding="utf-8")

        trainer = QLearningTrainer(config=cfg)
        trainer.train(episodes=2, verbose=False)
        saved_path = trainer.save_history_csv(output_path=target_csv)

        self.assertNotEqual(str(saved_path), str(target_csv))
        self.assertTrue(saved_path.exists())
        self.assertEqual(target_csv.read_text(encoding="utf-8"), "existing content")

    # -----------------------------------------------------------------------
    # Requirement 15: Matplotlib progress plots generate all required charts
    # -----------------------------------------------------------------------
    def test_progress_plots_generation(self):
        """generate_plots creates all 7 individual plots and the combined dashboard."""
        cfg = _small_config(episodes=5, tmp_dir=self.tmp_dir.name)
        plots_dir = Path(self.tmp_dir.name) / "plots"
        cfg.plots_dir = str(plots_dir)

        trainer = QLearningTrainer(config=cfg)
        trainer.train(episodes=5, verbose=False)
        saved_plots = trainer.generate_plots(output_dir=plots_dir)

        self.assertEqual(len(saved_plots), 8)  # 7 required + 1 combined dashboard

        expected_filenames = [
            "cumulative_reward.png",
            "average_reward.png",
            "epsilon_decay.png",
            "hotspot_count.png",
            "max_host_temperature.png",
            "thermal_imbalance.png",
            "successful_placements.png",
            "training_dashboard.png",
        ]

        saved_names = [p.name for p in saved_plots]
        for name in expected_filenames:
            self.assertIn(name, saved_names, f"Missing plot: {name}")
            p = plots_dir / name
            self.assertTrue(p.exists(), f"File does not exist: {p}")
            self.assertGreater(p.stat().st_size, 1000, f"Plot file seems empty: {p}")


if __name__ == "__main__":
    unittest.main()

