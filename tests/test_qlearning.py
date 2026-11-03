"""Unit tests for Tabular Q-Learning placement agent."""

import unittest
from src.datacenter import DataCenter, Host
from src.qlearning.agent import QLearningPlacementAgent, StateDiscretizerConfig
from src.workload import VM


class TestQLearningAgent(unittest.TestCase):
    def setUp(self):
        self.dc = DataCenter.create_standard_datacenter(
            num_racks=1,
            hosts_per_rack=3,
            cpu_per_host=8,
            ram_per_host=16,
            initial_temperature_c=25.0,
        )
        self.agent = QLearningPlacementAgent(
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            min_epsilon=0.05,
            epsilon_decay=0.995,
            seed=42,
        )

    # -------------------------------------------------------------------------
    # 1. State Discretization Tests
    # -------------------------------------------------------------------------

    def test_state_discretization_categories(self):
        """Verifies state discretization accurately partitions continuous metrics into bins."""
        # Test VM CPU tiers: small (<=2), medium (3-4), large (>4)
        s_small_cpu = self.agent.discretize_state(vm_cpu=2, vm_ram=4, host_cpu=0.1, host_ram=0.1, host_temp=40.0)
        s_med_cpu = self.agent.discretize_state(vm_cpu=4, vm_ram=4, host_cpu=0.1, host_ram=0.1, host_temp=40.0)
        s_lrg_cpu = self.agent.discretize_state(vm_cpu=8, vm_ram=4, host_cpu=0.1, host_ram=0.1, host_temp=40.0)
        self.assertEqual(s_small_cpu[0], "SMALL")
        self.assertEqual(s_med_cpu[0], "MEDIUM")
        self.assertEqual(s_lrg_cpu[0], "LARGE")

        # Test VM RAM tiers: small (<=4), medium (5-16), large (>16)
        s_small_ram = self.agent.discretize_state(vm_cpu=2, vm_ram=4, host_cpu=0.1, host_ram=0.1, host_temp=40.0)
        s_med_ram = self.agent.discretize_state(vm_cpu=2, vm_ram=16, host_cpu=0.1, host_ram=0.1, host_temp=40.0)
        s_lrg_ram = self.agent.discretize_state(vm_cpu=2, vm_ram=32, host_cpu=0.1, host_ram=0.1, host_temp=40.0)
        self.assertEqual(s_small_ram[1], "SMALL")
        self.assertEqual(s_med_ram[1], "MEDIUM")
        self.assertEqual(s_lrg_ram[1], "LARGE")

        # Test Host CPU utilization: low (<0.35), medium (0.35-0.70), high (>=0.70)
        s_low_u = self.agent.discretize_state(vm_cpu=2, vm_ram=4, host_cpu=0.20, host_ram=0.1, host_temp=40.0)
        s_med_u = self.agent.discretize_state(vm_cpu=2, vm_ram=4, host_cpu=0.50, host_ram=0.1, host_temp=40.0)
        s_high_u = self.agent.discretize_state(vm_cpu=2, vm_ram=4, host_cpu=0.85, host_ram=0.1, host_temp=40.0)
        self.assertEqual(s_low_u[2], "LOW")
        self.assertEqual(s_med_u[2], "MEDIUM")
        self.assertEqual(s_high_u[2], "HIGH")

        # Test Host RAM utilization
        s_low_ram = self.agent.discretize_state(vm_cpu=2, vm_ram=4, host_cpu=0.1, host_ram=0.20, host_temp=40.0)
        s_med_ram = self.agent.discretize_state(vm_cpu=2, vm_ram=4, host_cpu=0.1, host_ram=0.60, host_temp=40.0)
        s_high_ram = self.agent.discretize_state(vm_cpu=2, vm_ram=4, host_cpu=0.1, host_ram=0.90, host_temp=40.0)
        self.assertEqual(s_low_ram[3], "LOW")
        self.assertEqual(s_med_ram[3], "MEDIUM")
        self.assertEqual(s_high_ram[3], "HIGH")

        # Test Host temperature: cool (<55°C), normal (55-65°C), warm (65-75°C), hot (>=75°C)
        s_cool_t = self.agent.discretize_state(vm_cpu=2, vm_ram=4, host_cpu=0.1, host_ram=0.1, host_temp=45.0)
        s_norm_t = self.agent.discretize_state(vm_cpu=2, vm_ram=4, host_cpu=0.1, host_ram=0.1, host_temp=60.0)
        s_warm_t = self.agent.discretize_state(vm_cpu=2, vm_ram=4, host_cpu=0.1, host_ram=0.1, host_temp=70.0)
        s_hot_t = self.agent.discretize_state(vm_cpu=2, vm_ram=4, host_cpu=0.1, host_ram=0.1, host_temp=76.0)
        self.assertEqual(s_cool_t[4], "COOL")
        self.assertEqual(s_norm_t[4], "NORMAL")
        self.assertEqual(s_warm_t[4], "WARM")
        self.assertEqual(s_hot_t[4], "HOT")

    def test_state_discretization_with_entities(self):
        """Verifies state discretization works seamlessly when passed VM, Host, and DataCenter objects."""
        vm = VM(vm_id=1, cpu_demand=4, ram_demand=8, arrival_time=0)
        host = self.dc.hosts[0]
        host.used_cpu = 4
        host.used_ram = 8
        host.simulated_temperature_c = 60.0

        state = self.agent.discretize_state(vm, host, self.dc)
        self.assertIsInstance(state, tuple)
        self.assertEqual(len(state), 6)
        self.assertEqual(state[0], "MEDIUM")  # 4 cores
        self.assertEqual(state[1], "MEDIUM")  # 8 GB
        self.assertEqual(state[2], "MEDIUM")  # 4/8 = 50%
        self.assertEqual(state[3], "MEDIUM")  # 8/16 = 50%
        self.assertEqual(state[4], "NORMAL")  # 60.0°C
        self.assertIn(state[5], ["COOL", "MODERATE", "CRITICAL"])

    def test_datacenter_thermal_condition_hotspot_detection(self):
        """Discretization identifies overall datacenter thermal criticality when a host reaches hotspot."""
        vm = VM(vm_id=1, cpu_demand=2, ram_demand=4, arrival_time=0)
        host0 = self.dc.hosts[0]

        # Normal datacenter state
        state_normal = self.agent.discretize_state(vm, host0, self.dc)
        self.assertNotEqual(state_normal[5], "CRITICAL")

        # Induce hotspot on another host (e.g. 78°C >= 75°C)
        self.dc.hosts[2].simulated_temperature_c = 78.0
        state_critical = self.agent.discretize_state(vm, host0, self.dc)
        self.assertEqual(state_critical[5], "CRITICAL")

    # -------------------------------------------------------------------------
    # 2. Eligible-Host Identification Tests
    # -------------------------------------------------------------------------

    def test_eligible_hosts_capacity_filtering(self):
        """Correctly excludes hosts that violate CPU or RAM capacity constraints."""
        # Host 0: 6/8 CPU used (only 2 left), 4/16 RAM used (12 left)
        self.dc.hosts[0].used_cpu = 6
        self.dc.hosts[0].used_ram = 4

        # Host 1: 2/8 CPU used (6 left), 14/16 RAM used (only 2 left)
        self.dc.hosts[1].used_cpu = 2
        self.dc.hosts[1].used_ram = 14

        # Host 2: 0/8 CPU used (8 left), 0/16 RAM used (16 left) -> fully capable
        self.dc.hosts[2].used_cpu = 0
        self.dc.hosts[2].used_ram = 0

        # VM requiring 4 CPU and 4 RAM:
        # Host 0 fails on CPU (needs 4, only 2 left)
        # Host 1 fails on RAM (needs 4, only 2 left)
        # Host 2 satisfies both
        vm = VM(vm_id=1, cpu_demand=4, ram_demand=4, arrival_time=0)
        eligible = self.agent.get_eligible_hosts(vm, self.dc)

        self.assertEqual(len(eligible), 1)
        self.assertEqual(eligible[0].host_id, "R0_H2")

    # -------------------------------------------------------------------------
    # 3. Action Selection & Ineligible Host Exclusion Tests
    # -------------------------------------------------------------------------

    def test_agent_never_selects_ineligible_host_under_exploration(self):
        """Under 100% exploration (epsilon=1.0), agent must still never pick an ineligible host."""
        self.agent.epsilon = 1.0

        # Saturate Host 0 and Host 1 completely
        self.dc.hosts[0].used_cpu = 8
        self.dc.hosts[0].used_ram = 16
        self.dc.hosts[1].used_cpu = 8
        self.dc.hosts[1].used_ram = 16

        # Only Host 2 has capacity
        self.dc.hosts[2].used_cpu = 0
        self.dc.hosts[2].used_ram = 0

        vm = VM(vm_id=1, cpu_demand=2, ram_demand=4, arrival_time=0)

        # In 50 exploratory selections, it must ALWAYS select Host 2
        for _ in range(50):
            selected = self.agent.select_host(vm, self.dc)
            self.assertIsNotNone(selected)
            self.assertEqual(selected.host_id, "R0_H2")

    def test_no_eligible_host_returns_rejection(self):
        """When all hosts are saturated or incapable, agent returns None (rejection)."""
        # Saturate all hosts
        for host in self.dc.hosts:
            host.used_cpu = 8
            host.used_ram = 16

        vm = VM(vm_id=99, cpu_demand=2, ram_demand=4, arrival_time=0)
        self.assertEqual(self.agent.get_eligible_hosts(vm, self.dc), [])

        # Both greedy and exploratory modes must return None
        self.agent.epsilon = 0.0
        self.assertIsNone(self.agent.select_host(vm, self.dc))

        self.agent.epsilon = 1.0
        self.assertIsNone(self.agent.select_host(vm, self.dc))

        # Direct select_action on empty eligible list
        self.assertIsNone(self.agent.select_action("state", []))

    def test_exploration_selects_among_eligible_hosts(self):
        """Under exploration, choices cover all eligible hosts without picking ineligible ones."""
        self.agent.epsilon = 1.0
        # Host 0 and Host 1 are eligible, Host 2 is saturated
        self.dc.hosts[2].used_cpu = 8
        self.dc.hosts[2].used_ram = 16

        vm = VM(vm_id=1, cpu_demand=2, ram_demand=4, arrival_time=0)
        counts = {"R0_H0": 0, "R0_H1": 0, "R0_H2": 0}

        for _ in range(100):
            selected = self.agent.select_host(vm, self.dc)
            counts[selected.host_id] += 1

        self.assertGreater(counts["R0_H0"], 0)
        self.assertGreater(counts["R0_H1"], 0)
        self.assertEqual(counts["R0_H2"], 0)  # Ineligible host is never selected

    def test_greedy_selection_chooses_highest_q_value(self):
        """Under exploitation (epsilon=0.0), agent selects the eligible host with highest Q-value."""
        self.agent.epsilon = 0.0
        vm = VM(vm_id=1, cpu_demand=2, ram_demand=4, arrival_time=0)

        # All 3 hosts eligible
        state_h0 = self.agent.discretize_state(vm, self.dc.hosts[0], self.dc)
        state_h1 = self.agent.discretize_state(vm, self.dc.hosts[1], self.dc)
        state_h2 = self.agent.discretize_state(vm, self.dc.hosts[2], self.dc)

        # Bias Q-value for Host 1
        self.agent.set_q_value(state_h0, "R0_H0", 1.0)
        self.agent.set_q_value(state_h1, "R0_H1", 10.0)
        self.agent.set_q_value(state_h2, "R0_H2", 3.0)

        selected = self.agent.select_host(vm, self.dc)
        self.assertEqual(selected.host_id, "R0_H1")

        # Now bias Host 2 higher
        self.agent.set_q_value(state_h2, "R0_H2", 25.0)
        selected_updated = self.agent.select_host(vm, self.dc)
        self.assertEqual(selected_updated.host_id, "R0_H2")

    def test_select_action_standalone(self):
        """select_action functions standalone with explicit state and eligible actions list."""
        self.agent.epsilon = 0.0
        state = ("SMALL", "SMALL", "LOW", "LOW", "COOL", "NORMAL")
        actions = ["H0", "H1", "H2"]

        self.agent.set_q_value(state, "H0", -1.0)
        self.agent.set_q_value(state, "H1", 5.5)
        self.agent.set_q_value(state, "H2", 2.0)

        chosen = self.agent.select_action(state, actions)
        self.assertEqual(chosen, "H1")

    # -------------------------------------------------------------------------
    # 4. Q-Learning Update & Bellman Equation Tests
    # -------------------------------------------------------------------------

    def test_q_learning_exact_benchmark_update(self):
        """Validates specific mathematical update: Q=0, alpha=0.1, gamma=0.9, r=10, max_Q'=5 -> Q=1.45."""
        agent = QLearningPlacementAgent(learning_rate=0.1, discount_factor=0.9)
        s = "state_benchmark"
        a = "host_action"
        s_next = "state_next_benchmark"

        # Current Q-value = 0 (unseen state-action pair)
        self.assertEqual(agent.get_q_value(s, a), 0.0)

        # Maximum next-state Q-value = 5
        agent.set_q_value(s_next, "action_1", 2.0)
        agent.set_q_value(s_next, "action_2", 5.0)  # max Q(s', a') = 5.0
        agent.set_q_value(s_next, "action_3", 1.0)

        # Target = 10 + 0.9 * 5 = 14.5
        # Expected new Q = 0 + 0.1 * (14.5 - 0) = 1.45
        updated_q = agent.update(
            state=s,
            action=a,
            reward=10.0,
            next_state=s_next,
            terminal=False,
            next_eligible_actions=["action_1", "action_2", "action_3"],
        )

        self.assertAlmostEqual(updated_q, 1.45, places=5)
        self.assertAlmostEqual(agent.get_q_value(s, a), 1.45, places=5)

    def test_q_learning_update_mathematical_formula(self):
        """Verifies Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]."""
        self.agent.learning_rate = 0.1
        self.agent.discount_factor = 0.95

        s = "state_current"
        a = "action_host_1"
        s_prime = "state_next"

        # Initialize Q-values for next state actions
        self.agent.set_q_value(s_prime, "action_a", 4.0)
        self.agent.set_q_value(s_prime, "action_b", 8.0)  # max Q(s', a') = 8.0
        self.agent.set_q_value(s_prime, "action_c", 2.0)

        # Initial Q(s, a) = 0.0
        self.assertEqual(self.agent.get_q_value(s, a), 0.0)

        # Transition: reward = 10.0, non-terminal
        # Expected Target = 10.0 + 0.95 * 8.0 = 10.0 + 7.6 = 17.6
        # Expected new Q = 0.0 + 0.1 * (17.6 - 0.0) = 1.76
        new_q = self.agent.update(
            state=s,
            action=a,
            reward=10.0,
            next_state=s_prime,
            terminal=False,
            next_eligible_actions=["action_a", "action_b", "action_c"],
        )

        self.assertAlmostEqual(new_q, 1.76, places=5)
        self.assertAlmostEqual(self.agent.get_q_value(s, a), 1.76, places=5)

        # Second update from current state Q = 1.76
        # reward = -2.0, next actions max = 8.0
        # Expected Target = -2.0 + 0.95 * 8.0 = 5.6
        # Expected new Q = 1.76 + 0.1 * (5.6 - 1.76) = 1.76 + 0.1 * 3.84 = 2.144
        new_q_2 = self.agent.update(
            state=s,
            action=a,
            reward=-2.0,
            next_state=s_prime,
            terminal=False,
            next_eligible_actions=["action_a", "action_b", "action_c"],
        )
        self.assertAlmostEqual(new_q_2, 2.144, places=5)
        self.assertAlmostEqual(self.agent.get_q_value(s, a), 2.144, places=5)

    def test_terminal_transition_omits_future_q(self):
        """Terminal transitions must not include the gamma * max Q(s', a') future term."""
        self.agent.learning_rate = 0.1
        self.agent.discount_factor = 0.95

        s = "state_current"
        a = "action_host_1"
        s_prime = "state_next"

        # Even if next state has high Q-values
        self.agent.set_q_value(s_prime, "action_a", 100.0)

        # Initial Q(s, a) = 5.0
        self.agent.set_q_value(s, a, 5.0)

        # Terminal update with reward = 20.0
        # Expected Target = 20.0 (no future Q term!)
        # Expected new Q = 5.0 + 0.1 * (20.0 - 5.0) = 5.0 + 1.5 = 6.5
        new_q = self.agent.update(
            state=s,
            action=a,
            reward=20.0,
            next_state=s_prime,
            terminal=True,
            next_eligible_actions=["action_a"],
        )

        self.assertAlmostEqual(new_q, 6.5, places=5)
        self.assertAlmostEqual(self.agent.get_q_value(s, a), 6.5, places=5)

        # Also verify train_step alias with done=True
        new_q_ts = self.agent.train_step(
            state=s,
            action=a,
            reward=0.0,
            next_state=s_prime,
            done=True,
        )
        # Expected new Q = 6.5 + 0.1 * (0.0 - 6.5) = 5.85
        self.assertAlmostEqual(new_q_ts, 5.85, places=5)

    # -------------------------------------------------------------------------
    # 5. Exploration Decay Tests
    # -------------------------------------------------------------------------

    def test_epsilon_decay_and_minimum_bound(self):
        """Epsilon decays multiplicatively but never drops below min_epsilon."""
        agent = QLearningPlacementAgent(
            epsilon=1.0,
            min_epsilon=0.05,
            epsilon_decay=0.5,  # rapid decay for test
        )

        # Step 1: 1.0 * 0.5 = 0.5
        e1 = agent.decay_epsilon()
        self.assertAlmostEqual(e1, 0.5)

        # Step 2: 0.5 * 0.5 = 0.25
        e2 = agent.decay_epsilon()
        self.assertAlmostEqual(e2, 0.25)

        # Step 3: 0.25 * 0.5 = 0.125
        e3 = agent.decay_epsilon()
        self.assertAlmostEqual(e3, 0.125)

        # Step 4: 0.125 * 0.5 = 0.0625
        e4 = agent.decay_epsilon()
        self.assertAlmostEqual(e4, 0.0625)

        # Step 5: 0.0625 * 0.5 = 0.03125 -> capped at min_epsilon (0.05)
        e5 = agent.decay_epsilon()
        self.assertAlmostEqual(e5, 0.05)

        # Step 6: stays at min_epsilon
        e6 = agent.decay_epsilon()
        self.assertAlmostEqual(e6, 0.05)
        self.assertEqual(agent.epsilon, 0.05)

    # -------------------------------------------------------------------------
    # 6. Reproducibility & Sparse Q-Table Tests
    # -------------------------------------------------------------------------

    def test_random_seed_reproducibility(self):
        """Agents with identical seeds generate an identical sequence of exploration choices."""
        agent1 = QLearningPlacementAgent(epsilon=0.5, seed=12345)
        agent2 = QLearningPlacementAgent(epsilon=0.5, seed=12345)
        agent3 = QLearningPlacementAgent(epsilon=0.5, seed=99999)

        vm = VM(vm_id=1, cpu_demand=2, ram_demand=4, arrival_time=0)

        choices1 = [agent1.select_host(vm, self.dc).host_id for _ in range(50)]
        choices2 = [agent2.select_host(vm, self.dc).host_id for _ in range(50)]
        choices3 = [agent3.select_host(vm, self.dc).host_id for _ in range(50)]

        # Agent 1 and Agent 2 must match exactly at every step
        self.assertEqual(choices1, choices2)
        # Agent 3 with different seed must diverge
        self.assertNotEqual(choices1, choices3)

    def test_q_table_sparse_initialization_and_reset(self):
        """Unseen pairs return 0.0 without manual pre-allocation; reset clears table."""
        s = ("SMALL", "SMALL", "LOW", "LOW", "COOL", "NORMAL")
        self.assertEqual(self.agent.get_q_value(s, "R0_H0"), 0.0)
        self.assertEqual(self.agent.q_table[(s, "R0_H0")], 0.0)

        self.agent.set_q_value(s, "R0_H0", 4.2)
        self.assertEqual(self.agent.get_q_value(s, "R0_H0"), 4.2)
        self.assertEqual(self.agent.q_table[(s, "R0_H0")], 4.2)

        self.agent.reset_q_table()
        self.assertEqual(len(self.agent.q_table), 0)
        self.assertEqual(self.agent.get_q_value(s, "R0_H0"), 0.0)


if __name__ == "__main__":
    unittest.main()

