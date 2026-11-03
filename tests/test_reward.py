"""Unit tests for the thermal-aware reward function in ThermoPlace."""

import unittest
from src.datacenter import DataCenter, Host
from src.qlearning.reward import ThermalRewardCalculator, RewardConfig, RewardBreakdown
from src.workload import VM


class TestThermalReward(unittest.TestCase):
    def setUp(self):
        self.config = RewardConfig(
            w_success=10.0,
            w_util=1.0,
            w_temp=4.0,
            w_new_hotspot=15.0,
            w_hotspot_active=5.0,
            w_imbalance=2.0,
            w_reject=10.0,
            ambient_temp_c=25.0,
            hotspot_threshold_c=75.0,
            imbalance_scale_c=5.0,
        )
        self.calculator = ThermalRewardCalculator(config=self.config)

    # -------------------------------------------------------------------------
    # 1. Success & Positive Base Reward
    # -------------------------------------------------------------------------

    def test_successful_placement_receives_positive_base_reward(self):
        """Successful placement on a cool host receives a positive net reward dominated by w_success."""
        breakdown = self.calculator.compute_reward(
            placement_success=True,
            host_temp_c=25.0,  # Ambient temperature -> P_temp = 0
            host_util=0.5,     # 50% CPU util
            is_new_hotspot=False,
            is_existing_hotspot=False,
            imbalance_before=1.0,
            imbalance_after=1.0,
        )

        self.assertTrue(breakdown.is_successful)
        self.assertGreater(breakdown.total_reward, 0.0)
        # Expected: R_success = 10.0 + 1.0 * 0.5 = 10.5
        # P_temp = 0.0, P_hotspot = 0.0, P_imbalance = 0.0
        self.assertAlmostEqual(breakdown.success_reward, 10.5)
        self.assertAlmostEqual(breakdown.temperature_penalty, 0.0)
        self.assertAlmostEqual(breakdown.total_reward, 10.5)

    # -------------------------------------------------------------------------
    # 2. Temperature Gradient Penalty
    # -------------------------------------------------------------------------

    def test_hotter_hosts_receive_lower_rewards_under_comparable_conditions(self):
        """Placing on hotter hosts increases temperature penalty and reduces overall reward."""
        # Cool host (35°C): elevation = 10°C / 50°C = 0.20 -> P_temp = 4.0 * 0.20 = 0.8
        cool_breakdown = self.calculator.compute_reward(
            placement_success=True,
            host_temp_c=35.0,
            host_util=0.5,
            is_new_hotspot=False,
            is_existing_hotspot=False,
            imbalance_before=1.0,
            imbalance_after=1.0,
        )

        # Warm host (65°C): elevation = 40°C / 50°C = 0.80 -> P_temp = 4.0 * 0.80 = 3.2
        warm_breakdown = self.calculator.compute_reward(
            placement_success=True,
            host_temp_c=65.0,
            host_util=0.5,
            is_new_hotspot=False,
            is_existing_hotspot=False,
            imbalance_before=1.0,
            imbalance_after=1.0,
        )

        self.assertGreater(warm_breakdown.temperature_penalty, cool_breakdown.temperature_penalty)
        self.assertGreater(cool_breakdown.total_reward, warm_breakdown.total_reward)
        self.assertAlmostEqual(cool_breakdown.temperature_penalty, 0.8)
        self.assertAlmostEqual(warm_breakdown.temperature_penalty, 3.2)
        # Difference in reward is exactly 3.2 - 0.8 = 2.4
        self.assertAlmostEqual(cool_breakdown.total_reward - warm_breakdown.total_reward, 2.4)

    # -------------------------------------------------------------------------
    # 3. Hotspot Penalties
    # -------------------------------------------------------------------------

    def test_placement_creating_new_hotspot_receives_substantial_penalty(self):
        """Triggering a new hotspot applies the w_new_hotspot penalty, driving reward negative."""
        no_hotspot = self.calculator.compute_reward(
            placement_success=True,
            host_temp_c=74.0,
            host_util=0.6,
            is_new_hotspot=False,
            is_existing_hotspot=False,
        )

        new_hotspot = self.calculator.compute_reward(
            placement_success=True,
            host_temp_c=75.5,
            host_util=0.6,
            is_new_hotspot=True,
            is_existing_hotspot=False,
        )

        self.assertTrue(new_hotspot.is_new_hotspot)
        self.assertGreaterEqual(new_hotspot.hotspot_penalty, self.config.w_new_hotspot)
        # Hotspot penalty must severely reduce reward below zero
        self.assertLess(new_hotspot.total_reward, 0.0)
        self.assertLess(new_hotspot.total_reward, no_hotspot.total_reward)

    def test_existing_hotspot_penalty(self):
        """Allocating to a host already in a hotspot applies w_hotspot_active penalty."""
        breakdown = self.calculator.compute_reward(
            placement_success=True,
            host_temp_c=78.0,
            host_util=0.8,
            is_new_hotspot=False,
            is_existing_hotspot=True,
        )
        self.assertGreaterEqual(breakdown.hotspot_penalty, self.config.w_hotspot_active)

    # -------------------------------------------------------------------------
    # 4. Thermal Imbalance Penalty & Benefit
    # -------------------------------------------------------------------------

    def test_increased_thermal_imbalance_receives_penalty(self):
        """Increasing spatial standard deviation across hosts increases normalized imbalance penalty."""
        # Imbalance increases from 1.0 to 3.0 (delta = +2.0). Normalized delta = 2.0 / 5.0 = 0.40
        breakdown = self.calculator.compute_reward(
            placement_success=True,
            host_temp_c=50.0,
            host_util=0.4,
            imbalance_before=1.0,
            imbalance_after=3.0,
        )
        # P_imbalance = w_imbalance * (2.0 / 5.0) = 2.0 * 0.40 = 0.80
        self.assertAlmostEqual(breakdown.imbalance_penalty, 0.80)
        self.assertAlmostEqual(breakdown.imbalance_delta, 2.0)

    def test_improved_thermal_balance_receives_benefit(self):
        """Evening out host temperatures (reducing standard deviation) benefits the agent."""
        # Imbalance decreases from 4.0 to 2.0 (delta = -2.0). Normalized delta = -2.0 / 5.0 = -0.40
        breakdown = self.calculator.compute_reward(
            placement_success=True,
            host_temp_c=50.0,
            host_util=0.4,
            imbalance_before=4.0,
            imbalance_after=2.0,
        )
        # P_imbalance = 2.0 * (-0.40) = -0.80 (subtracting negative adds positive value!)
        self.assertAlmostEqual(breakdown.imbalance_penalty, -0.80)
        self.assertAlmostEqual(breakdown.imbalance_delta, -2.0)

        # Baseline with zero change in imbalance
        baseline = self.calculator.compute_reward(
            placement_success=True,
            host_temp_c=50.0,
            host_util=0.4,
            imbalance_before=2.0,
            imbalance_after=2.0,
        )

        # Total reward with improved balance must exceed baseline by exactly 0.80
        self.assertAlmostEqual(breakdown.total_reward - baseline.total_reward, 0.80)

    # -------------------------------------------------------------------------
    # 5. Rejection & Admission Failure
    # -------------------------------------------------------------------------

    def test_vm_rejection_receives_rejection_penalty(self):
        """When placement cannot be accommodated, returns negative rejection penalty."""
        breakdown = self.calculator.compute_reward(placement_success=False)

        self.assertFalse(breakdown.is_successful)
        self.assertEqual(breakdown.total_reward, -self.config.w_reject)
        self.assertEqual(breakdown.rejection_penalty, self.config.w_reject)
        self.assertEqual(breakdown.success_reward, 0.0)

    # -------------------------------------------------------------------------
    # 6. Edge Cases and Defensive Input Handling
    # -------------------------------------------------------------------------

    def test_edge_cases_and_invalid_inputs(self):
        """Calculator handles temperatures below ambient, above bounds, and None objects gracefully."""
        # Host temperature below ambient CRAH temperature (e.g. 20°C < 25°C) -> penalty clamped to 0
        below_ambient = self.calculator.compute_reward(
            placement_success=True,
            host_temp_c=20.0,
            host_util=0.2,
        )
        self.assertAlmostEqual(below_ambient.temperature_penalty, 0.0)

        # Negative utilization clamped to 0.0
        neg_util = self.calculator.compute_reward(
            placement_success=True,
            host_temp_c=25.0,
            host_util=-0.5,
        )
        self.assertAlmostEqual(neg_util.success_reward, self.config.w_success)

        # Extreme high utilization (> 1.0) clamped to 1.0
        over_util = self.calculator.compute_reward(
            placement_success=True,
            host_temp_c=25.0,
            host_util=1.5,
        )
        self.assertAlmostEqual(over_util.success_reward, self.config.w_success + self.config.w_util)

        # None host passed to calculate_from_states returns rejection
        state_breakdown = self.calculator.calculate_from_states(
            placement_success=False,
            host=None,
        )
        self.assertFalse(state_breakdown.is_successful)
        self.assertEqual(state_breakdown.total_reward, -self.config.w_reject)

    # -------------------------------------------------------------------------
    # 7. Exact Mathematical Decomposition
    # -------------------------------------------------------------------------

    def test_individual_reward_components_sum_correctly(self):
        """Verifies total_reward exactly equals R_success - P_temp - P_hotspot - P_imbalance."""
        breakdown = self.calculator.compute_reward(
            placement_success=True,
            host_temp_c=60.0,  # elevation = 35 / 50 = 0.70 -> P_temp = 4 * 0.7 = 2.8
            host_util=0.8,     # R_success = 10 + 1 * 0.8 = 10.8
            is_new_hotspot=False,
            is_existing_hotspot=False,
            imbalance_before=1.5,
            imbalance_after=2.0,  # delta = +0.5 / 5.0 = 0.10 -> P_imbalance = 2 * 0.1 = 0.2
        )

        expected_total = (
            breakdown.success_reward
            - breakdown.temperature_penalty
            - breakdown.hotspot_penalty
            - breakdown.imbalance_penalty
        )

        self.assertAlmostEqual(breakdown.total_reward, expected_total, places=5)
        # 10.8 - 2.8 - 0.0 - 0.2 = 7.80
        self.assertAlmostEqual(breakdown.total_reward, 7.80, places=5)

    # -------------------------------------------------------------------------
    # 8. Determinism and Repeatability
    # -------------------------------------------------------------------------

    def test_identical_inputs_produce_identical_rewards(self):
        """Evaluating identical placement conditions produces deterministic, identical outputs."""
        r1 = self.calculator.compute_reward(
            placement_success=True,
            host_temp_c=58.5,
            host_util=0.65,
            is_new_hotspot=False,
            imbalance_before=2.2,
            imbalance_after=2.1,
        )
        r2 = self.calculator.compute_reward(
            placement_success=True,
            host_temp_c=58.5,
            host_util=0.65,
            is_new_hotspot=False,
            imbalance_before=2.2,
            imbalance_after=2.1,
        )

        self.assertEqual(r1.total_reward, r2.total_reward)
        self.assertEqual(r1.to_dict(), r2.to_dict())

    # -------------------------------------------------------------------------
    # 9. Configurable Weights
    # -------------------------------------------------------------------------

    def test_configurable_reward_weights(self):
        """Customizing reward weights alters component scaling predictably."""
        custom_config = RewardConfig(
            w_success=50.0,
            w_temp=10.0,
            w_imbalance=5.0,
            w_reject=25.0,
        )
        custom_calc = ThermalRewardCalculator(config=custom_config)

        # Base success
        b1 = custom_calc.compute_reward(placement_success=True, host_temp_c=25.0, host_util=0.0)
        self.assertAlmostEqual(b1.success_reward, 50.0)

        # Rejection
        b_rej = custom_calc.compute_reward(placement_success=False)
        self.assertEqual(b_rej.total_reward, -25.0)

    # -------------------------------------------------------------------------
    # 10. DataCenter Snapshot Integration
    # -------------------------------------------------------------------------

    def test_calculate_from_states_with_datacenter(self):
        """Verifies calculate_from_states accurately extracts metrics from DataCenter and Host objects."""
        dc_before = DataCenter.create_standard_datacenter(num_racks=1, hosts_per_rack=3)
        dc_before.hosts[0].simulated_temperature_c = 30.0
        dc_before.hosts[1].simulated_temperature_c = 30.0
        dc_before.hosts[2].simulated_temperature_c = 30.0

        dc_after = DataCenter.create_standard_datacenter(num_racks=1, hosts_per_rack=3)
        dc_after.hosts[0].simulated_temperature_c = 45.0  # target host placed
        dc_after.hosts[0].used_cpu = 8                    # 8/16 = 50%
        dc_after.hosts[1].simulated_temperature_c = 30.0
        dc_after.hosts[2].simulated_temperature_c = 30.0

        breakdown = self.calculator.calculate_from_states(
            placement_success=True,
            host=dc_after.hosts[0],
            datacenter_before=dc_before,
            datacenter_after=dc_after,
        )

        self.assertTrue(breakdown.is_successful)
        self.assertEqual(breakdown.host_temperature_c, 45.0)
        self.assertGreater(breakdown.imbalance_delta, 0.0)
        self.assertFalse(breakdown.is_new_hotspot)
        self.assertGreater(breakdown.total_reward, 0.0)


if __name__ == "__main__":
    unittest.main()
