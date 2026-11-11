"""Q-Learning simulator strategy for ThermoPlace — Phase 3 integration.

Integrates the tabular Q-Learning agent (Phase 1) and thermal reward function
(Phase 2) with the simulator's placement lifecycle hooks to deliver a complete
learning placement strategy that:

1. Records the pre-placement state for each VM at the moment of host selection.
2. Accumulates pending transitions within a time step (placement → outcome pairs).
3. After the thermal model is updated (``on_step_end``), computes rewards from
   the post-thermal datacenter state and writes Q-table updates.
4. Applies epsilon decay once per episode (simulation run), not once per step.

Training vs Evaluation Modes
-----------------------------
Training mode (``training=True``, default):
  - Epsilon-greedy exploration is active.
  - Q-table updates are written after every step.
  - Epsilon decays at the end of each episode (``on_simulation_end``).

Evaluation mode (``training=False``):
  - Epsilon is forced to 0.0 (pure greedy selection).
  - Q-table is read-only — no updates are written.
  - Existing learned Q-values are preserved exactly.

Usage
-----
Training::

    agent = QLearningPlacementAgent(seed=42)
    strategy = QLearningSimulatorStrategy(agent=agent, training=True)
    sim.run(schedule, strategy=strategy)

Evaluation::

    strategy.set_training(False)
    sim.run(eval_schedule, strategy=strategy)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.datacenter import DataCenter, Host
from src.placement.base import BasePlacementStrategy
from src.qlearning.agent import QLearningPlacementAgent
from src.qlearning.reward import RewardConfig, ThermalRewardCalculator
from src.workload import VM

logger = logging.getLogger(__name__)


@dataclass
class _PendingTransition:
    """Stores all data needed to compute a Q-table update after the thermal step."""
    vm: VM
    host: Optional[Host]
    placed: bool
    state: Tuple  # discretized state at the moment of host selection
    action: Optional[Any]  # host_id selected (None if rejected)
    imbalance_before: float  # datacenter std(T) before placement
    hotspots_before: int  # count of hotspot hosts before placement


@dataclass
class QLearningTrainingStats:
    """Accumulates training statistics across episodes."""
    total_placements: int = 0
    total_rejections: int = 0
    total_q_updates: int = 0
    cumulative_reward: float = 0.0
    episode_count: int = 0
    epsilon_history: List[float] = field(default_factory=list)
    episode_rewards: List[float] = field(default_factory=list)


class QLearningSimulatorStrategy(BasePlacementStrategy):
    """Placement strategy that learns via tabular Q-Learning with thermal rewards.

    This class implements ``BasePlacementStrategy`` and uses the lifecycle hooks
    provided by ``Simulator`` to orchestrate the correct RL update sequence:

        [step_begin]
            → For each arriving VM:
                1. Record imbalance_before and hotspots_before.
                2. Select host (epsilon-greedy via agent).
                3. Allocate VM.
                4. Record pending transition.
            → [step_end] (after thermal model update)
                5. Compute reward from post-thermal observations.
                6. Build next state for next-VM lookahead.
                7. Write Q-table update (training mode only).
            → Repeat.
        [simulation_end]
            → Decay epsilon (once per episode).

    Args:
        agent: Configured ``QLearningPlacementAgent`` instance.
        reward_config: Optional ``RewardConfig`` overriding default reward weights.
        training: Whether to update the Q-table (True) or run greedy-only (False).
        name: Human-readable strategy name used in metrics output.
    """

    def __init__(
        self,
        agent: QLearningPlacementAgent,
        reward_config: Optional[RewardConfig] = None,
        training: bool = True,
        name: str = "Q-Learning",
    ) -> None:
        super().__init__(name=name)
        self.agent = agent
        self.reward_calculator = ThermalRewardCalculator(reward_config or RewardConfig())
        self.training = training
        self.feature_set = getattr(agent, "feature_set", "cpu_ram")
        self.stats = QLearningTrainingStats()
        self.current_episode_reward: float = 0.0
        self.last_episode_reward: float = 0.0

        if not training:
            self._eval_epsilon = self.agent.epsilon
            self.agent.epsilon = 0.0

        # Pending transitions buffered within a single time step
        self._pending: List[_PendingTransition] = []
        # Next-arriving VM lookahead (set by on_step_begin for the following step)
        self._next_vm: Optional[VM] = None
        # Cache all arriving VMs this step for lookahead calculation
        self._current_step_vms: List[VM] = []

    # ------------------------------------------------------------------
    # Public control API
    # ------------------------------------------------------------------

    def reset_episode(self) -> None:
        """Resets per-episode state while strictly preserving learned Q-values."""
        self._pending.clear()
        self._current_step_vms.clear()
        self.current_episode_reward = 0.0

    def set_training(self, training: bool) -> None:
        """Switch between training and evaluation mode at runtime.

        In evaluation mode epsilon is forced to 0.0 and the Q-table is frozen.
        """
        self.training = training
        if not training:
            if not hasattr(self, "_eval_epsilon"):
                self._eval_epsilon = self.agent.epsilon  # save so it can be restored
            self.agent.epsilon = 0.0
        else:
            self.restore_training()

    def restore_training(self) -> None:
        """Restore saved epsilon after a temporary evaluation run."""
        if hasattr(self, "_eval_epsilon"):
            self.agent.epsilon = self._eval_epsilon

    # ------------------------------------------------------------------
    # BasePlacementStrategy — required abstract method
    # ------------------------------------------------------------------

    def select_host(self, vm: VM, datacenter: DataCenter) -> Optional[Host]:
        """Selects a host using epsilon-greedy policy.

        Records the pre-placement state and eligible hosts for later Q-update.
        Does NOT perform the Q-update here — that happens in ``on_step_end``
        after the thermal model is advanced.

        Args:
            vm: VM requesting placement.
            datacenter: Current datacenter state.

        Returns:
            Selected Host (or None if no eligible host exists).
        """
        eligible_hosts = self.agent.get_eligible_hosts(vm, datacenter)

        # Snapshot imbalance and hotspot count BEFORE placement
        host_temps_before = [h.simulated_temperature_c for h in datacenter.hosts]
        imbalance_before = float(np.std(host_temps_before)) if len(host_temps_before) > 1 else 0.0
        hotspots_before = sum(
            1 for t in host_temps_before
            if t >= self.reward_calculator.config.hotspot_threshold_c
        )

        if not eligible_hosts:
            # Record a rejection-pending transition (no state/action)
            self._pending.append(_PendingTransition(
                vm=vm,
                host=None,
                placed=False,
                state=(),
                action=None,
                imbalance_before=imbalance_before,
                hotspots_before=hotspots_before,
            ))
            self.stats.total_rejections += 1
            return None

        # Build state for the first eligible host to represent the global state
        # (The agent's select_host uses per-host states internally for exploitation;
        # for the Q-table update we record the state corresponding to the chosen host.)
        # We use the agent's existing select_host to pick the host, then reconstruct
        # the state for that specific host for the transition record.
        selected_host = self.agent.select_host(vm, datacenter)
        if selected_host is None:
            # Fallback: exploration chose None (should not happen, but guard)
            selected_host = eligible_hosts[0]

        pre_state = self.agent.discretize_state(vm, selected_host, datacenter)

        self._pending.append(_PendingTransition(
            vm=vm,
            host=selected_host,
            placed=False,  # will be updated in on_placement_outcome
            state=pre_state,
            action=selected_host.host_id,
            imbalance_before=imbalance_before,
            hotspots_before=hotspots_before,
        ))

        return selected_host

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_step_begin(
        self,
        time_step: int,
        arriving_vms: List[VM],
        datacenter: DataCenter,
    ) -> None:
        """Cache VMs arriving this step (used for lookahead in next-state calc)."""
        if time_step == 0:
            self.current_episode_reward = 0.0
        self._current_step_vms = list(arriving_vms)
        self._pending.clear()

    def on_placement_outcome(
        self,
        vm: VM,
        host: Optional[Host],
        placed: bool,
        time_step: int,
        datacenter: DataCenter,
    ) -> None:
        """Update the placed flag on the most recent pending transition."""
        # Match by VM identity — the last _pending entry for this vm
        for trans in reversed(self._pending):
            if trans.vm is vm:
                trans.placed = placed
                break

    def on_step_end(
        self,
        time_step: int,
        datacenter: DataCenter,
        thermal_model: Any = None,
    ) -> None:
        """Compute rewards and perform Q-table updates after the thermal update.

        Called by the simulator after ``thermal_model.update_datacenter()`` so
        that reward components based on post-placement temperatures are accurate.
        """
        if not self._pending:
            return

        # Snapshot post-thermal temperatures for imbalance calculation
        host_temps_after = [h.simulated_temperature_c for h in datacenter.hosts]
        imbalance_after = float(np.std(host_temps_after)) if len(host_temps_after) > 1 else 0.0
        hotspots_after = sum(
            1 for t in host_temps_after
            if t >= self.reward_calculator.config.hotspot_threshold_c
        )

        # Determine the next VM (lookahead for non-terminal transitions).
        # In the existing simulator design, all arriving VMs for this step are
        # already in self._current_step_vms. The "next VM" is the first VM in
        # the next step, which we don't yet know. We use a simple heuristic:
        # treat each transition as terminal (no future Q) because:
        #   (a) We don't have the next step's workload at this point.
        #   (b) The imbalance/hotspot reward signal is rich enough per-step.
        # This is the consistent approach as specified in Phase 3 requirements
        # ("if the simulator cannot determine the next VM, implement a consistent
        # approach based on the existing event sequence").
        is_terminal = True

        for trans in self._pending:
            if trans.placed and trans.host is not None:
                # Compute reward from post-thermal observations
                host_temp_after = trans.host.simulated_temperature_c
                host_util = trans.host.cpu_utilization
                host_disk_util = trans.host.disk_io_utilization
                is_new_hotspot = (hotspots_after > trans.hotspots_before)
                is_existing_hotspot = (
                    host_temp_after >= self.reward_calculator.config.hotspot_threshold_c
                    and not is_new_hotspot
                )

                breakdown = self.reward_calculator.compute_reward(
                    placement_success=True,
                    host_temp_c=host_temp_after,
                    host_util=host_util,
                    is_new_hotspot=is_new_hotspot,
                    is_existing_hotspot=is_existing_hotspot,
                    imbalance_before=trans.imbalance_before,
                    imbalance_after=imbalance_after,
                    host_disk_util=host_disk_util,
                )
                reward = breakdown.total_reward

            else:
                # Rejection reward
                breakdown = self.reward_calculator.compute_reward(placement_success=False)
                reward = breakdown.total_reward

            self.stats.cumulative_reward += reward
            self.current_episode_reward += reward

            if self.training and trans.state and trans.action is not None:
                # Q-table update: terminal transition (no future Q term)
                self.agent.update(
                    state=trans.state,
                    action=trans.action,
                    reward=reward,
                    next_state=None,
                    terminal=is_terminal,
                    next_eligible_actions=None,
                )
                self.stats.total_q_updates += 1

            if trans.placed:
                self.stats.total_placements += 1
            # rejections are counted in select_host to avoid double-counting

        self._pending.clear()

    def on_simulation_end(self, datacenter: DataCenter) -> None:
        """Decay epsilon at episode end (once per simulation run)."""
        self.stats.episode_count += 1
        self.last_episode_reward = self.current_episode_reward
        self.stats.episode_rewards.append(self.last_episode_reward)
        if self.training:
            new_eps = self.agent.decay_epsilon()
            self.stats.epsilon_history.append(new_eps)
            logger.info(
                "Episode %d complete | cumulative_reward=%.2f | episode_reward=%.2f | epsilon=%.4f",
                self.stats.episode_count,
                self.stats.cumulative_reward,
                self.last_episode_reward,
                new_eps,
            )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        """Current exploration rate."""
        return self.agent.epsilon

    @property
    def q_table_size(self) -> int:
        """Number of non-zero entries in the Q-table."""
        return len(self.agent.q_table)

