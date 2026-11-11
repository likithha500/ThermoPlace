"""Training orchestration module for Q-Learning VM placement agent.

Implements the multi-episode reinforcement learning training loop:
1. Environment and data center state initialization / reset.
2. Consistent workload trace replay across episodes.
3. Epsilon-greedy exploration and thermal-aware Q-table updates.
4. Episode-level telemetry recording to CSV.
5. Training progress visualization with Matplotlib.
6. Q-table model persistence with metadata validation.
7. Strict separation between training and evaluation modes.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from src.datacenter import DataCenter
from src.metrics import MetricsCollector
from src.qlearning.agent import QLearningPlacementAgent, StateDiscretizerConfig
from src.qlearning.reward import RewardConfig
from src.qlearning.strategy import QLearningSimulatorStrategy
from src.simulator import Simulator
from src.thermal import ThermalConfig, ThermalModel
from src.workload import VM, TraceWorkloadGenerator, WorkloadGenerator

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configurable training parameters for tabular Q-Learning.

    Attributes:
        episodes: Total simulation training episodes to execute (default: 500).
        learning_rate: Step size alpha in [0, 1] for TD updates (default: 0.1).
        discount_factor: Discount factor gamma in [0, 1] for future returns (default: 0.95).
        initial_epsilon: Starting exploration rate in [0, 1] (default: 1.0).
        min_epsilon: Minimum exploration floor (default: 0.05).
        epsilon_decay: Multiplicative decay factor per episode (default: 0.995).
        seed: Random seed for arrival scheduling, flavors, and agent exploration (default: 42).
        dataset_dir: Path to dataset directory. Used with dataset_type to select the loader.
        dataset_type: Dataset loader to use: 'bitbrains' (default) or 'planetlab'.
        max_vms: Maximum number of VM traces to load from dataset (default: 50).
        steps: Total discrete simulation time steps per episode (default: 100).
        racks: Number of server racks in the data center (default: 2).
        hosts_per_rack: Number of hosts per server rack (default: 4).
        cpu_per_host: CPU cores capacity per physical host (default: 16).
        ram_per_host: RAM capacity per physical host in GB (default: 64).
        arrival_rate: Mean VM arrivals per time step (Poisson process, default: 2.0).
        ambient_temp: Computer Room Air Handler ambient temperature in Celsius (default: 25.0).
        hotspot_threshold: Critical temperature threshold in Celsius (default: 75.0).
        output_dir: Directory to store training CSV history (default: 'results/training').
        csv_filename: Base filename for CSV history (default: 'training_history.csv').
        plots_dir: Directory to save generated progress plots (default: 'results/training/plots').
        model_path: Output path for saved Q-table JSON (default: 'models/qlearning/q_table.json').
        overwrite: Whether to overwrite existing CSV and model files (default: False).
        save_plots: Whether to generate and save Matplotlib figures after training (default: True).
        save_model: Whether to save the trained Q-table after training (default: True).
    """
    episodes: int = 500
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    initial_epsilon: float = 1.0
    min_epsilon: float = 0.05
    epsilon_decay: float = 0.995
    seed: int = 42

    dataset_dir: Optional[str] = None
    dataset_type: str = "bitbrains"
    max_vms: int = 50
    steps: int = 100
    racks: int = 2
    hosts_per_rack: int = 4
    cpu_per_host: int = 16
    ram_per_host: int = 64
    arrival_rate: float = 2.0
    ambient_temp: float = 25.0
    hotspot_threshold: float = 75.0

    output_dir: str = "results/training"
    csv_filename: str = "training_history.csv"
    plots_dir: str = "results/training/plots"
    model_path: str = "models/qlearning/q_table.json"
    overwrite: bool = False
    save_plots: bool = True
    save_model: bool = True


@dataclass
class EpisodeRecord:
    """Performance statistics captured for a single training episode."""
    episode: int
    cumulative_reward: float
    avg_reward_per_decision: float
    successful_placements: int
    rejected_vms: int
    total_hotspots: int
    max_host_temperature: float
    avg_host_temperature: float
    thermal_imbalance: float
    avg_cpu_utilization: float
    avg_ram_utilization: float
    epsilon: float
    q_table_entries: int

    def to_dict(self) -> Dict[str, Any]:
        """Converts episode record to dictionary for CSV formatting."""
        return asdict(self)


class QLearningTrainer:
    """Manages the full multi-episode training lifecycle for the Q-Learning agent."""

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        agent: Optional[QLearningPlacementAgent] = None,
        reward_config: Optional[RewardConfig] = None,
        workload_generator: Optional[Union[TraceWorkloadGenerator, WorkloadGenerator]] = None,
        schedule: Optional[Dict[int, List[VM]]] = None,
    ) -> None:
        """Initializes the trainer with configuration, components, and workload.

        Args:
            config: TrainingConfig instance or None for defaults.
            agent: Optional pre-configured QLearningPlacementAgent.
            reward_config: Optional reward weight configuration.
            workload_generator: Optional workload generator.
            schedule: Optional pre-constructed workload schedule (used mainly in tests).
        """
        self.config = config or TrainingConfig()
        self.reward_config = reward_config or RewardConfig(
            ambient_temp_c=self.config.ambient_temp,
            hotspot_threshold_c=self.config.hotspot_threshold,
        )
        self.workload_generator = workload_generator
        self._prebuilt_schedule = schedule
        self.history: List[EpisodeRecord] = []

        # Components built during setup
        self.datacenter: Optional[DataCenter] = None
        self.thermal_model: Optional[ThermalModel] = None
        self.agent: Optional[QLearningPlacementAgent] = agent
        self.strategy: Optional[QLearningSimulatorStrategy] = None
        self.simulator: Optional[Simulator] = None

        self._setup_components()

    def _setup_components(self) -> None:
        """Initializes datacenter topology, thermal model, agent, and strategy."""
        cfg = self.config

        # 1. Physical Data Center
        self.datacenter = DataCenter.create_standard_datacenter(
            num_racks=cfg.racks,
            hosts_per_rack=cfg.hosts_per_rack,
            cpu_per_host=cfg.cpu_per_host,
            ram_per_host=cfg.ram_per_host,
            initial_temperature_c=cfg.ambient_temp,
        )

        # 2. Analytical RC Thermal Model
        thermal_cfg = ThermalConfig(
            ambient_temperature_c=cfg.ambient_temp,
            hotspot_threshold_c=cfg.hotspot_threshold,
        )
        self.thermal_model = ThermalModel(config=thermal_cfg)

        # 3. Agent (preserve if provided, else construct with config hyperparameters)
        if self.agent is None:
            self.agent = QLearningPlacementAgent(
                learning_rate=cfg.learning_rate,
                discount_factor=cfg.discount_factor,
                epsilon=cfg.initial_epsilon,
                min_epsilon=cfg.min_epsilon,
                epsilon_decay=cfg.epsilon_decay,
                seed=cfg.seed,
            )

        # 4. Learning Placement Strategy
        self.strategy = QLearningSimulatorStrategy(
            agent=self.agent,
            reward_config=self.reward_config,
            training=True,
        )

        # 5. Discrete Simulator
        self.simulator = Simulator(
            datacenter=self.datacenter,
            thermal_model=self.thermal_model,
        )

        # 6. Workload Generator if not provided and dataset_dir specified
        if self.workload_generator is None and self._prebuilt_schedule is None:
            if cfg.dataset_dir is not None:
                self._load_dataset_workload(cfg.dataset_dir, cfg.dataset_type)
            else:
                # Default synthetic workload for testing/fallback
                self.workload_generator = WorkloadGenerator(
                    seed=cfg.seed,
                    total_time_steps=cfg.steps,
                    arrival_rate_lambda=cfg.arrival_rate,
                )

    def _load_dataset_workload(self, dataset_dir: Union[str, Path], dataset_type: str = "bitbrains") -> None:
        """Loads traces from the specified dataset and initializes TraceWorkloadGenerator.

        Args:
            dataset_dir: Path to the dataset directory.
            dataset_type: Which loader to use — 'bitbrains' (default) or 'planetlab'.
        """
        dtype = dataset_type.lower().strip()
        if dtype == "planetlab":
            from src.planetlab_loader import PlanetLabLoader
            loader: Union[PlanetLabLoader, "BitbrainsLoader"] = PlanetLabLoader()
            loader_label = "PlanetLab"
        else:
            from src.bitbrains_loader import BitbrainsLoader
            loader = BitbrainsLoader()
            loader_label = "Bitbrains"

        report = loader.load_with_report(
            directory_path=dataset_dir,
            max_traces=self.config.max_vms if self.config.max_vms > 0 else None,
            recursive=True,
        )
        if report.total_loaded == 0:
            raise ValueError(f"No valid {loader_label} traces found in '{dataset_dir}'.")

        self.workload_generator = TraceWorkloadGenerator(
            traces=report.valid_traces,
            arrival_rate_lambda=self.config.arrival_rate,
            max_arrival_time=self.config.steps,
            seed=self.config.seed,
        )

    def _load_bitbrains_workload(self, dataset_dir: Union[str, Path]) -> None:
        """Legacy alias for Bitbrains loading; delegates to _load_dataset_workload."""
        self._load_dataset_workload(dataset_dir, dataset_type="bitbrains")

    def get_episode_schedule(self) -> Dict[int, List[VM]]:
        """Provides a freshly reset or generated workload schedule for an episode.

        Guarantees that VM durations, arrival times, and trace offsets start from
        their initial states and do not leak between episodes.
        """
        if self.workload_generator is not None:
            # Re-generate schedule deterministically; generators initialize RNG with self.seed
            return self.workload_generator.generate_schedule_by_step(
                total_time_steps=self.config.steps
            )
        elif self._prebuilt_schedule is not None:
            # Reset existing VMs in the pre-built schedule
            Simulator.reset_workload(self._prebuilt_schedule)
            return self._prebuilt_schedule
        else:
            raise RuntimeError("No workload generator or schedule available for training.")

    def run_episode(self, episode_num: int) -> EpisodeRecord:
        """Executes a single training episode and captures performance statistics.

        Steps:
        1. Resets datacenter and simulator to baseline initial conditions.
        2. Obtains fresh unexecuted VM workload schedule.
        3. Resets strategy episode buffers.
        4. Runs simulation with Q-Learning strategy under epsilon-greedy exploration.
        5. Computes and records episode performance statistics.
        6. Note: Epsilon decay is automatically invoked by strategy.on_simulation_end().

        Args:
            episode_num: 1-based index of current episode.

        Returns:
            EpisodeRecord containing complete telemetry for this episode.
        """
        # 1. Reset simulator and data center
        self.simulator.reset(initial_temp=self.config.ambient_temp)

        # 2. Obtain clean workload schedule
        schedule = self.get_episode_schedule()

        # 3. Reset strategy buffers (preserves learned Q-table)
        self.strategy.reset_episode()

        # Record epsilon at start of episode before decay
        eps_at_episode_end = self.agent.epsilon

        # 4. Run discrete simulation
        metrics = self.simulator.run(
            workload_schedule=schedule,
            strategy=self.strategy,
            total_steps=self.config.steps,
        )

        # Note: on_simulation_end was called inside sim.run(), which decayed agent.epsilon
        current_eps = self.agent.epsilon

        # 5. Extract episode metrics
        df = metrics.to_dataframe()
        summary = metrics.summary()

        ep_reward = float(self.strategy.last_episode_reward)
        total_decisions = metrics.total_placed_vms + metrics.total_rejected_vms
        avg_reward = (ep_reward / total_decisions) if total_decisions > 0 else 0.0

        max_host_temp = float(df["max_temp_c"].max()) if not df.empty else self.config.ambient_temp
        avg_host_temp = float(df["mean_temp_c"].mean()) if not df.empty else self.config.ambient_temp
        thermal_imbalance = float(df["thermal_imbalance_std"].mean()) if not df.empty else 0.0
        avg_cpu = float(df["mean_cpu_util"].mean()) if not df.empty else 0.0
        avg_ram = float(df["mean_ram_util"].mean()) if not df.empty else 0.0
        total_hotspots = int(summary.get("cumulative_hotspot_events", 0))

        record = EpisodeRecord(
            episode=episode_num,
            cumulative_reward=round(ep_reward, 4),
            avg_reward_per_decision=round(avg_reward, 4),
            successful_placements=int(metrics.total_placed_vms),
            rejected_vms=int(metrics.total_rejected_vms),
            total_hotspots=total_hotspots,
            max_host_temperature=round(max_host_temp, 2),
            avg_host_temperature=round(avg_host_temp, 2),
            thermal_imbalance=round(thermal_imbalance, 4),
            avg_cpu_utilization=round(avg_cpu, 4),
            avg_ram_utilization=round(avg_ram, 4),
            epsilon=round(current_eps, 5),
            q_table_entries=len(self.agent.q_table),
        )

        self.history.append(record)
        return record

    def train(
        self,
        episodes: Optional[int] = None,
        verbose: bool = True,
        log_interval: int = 25,
        callback: Optional[Callable[[EpisodeRecord], None]] = None,
    ) -> List[EpisodeRecord]:
        """Executes the full multi-episode training loop.

        Args:
            episodes: Total episodes override; defaults to config.episodes.
            verbose: If True, prints periodic training logs to stdout.
            log_interval: Frequency of console progress logging in episodes.
            callback: Optional hook invoked with EpisodeRecord after each episode.

        Returns:
            List of EpisodeRecord instances across all executed episodes.
        """
        total_episodes = episodes or self.config.episodes
        self.strategy.set_training(True)

        if verbose:
            print("=" * 80)
            print(f"  Starting Q-Learning Training ({total_episodes} Episodes)")
            print("=" * 80)
            print(
                f"  Hyperparameters: alpha={self.config.learning_rate}, "
                f"gamma={self.config.discount_factor}, "
                f"init_eps={self.config.initial_epsilon}, "
                f"min_eps={self.config.min_epsilon}, "
                f"decay={self.config.epsilon_decay}"
            )
            print(
                f"  Topology: {self.config.racks} racks x {self.config.hosts_per_rack} hosts | "
                f"Steps/Episode: {self.config.steps} | Seed: {self.config.seed}"
            )
            print("-" * 80)

        for ep in range(1, total_episodes + 1):
            record = self.run_episode(ep)

            if callback is not None:
                callback(record)

            if verbose and (ep == 1 or ep % log_interval == 0 or ep == total_episodes):
                print(
                    f"  Ep {ep:>4}/{total_episodes} | "
                    f"Reward: {record.cumulative_reward:>8.2f} | "
                    f"Placed: {record.successful_placements:>3} | "
                    f"Rej: {record.rejected_vms:>2} | "
                    f"Hotspots: {record.total_hotspots:>2} | "
                    f"MaxT: {record.max_host_temperature:>5.1f}°C | "
                    f"Imbal: {record.thermal_imbalance:>5.2f} | "
                    f"Eps: {record.epsilon:>6.4f} | "
                    f"Q-size: {record.q_table_entries:>4}"
                )

        if verbose:
            print("-" * 80)
            print(f"[OK] Training complete. Total Q-table entries learned: {len(self.agent.q_table)}")

        # Save history CSV
        csv_path = self.save_history_csv()
        if verbose:
            print(f"[*] Training history saved to: {csv_path}")

        # Save plots if configured
        if self.config.save_plots and self.history:
            plots = self.generate_plots()
            if verbose:
                print(f"[*] Generated {len(plots)} training plots in: {self.config.plots_dir}")

        # Save model if configured
        if self.config.save_model:
            model_path = self.save_model()
            if verbose:
                print(f"[*] Saved trained Q-table model to: {model_path}")

        if verbose:
            print("=" * 80 + "\n")

        return self.history

    # -------------------------------------------------------------------------
    # Persistence & File Output
    # -------------------------------------------------------------------------

    # @flows QLearningTrainer -> FileSystem via CSV -- "Writes episode-level training history to CSV"
    # @handles training_history on QLearningTrainer -- "Formats episode performance statistics into structured CSV"
    def save_history_csv(
        self,
        output_path: Optional[Union[str, Path]] = None,
        overwrite: Optional[bool] = None,
    ) -> Path:
        """Saves accumulated episode history to a structured CSV file.

        Creates output directories automatically. Avoids accidental overwrite of
        prior runs unless overwrite=True is explicitly set.

        Args:
            output_path: Path override for CSV file.
            overwrite: Overwrite override. Defaults to config.overwrite.

        Returns:
            Path object pointing to the written CSV file.
        """
        allow_overwrite = self.config.overwrite if overwrite is None else overwrite

        if output_path is not None:
            target_path = Path(output_path)
        else:
            base_dir = Path(self.config.output_dir)
            target_path = base_dir / self.config.csv_filename

        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.exists() and not allow_overwrite:
            # Preserve previous results: write to timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = target_path.stem
            suffix = target_path.suffix
            target_path = target_path.parent / f"{stem}_{timestamp}{suffix}"

        fieldnames = [
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

        with open(target_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.history:
                writer.writerow(record.to_dict())

        return target_path

    # @flows QLearningTrainer -> FileSystem via JSON -- "Saves trained Q-table with metadata"
    # @handles model_weights on QLearningTrainer -- "Serializes trained agent state and training hyperparameters"
    def save_model(
        self,
        model_path: Optional[Union[str, Path]] = None,
        overwrite: Optional[bool] = None,
    ) -> Path:
        """Saves the trained agent Q-table along with training configuration metadata.

        Args:
            model_path: Destination JSON file path.
            overwrite: Overwrite override.

        Returns:
            Path to saved JSON file.
        """
        allow_overwrite = self.config.overwrite if overwrite is None else overwrite
        target_path = Path(model_path or self.config.model_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.exists() and not allow_overwrite:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_path = target_path.parent / f"{target_path.stem}_{timestamp}{target_path.suffix}"

        metadata = {
            "training_completed_at": datetime.now().isoformat(),
            "total_episodes": len(self.history),
            "final_epsilon": self.agent.epsilon,
            "training_config": asdict(self.config),
            "total_q_updates": self.strategy.stats.total_q_updates,
            "total_placements": self.strategy.stats.total_placements,
            "total_rejections": self.strategy.stats.total_rejections,
        }

        self.agent.save(target_path, metadata=metadata)
        return target_path

    # -------------------------------------------------------------------------
    # Progress Visualization (Matplotlib)
    # -------------------------------------------------------------------------

    def generate_plots(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        history: Optional[List[EpisodeRecord]] = None,
    ) -> List[Path]:
        """Generates required training progress figures using Matplotlib.

        Required Graphs:
        1. Cumulative reward versus training episodes.
        2. Average reward versus training episodes.
        3. Epsilon decay versus training episodes.
        4. Hotspot count versus training episodes.
        5. Maximum host temperature versus training episodes.
        6. Thermal imbalance versus training episodes.
        7. Successful VM placements versus training episodes.
        Plus a combined 7-panel summary dashboard figure.

        Args:
            output_dir: Destination directory for PNG image files.
            history: Optional EpisodeRecord list override.

        Returns:
            List of Path objects for all created plot files.
        """
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend safe for headless servers
        import matplotlib.pyplot as plt

        records = history or self.history
        if not records:
            logger.warning("No training history records available to plot.")
            return []

        out_path = Path(output_dir or self.config.plots_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        episodes = [r.episode for r in records]
        cum_rewards = [r.cumulative_reward for r in records]
        avg_rewards = [r.avg_reward_per_decision for r in records]
        epsilons = [r.epsilon for r in records]
        hotspots = [r.total_hotspots for r in records]
        max_temps = [r.max_host_temperature for r in records]
        imbalance = [r.thermal_imbalance for r in records]
        placements = [r.successful_placements for r in records]

        # Helper for moving average smoothing
        def _smooth(values: List[float], window: int = 15) -> np.ndarray:
            if len(values) < window:
                return np.array(values, dtype=float)
            kernel = np.ones(window) / window
            return np.convolve(values, kernel, mode="same")

        saved_files: List[Path] = []

        # 1. Cumulative Reward
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(episodes, cum_rewards, color="#1f77b4", alpha=0.35, label="Raw Episode Reward")
        if len(episodes) >= 15:
            ax.plot(episodes, _smooth(cum_rewards), color="#0d47a1", linewidth=2.0, label="Moving Average (15)")
        ax.set_title("Cumulative Reward vs. Training Episodes", fontsize=13, fontweight="bold")
        ax.set_xlabel("Training Episode", fontsize=11)
        ax.set_ylabel("Total Cumulative Reward", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best")
        fig.tight_layout()
        p1 = out_path / "cumulative_reward.png"
        fig.savefig(p1, dpi=150)
        plt.close(fig)
        saved_files.append(p1)

        # 2. Average Reward per Decision
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(episodes, avg_rewards, color="#2ca02c", alpha=0.35, label="Raw Avg Reward")
        if len(episodes) >= 15:
            ax.plot(episodes, _smooth(avg_rewards), color="#1b5e20", linewidth=2.0, label="Moving Average (15)")
        ax.set_title("Average Reward per Decision vs. Training Episodes", fontsize=13, fontweight="bold")
        ax.set_xlabel("Training Episode", fontsize=11)
        ax.set_ylabel("Average Reward / Decision", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best")
        fig.tight_layout()
        p2 = out_path / "average_reward.png"
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        saved_files.append(p2)

        # 3. Epsilon Decay
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(episodes, epsilons, color="#e65100", linewidth=2.0, label="Exploration Rate (ε)")
        ax.axhline(self.config.min_epsilon, color="red", linestyle=":", label=f"Min ε ({self.config.min_epsilon})")
        ax.set_title("Epsilon Exploration Decay vs. Training Episodes", fontsize=13, fontweight="bold")
        ax.set_xlabel("Training Episode", fontsize=11)
        ax.set_ylabel("Epsilon (ε)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best")
        fig.tight_layout()
        p3 = out_path / "epsilon_decay.png"
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        saved_files.append(p3)

        # 4. Hotspot Count
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(episodes, hotspots, color="#d32f2f", alpha=0.35, label="Raw Hotspot Count")
        if len(episodes) >= 15:
            ax.plot(episodes, _smooth(hotspots), color="#b71c1c", linewidth=2.0, label="Moving Average (15)")
        ax.set_title("Hotspot Event Count vs. Training Episodes", fontsize=13, fontweight="bold")
        ax.set_xlabel("Training Episode", fontsize=11)
        ax.set_ylabel("Total Hotspots (T >= 75°C)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best")
        fig.tight_layout()
        p4 = out_path / "hotspot_count.png"
        fig.savefig(p4, dpi=150)
        plt.close(fig)
        saved_files.append(p4)

        # 5. Maximum Host Temperature
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(episodes, max_temps, color="#ff9800", alpha=0.35, label="Raw Max Temperature")
        if len(episodes) >= 15:
            ax.plot(episodes, _smooth(max_temps), color="#e65100", linewidth=2.0, label="Moving Average (15)")
        ax.axhline(self.config.hotspot_threshold, color="red", linestyle="--", label=f"Hotspot Limit ({self.config.hotspot_threshold}°C)")
        ax.set_title("Maximum Host Temperature vs. Training Episodes", fontsize=13, fontweight="bold")
        ax.set_xlabel("Training Episode", fontsize=11)
        ax.set_ylabel("Peak Temperature (°C)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best")
        fig.tight_layout()
        p5 = out_path / "max_host_temperature.png"
        fig.savefig(p5, dpi=150)
        plt.close(fig)
        saved_files.append(p5)

        # 6. Thermal Imbalance
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(episodes, imbalance, color="#9c27b0", alpha=0.35, label="Raw Imbalance std(T)")
        if len(episodes) >= 15:
            ax.plot(episodes, _smooth(imbalance), color="#4a148c", linewidth=2.0, label="Moving Average (15)")
        ax.set_title("Thermal Imbalance vs. Training Episodes", fontsize=13, fontweight="bold")
        ax.set_xlabel("Training Episode", fontsize=11)
        ax.set_ylabel("Spatial Temp Std Dev σ_T (°C)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best")
        fig.tight_layout()
        p6 = out_path / "thermal_imbalance.png"
        fig.savefig(p6, dpi=150)
        plt.close(fig)
        saved_files.append(p6)

        # 7. Successful VM Placements
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(episodes, placements, color="#00897b", alpha=0.35, label="Raw Placements")
        if len(episodes) >= 15:
            ax.plot(episodes, _smooth(placements), color="#004d40", linewidth=2.0, label="Moving Average (15)")
        ax.set_title("Successful VM Placements vs. Training Episodes", fontsize=13, fontweight="bold")
        ax.set_xlabel("Training Episode", fontsize=11)
        ax.set_ylabel("Number of Placed VMs", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best")
        fig.tight_layout()
        p7 = out_path / "successful_placements.png"
        fig.savefig(p7, dpi=150)
        plt.close(fig)
        saved_files.append(p7)

        # Combined Multi-Panel Training Dashboard
        fig, axes = plt.subplots(4, 2, figsize=(16, 18))
        fig.suptitle("ThermoPlace Q-Learning Training Metrics Dashboard", fontsize=16, fontweight="bold", y=0.995)

        # (0, 0) Cumulative Reward
        axes[0, 0].plot(episodes, cum_rewards, alpha=0.3, color="#1f77b4")
        if len(episodes) >= 15:
            axes[0, 0].plot(episodes, _smooth(cum_rewards), color="#0d47a1", label="Smoothed")
        axes[0, 0].set_title("Cumulative Reward", fontweight="bold")
        axes[0, 0].set_ylabel("Total Reward")
        axes[0, 0].grid(True, linestyle="--", alpha=0.5)

        # (0, 1) Average Reward
        axes[0, 1].plot(episodes, avg_rewards, alpha=0.3, color="#2ca02c")
        if len(episodes) >= 15:
            axes[0, 1].plot(episodes, _smooth(avg_rewards), color="#1b5e20", label="Smoothed")
        axes[0, 1].set_title("Average Reward per Decision", fontweight="bold")
        axes[0, 1].set_ylabel("Avg Reward")
        axes[0, 1].grid(True, linestyle="--", alpha=0.5)

        # (1, 0) Epsilon Decay
        axes[1, 0].plot(episodes, epsilons, color="#e65100", linewidth=2.0)
        axes[1, 0].axhline(self.config.min_epsilon, color="red", linestyle=":", label="Min ε")
        axes[1, 0].set_title("Epsilon Exploration Decay", fontweight="bold")
        axes[1, 0].set_ylabel("Epsilon (ε)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, linestyle="--", alpha=0.5)

        # (1, 1) Hotspot Count
        axes[1, 1].plot(episodes, hotspots, alpha=0.3, color="#d32f2f")
        if len(episodes) >= 15:
            axes[1, 1].plot(episodes, _smooth(hotspots), color="#b71c1c", label="Smoothed")
        axes[1, 1].set_title("Hotspot Count", fontweight="bold")
        axes[1, 1].set_ylabel("Events")
        axes[1, 1].grid(True, linestyle="--", alpha=0.5)

        # (2, 0) Max Host Temperature
        axes[2, 0].plot(episodes, max_temps, alpha=0.3, color="#ff9800")
        if len(episodes) >= 15:
            axes[2, 0].plot(episodes, _smooth(max_temps), color="#e65100", label="Smoothed")
        axes[2, 0].axhline(self.config.hotspot_threshold, color="red", linestyle="--", label="Threshold")
        axes[2, 0].set_title("Max Host Temperature", fontweight="bold")
        axes[2, 0].set_ylabel("Temperature (°C)")
        axes[2, 0].legend()
        axes[2, 0].grid(True, linestyle="--", alpha=0.5)

        # (2, 1) Thermal Imbalance
        axes[2, 1].plot(episodes, imbalance, alpha=0.3, color="#9c27b0")
        if len(episodes) >= 15:
            axes[2, 1].plot(episodes, _smooth(imbalance), color="#4a148c", label="Smoothed")
        axes[2, 1].set_title("Thermal Imbalance (std)", fontweight="bold")
        axes[2, 1].set_ylabel("Std Dev (°C)")
        axes[2, 1].grid(True, linestyle="--", alpha=0.5)

        # (3, 0) Placements vs Rejections
        rejections = [r.rejected_vms for r in records]
        axes[3, 0].plot(episodes, placements, color="#00897b", label="Placed")
        axes[3, 0].plot(episodes, rejections, color="#e53935", linestyle="--", label="Rejected")
        axes[3, 0].set_title("VM Placements vs. Rejections", fontweight="bold")
        axes[3, 0].set_xlabel("Training Episode")
        axes[3, 0].set_ylabel("VMs")
        axes[3, 0].legend()
        axes[3, 0].grid(True, linestyle="--", alpha=0.5)

        # (3, 1) Q-Table State Growth
        q_sizes = [r.q_table_entries for r in records]
        axes[3, 1].plot(episodes, q_sizes, color="#3949ab", linewidth=2.0)
        axes[3, 1].set_title("Learned Q-Table Size Growth", fontweight="bold")
        axes[3, 1].set_xlabel("Training Episode")
        axes[3, 1].set_ylabel("Non-Zero Q-Entries")
        axes[3, 1].grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()
        p_dash = out_path / "training_dashboard.png"
        fig.savefig(p_dash, dpi=150)
        plt.close(fig)
        saved_files.append(p_dash)

        return saved_files

    # -------------------------------------------------------------------------
    # Evaluation Mode Execution
    # -------------------------------------------------------------------------

    def evaluate(
        self,
        eval_schedule: Optional[Dict[int, List[VM]]] = None,
        steps: Optional[int] = None,
    ) -> MetricsCollector:
        """Runs the simulation in pure evaluation mode using the learned Q-table.

        Guarantees:
        - Epsilon is set to 0.0 (pure greedy exploitation).
        - Q-table updates are strictly disabled.
        - Epsilon decay does not occur.
        - Existing learned Q-values remain unmodified.

        Args:
            eval_schedule: Optional workload schedule for evaluation. Defaults to training schedule.
            steps: Discrete time steps to run. Defaults to config.steps.

        Returns:
            MetricsCollector containing full evaluation telemetry.
        """
        # Save training state and switch to evaluation mode
        self.strategy.set_training(False)
        self.simulator.reset(initial_temp=self.config.ambient_temp)

        schedule = eval_schedule or self.get_episode_schedule()
        Simulator.reset_workload(schedule)

        total_steps = steps or self.config.steps
        metrics = self.simulator.run(
            workload_schedule=schedule,
            strategy=self.strategy,
            total_steps=total_steps,
        )

        # Restore strategy training state
        self.strategy.restore_training()
        self.strategy.set_training(True)

        return metrics

