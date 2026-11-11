"""Dedicated training CLI entrypoint for ThermoPlace Q-Learning Agent.

Executes multi-episode tabular Q-Learning training driven by real Bitbrains CPU traces
or synthetic workloads, logging performance metrics to CSV, generating progress plots,
and serializing the learned Q-table.

Example:
    python train.py --dataset-dir dataset/bitbrains/fastStorage/2013-8 --episodes 500
"""

import argparse
from pathlib import Path
import sys

# Ensure UTF-8 stdout formatting for degree symbols on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.qlearning.trainer import QLearningTrainer, TrainingConfig


# @flows User -> CLI via parse_args -- "CLI configuration input parsing"
# @handles configuration on CLI -- "Validates hyperparameter inputs and dataset paths"
def parse_args(args=None):
    """Parses command-line arguments for Q-Learning training."""
    parser = argparse.ArgumentParser(
        description="ThermoPlace: Q-Learning Agent Training"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset/bitbrains/fastStorage/2013-8",
        help="Path to Bitbrains dataset directory (default: dataset/bitbrains/fastStorage/2013-8)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Total training episodes to run (default: 500)",
    )
    parser.add_argument(
        "--learning-rate", "--alpha",
        type=float,
        default=0.1,
        dest="learning_rate",
        help="Learning rate alpha in [0, 1] (default: 0.1)",
    )
    parser.add_argument(
        "--discount-factor", "--gamma",
        type=float,
        default=0.95,
        dest="discount_factor",
        help="Discount factor gamma in [0, 1] (default: 0.95)",
    )
    parser.add_argument(
        "--initial-epsilon",
        type=float,
        default=1.0,
        help="Initial exploration rate in [0, 1] (default: 1.0)",
    )
    parser.add_argument(
        "--min-epsilon",
        type=float,
        default=0.05,
        help="Minimum exploration rate floor (default: 0.05)",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.995,
        help="Multiplicative decay factor per episode (default: 0.995)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max-vms",
        type=int,
        default=50,
        help="Maximum number of VM traces to load (default: 50; 0 for all)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Simulation time steps per episode (default: 100)",
    )
    parser.add_argument(
        "--racks",
        type=int,
        default=2,
        help="Number of server racks in the data center (default: 2)",
    )
    parser.add_argument(
        "--hosts-per-rack",
        type=int,
        default=4,
        help="Number of hosts per rack (default: 4)",
    )
    parser.add_argument(
        "--cpu-per-host",
        type=int,
        default=16,
        help="CPU cores capacity per physical host (default: 16)",
    )
    parser.add_argument(
        "--ram-per-host",
        type=int,
        default=64,
        help="RAM capacity per physical host in GB (default: 64)",
    )
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=2.0,
        help="Mean VM arrivals per time step (Poisson process, default: 2.0)",
    )
    parser.add_argument(
        "--ambient-temp",
        type=float,
        default=25.0,
        help="CRAH supply air temperature in Celsius (default: 25.0)",
    )
    parser.add_argument(
        "--hotspot-threshold",
        type=float,
        default=75.0,
        help="Hotspot safety threshold in Celsius (default: 75.0)",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="cpu_ram",
        choices=["cpu_ram", "cpu_ram_disk_read", "cpu_ram_disk_write", "cpu_ram_disk"],
        help="Resource feature set: 'cpu_ram' (default), 'cpu_ram_disk_read', 'cpu_ram_disk_write', 'cpu_ram_disk'",
    )
    parser.add_argument(
        "--disk-io-per-host",
        type=float,
        default=100000.0,
        help="Disk I/O throughput capacity per host in KB/s (default: 100000.0 = 100 MB/s)",
    )
    parser.add_argument(
        "--w-disk-penalty",
        type=float,
        default=0.0,
        help="Weight for excessive disk saturation penalty (default: 0.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/training",
        help="Directory to save CSV history (default: results/training)",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="results/training/plots",
        help="Directory to save training plots (default: results/training/plots)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/qlearning/q_table.json",
        help="File path to save the trained Q-table JSON (default: models/qlearning/q_table.json)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing CSV history and model files instead of preserving them",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="Skip generating Matplotlib progress plots",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Console progress log interval in episodes (default: 10)",
    )
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)

    # Validate dataset path if specified
    dataset_path = None
    if parsed.dataset_dir:
        dataset_path = Path(parsed.dataset_dir)
        if not dataset_path.exists():
            print(f"[ERROR] Dataset directory not found: '{parsed.dataset_dir}'", file=sys.stderr)
            sys.exit(1)
        if not dataset_path.is_dir():
            print(f"[ERROR] Dataset path is not a directory: '{parsed.dataset_dir}'", file=sys.stderr)
            sys.exit(1)

    config = TrainingConfig(
        episodes=parsed.episodes,
        learning_rate=parsed.learning_rate,
        discount_factor=parsed.discount_factor,
        initial_epsilon=parsed.initial_epsilon,
        min_epsilon=parsed.min_epsilon,
        epsilon_decay=parsed.epsilon_decay,
        seed=parsed.seed,
        dataset_dir=str(dataset_path) if dataset_path else None,
        max_vms=parsed.max_vms,
        steps=parsed.steps,
        racks=parsed.racks,
        hosts_per_rack=parsed.hosts_per_rack,
        cpu_per_host=parsed.cpu_per_host,
        ram_per_host=parsed.ram_per_host,
        arrival_rate=parsed.arrival_rate,
        ambient_temp=parsed.ambient_temp,
        hotspot_threshold=parsed.hotspot_threshold,
        output_dir=parsed.output_dir,
        plots_dir=parsed.plots_dir,
        model_path=parsed.model_path,
        overwrite=parsed.overwrite,
        save_plots=not parsed.no_plots,
        save_model=True,
        features=parsed.features,
        disk_io_per_host=parsed.disk_io_per_host,
        w_disk_penalty=parsed.w_disk_penalty,
    )

    trainer = QLearningTrainer(config=config)
    trainer.train(log_interval=parsed.log_interval)


if __name__ == "__main__":
    main()

