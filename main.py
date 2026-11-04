"""Main entry point for ThermoPlace — Thermal-Aware VM Scheduling Simulator.

Provides command-line interface for running thermal-aware VM placement simulations
using the Bitbrains GWA-T-12 dataset and either the First Fit or Q-Learning strategy.

Supported strategies
--------------------
  ``--strategy first_fit``    — Traditional First Fit baseline (default).
  ``--strategy q_learning``   — Tabular Q-Learning thermal-aware placement.

Example usage
-------------
Run First Fit simulation::

    python main.py --dataset-dir dataset/bitbrains/fastStorage/2013-8 --strategy first_fit

Run Q-Learning training (multi-episode)::

    python main.py --dataset-dir dataset/bitbrains/fastStorage/2013-8 --strategy q_learning --train --episodes 500

Run Q-Learning in evaluation mode using a saved Q-table::

    python main.py --dataset-dir dataset/bitbrains/fastStorage/2013-8 \
                   --strategy q_learning --eval-mode --q-table models/qlearning/q_table.json
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

from src.datacenter import DataCenter
from src.thermal import ThermalModel, ThermalConfig
from src.placement.first_fit import FirstFitPlacement
from src.simulator import Simulator
from src.workload import TraceWorkloadGenerator


# @flows User -> CLI via parse_args -- "CLI configuration input parsing"
# @handles configuration on CLI -- "Validates hyperparameter inputs and dataset paths"
def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="ThermoPlace: Thermal-Aware VM Scheduling Simulator"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to Bitbrains dataset directory (e.g. dataset/bitbrains/fastStorage/2013-8)",
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
        help="Total discrete simulation time steps (default: 100)",
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
        help="CPU cores capacity per host (default: 16)",
    )
    parser.add_argument(
        "--ram-per-host",
        type=int,
        default=64,
        help="RAM capacity per host in GB (default: 64)",
    )
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=2.0,
        help="Mean VM arrivals per time step (Poisson process, default: 2.0; 0 for all at t=0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic VM arrival timing and resource flavor assignment (default: 42)",
    )
    parser.add_argument(
        "--ambient-temp",
        type=float,
        default=25.0,
        help="Ambient CRAH temperature in Celsius (default: 25.0)",
    )
    parser.add_argument(
        "--hotspot-threshold",
        type=float,
        default=75.0,
        help="Critical hotspot temperature threshold in Celsius (default: 75.0)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="first_fit",
        choices=["first_fit", "q_learning"],
        help="Placement strategy to use: 'first_fit' (default) or 'q_learning'",
    )
    parser.add_argument(
        "--eval-mode",
        action="store_true",
        default=False,
        help="Run Q-Learning in evaluation mode (no Q-table updates, greedy selection only)",
    )
    # Training and persistence options
    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="Run multi-episode Q-Learning training loop",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of training episodes (default: 1 for single simulation; defaults to 500 when --train specified without --episodes)",
    )
    parser.add_argument(
        "--q-table",
        type=str,
        default=None,
        help="Path to saved Q-table JSON file (loaded for evaluation or continued training)",
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
        "--output-dir",
        type=str,
        default="results/training",
        help="Directory to save CSV history (default: results/training)",
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
    return parser.parse_args(args)


def print_banner():
    banner = r"""
================================================================================
  ThermoPlace: Thermal-Aware VM Scheduling Simulator
================================================================================
    """
    print(banner)


def print_summary(summary: dict, strategy_name: str) -> None:
    """Prints a formatted simulation result summary."""
    print("\n" + "=" * 80)
    print(f"  Simulation Results — {strategy_name}")
    print("=" * 80)
    for key, value in summary.items():
        label = key.replace("_", " ").title()
        if isinstance(value, float):
            print(f"    {label:<40}: {value:.4f}")
        else:
            print(f"    {label:<40}: {value}")
    print("-" * 80)


def _build_strategy(parsed_args):
    """Constructs the placement strategy from CLI arguments."""
    if parsed_args.strategy == "first_fit":
        return FirstFitPlacement()

    # Q-Learning strategy
    from src.qlearning.agent import QLearningPlacementAgent
    from src.qlearning.strategy import QLearningSimulatorStrategy

    # Check if a saved Q-table is provided
    if parsed_args.q_table is not None:
        agent = QLearningPlacementAgent.load_from_file(parsed_args.q_table)
        print(f"[*] Loaded Q-table from: {parsed_args.q_table} ({len(agent.q_table)} entries)")
    else:
        agent = QLearningPlacementAgent(
            learning_rate=parsed_args.learning_rate,
            discount_factor=parsed_args.discount_factor,
            epsilon=parsed_args.initial_epsilon,
            min_epsilon=parsed_args.min_epsilon,
            epsilon_decay=parsed_args.epsilon_decay,
            seed=parsed_args.seed,
        )

    # In evaluation mode: force epsilon = 0.0 and freeze updates
    is_training = not parsed_args.eval_mode
    if parsed_args.eval_mode:
        agent.epsilon = 0.0

    strategy = QLearningSimulatorStrategy(
        agent=agent,
        training=is_training,
    )
    return strategy


def main(args=None):
    parsed_args = parse_args(args)
    print_banner()

    print("[*] Simulation Configuration:")
    print(f"    - Dataset Directory    : {parsed_args.dataset_dir}")
    print(f"    - Max VM Traces        : {parsed_args.max_vms if parsed_args.max_vms > 0 else 'All'}")
    print(f"    - Total Time Steps     : {parsed_args.steps}")
    print(f"    - Data Center Topology : {parsed_args.racks} racks x {parsed_args.hosts_per_rack} hosts = {parsed_args.racks * parsed_args.hosts_per_rack} hosts")
    print(f"    - Host Physical Limits : {parsed_args.cpu_per_host} CPU cores, {parsed_args.ram_per_host} GB RAM")
    print(f"    - Ambient Temp (CRAH)  : {parsed_args.ambient_temp:.1f} °C")
    print(f"    - Hotspot Threshold    : {parsed_args.hotspot_threshold:.1f} °C")
    print(f"    - Random Seed          : {parsed_args.seed} (deterministic arrival scheduling)")
    print(f"    - Placement Strategy   : {parsed_args.strategy}")
    if parsed_args.strategy == "q_learning":
        if parsed_args.train:
            mode = f"multi-episode training ({parsed_args.episodes if parsed_args.episodes > 1 else 500} episodes)"
        elif parsed_args.eval_mode:
            mode = "evaluation (greedy, no Q-table updates)"
        else:
            mode = "single-episode training (epsilon-greedy)"
        print(f"    - Q-Learning Mode      : {mode}")
    print("-" * 80)

    # Validate dataset configuration
    if parsed_args.dataset_dir is None:
        print(
            "\n[ERROR] No dataset directory specified.\n"
            "Please provide a path via --dataset-dir.\n"
            "Example: python main.py --dataset-dir dataset/bitbrains/fastStorage/2013-8",
            file=sys.stderr,
        )
        sys.exit(1)

    dataset_path = Path(parsed_args.dataset_dir)
    if not dataset_path.exists():
        print(f"\n[ERROR] Dataset directory not found: '{dataset_path}'", file=sys.stderr)
        sys.exit(1)
    if not dataset_path.is_dir():
        print(f"\n[ERROR] Path is not a directory: '{dataset_path}'", file=sys.stderr)
        sys.exit(1)

    # Multi-episode training path
    if parsed_args.train and parsed_args.strategy == "q_learning":
        from src.qlearning.trainer import QLearningTrainer, TrainingConfig

        num_episodes = parsed_args.episodes if parsed_args.episodes > 1 else 500
        train_cfg = TrainingConfig(
            episodes=num_episodes,
            learning_rate=parsed_args.learning_rate,
            discount_factor=parsed_args.discount_factor,
            initial_epsilon=parsed_args.initial_epsilon,
            min_epsilon=parsed_args.min_epsilon,
            epsilon_decay=parsed_args.epsilon_decay,
            seed=parsed_args.seed,
            dataset_dir=str(dataset_path),
            max_vms=parsed_args.max_vms,
            steps=parsed_args.steps,
            racks=parsed_args.racks,
            hosts_per_rack=parsed_args.hosts_per_rack,
            cpu_per_host=parsed_args.cpu_per_host,
            ram_per_host=parsed_args.ram_per_host,
            arrival_rate=parsed_args.arrival_rate,
            ambient_temp=parsed_args.ambient_temp,
            hotspot_threshold=parsed_args.hotspot_threshold,
            output_dir=parsed_args.output_dir,
            model_path=parsed_args.model_path,
            overwrite=parsed_args.overwrite,
        )
        trainer = QLearningTrainer(config=train_cfg)
        trainer.train()
        return

    # Load Bitbrains traces for single-run simulation
    print(f"\n[*] Loading Bitbrains traces from: {dataset_path}")
    try:
        from src.bitbrains_loader import BitbrainsLoader
        loader = BitbrainsLoader()
        max_traces = parsed_args.max_vms if parsed_args.max_vms > 0 else None
        report = loader.load_with_report(
            directory_path=dataset_path,
            max_traces=max_traces,
            recursive=True,
        )
    except Exception as exc:
        print(f"\n[ERROR] Failed to load dataset: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"    Loaded  : {report.total_loaded} VM traces")
    print(f"    Skipped : {report.total_skipped} files")

    if report.total_loaded == 0:
        print("\n[ERROR] No VM traces were loaded. Cannot run simulation.", file=sys.stderr)
        sys.exit(1)

    # Build workload schedule from traces
    gen = TraceWorkloadGenerator(
        traces=report.valid_traces,
        arrival_rate_lambda=parsed_args.arrival_rate,
        max_arrival_time=parsed_args.steps,
        seed=parsed_args.seed,
    )
    schedule = gen.generate_schedule_by_step(total_time_steps=parsed_args.steps)

    # Build data center
    thermal_config = ThermalConfig(
        ambient_temperature_c=parsed_args.ambient_temp,
        hotspot_threshold_c=parsed_args.hotspot_threshold,
    )
    dc = DataCenter.create_standard_datacenter(
        num_racks=parsed_args.racks,
        hosts_per_rack=parsed_args.hosts_per_rack,
        cpu_per_host=parsed_args.cpu_per_host,
        ram_per_host=parsed_args.ram_per_host,
        initial_temperature_c=parsed_args.ambient_temp,
    )
    thermal_model = ThermalModel(config=thermal_config)

    # Build placement strategy
    strategy = _build_strategy(parsed_args)

    # Run simulation
    sim = Simulator(datacenter=dc, thermal_model=thermal_model)
    print(f"\n[*] Running simulation with strategy: {strategy.name} ...")
    metrics = sim.run(
        workload_schedule=schedule,
        strategy=strategy,
        total_steps=parsed_args.steps,
    )

    summary = metrics.summary()
    print_summary(summary, strategy.name)

    # Q-Learning specific stats
    if parsed_args.strategy == "q_learning":
        from src.qlearning.strategy import QLearningSimulatorStrategy
        if isinstance(strategy, QLearningSimulatorStrategy):
            stats = strategy.stats
            print("\n[*] Q-Learning Statistics:")
            print(f"    Total placements     : {stats.total_placements}")
            print(f"    Total rejections     : {stats.total_rejections}")
            print(f"    Q-table entries      : {strategy.q_table_size}")
            print(f"    Total Q-updates      : {stats.total_q_updates}")
            print(f"    Cumulative reward    : {stats.cumulative_reward:.2f}")
            print(f"    Current epsilon      : {strategy.epsilon:.4f}")
            print("-" * 80)

    print("\n[OK] Simulation complete.\n")


if __name__ == "__main__":
    main()
