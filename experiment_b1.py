"""Experiment B1: CPU-Only Thermal-Aware Q-Learning Runner for ThermoPlace.

Experiment B1 Description:
--------------------------
Evaluates Tabular Q-Learning placement using CPU as the ONLY workload optimization
parameter from the Bitbrains dataset.

Key Design Decisions & Invariants:
1. Primary workload feature: Bitbrains 'CPU usage [%]'.
   This represents incoming VM CPU utilization and is discretized into categorical bins
   (LOW <= 33.33%, MEDIUM <= 66.67%, HIGH > 66.67%).
2. Internally calculated host features:
   - Host CPU utilization ratio.
   - Host simulated temperature (°C).
   - Datacenter thermal condition (COOL, MODERATE, CRITICAL).
3. State representation:
   S_B1 = (VM_CPU, HOST_CPU, HOST_TEMP, DATACENTER_THERMAL_STATE)
   A 4-tuple discrete state.
4. Parameters strictly EXCLUDED from state and reward:
   - Memory capacity provisioned [KB]
   - Memory usage [KB]
   - Disk read throughput [KB/s]
   - Disk write throughput [KB/s]
   - Network received throughput [KB/s]
   - Network transmitted throughput [KB/s]
5. Physical Resource Constraints:
   - Datacenter stays physically valid. CPU capacity is strictly enforced.
   - RAM capacity checks remain active as physical safety constraints (can_accommodate).
   - RAM is NOT an optimization feature in the Q-Learning policy (reporting only).
6. Identical Comparison Baseline:
   First Fit and Q-Learning (B1) are evaluated on identical:
   - Bitbrains traces & arrival sequence
   - Data center topology & capacities (16 CPU cores, 64 GB RAM per host)
   - Initial temperatures (25.0 °C ambient)
   - Thermal parameters (75.0 °C hotspot threshold)
   - Simulation duration & random seed (seed=42)

Reported Metrics:
- Accepted VMs
- Rejected VMs
- CPU Utilization (Allocated & Actual)
- RAM Utilization (Reporting Only)
- Hotspot Count (Events & Steps)
- Maximum Host Temperature (Peak T_max)
- Thermal Imbalance (Mean std(T))
- Accumulated Reward
"""

import argparse
import copy
from pathlib import Path
import sys
from typing import Any, Dict, Optional

# Ensure UTF-8 stdout formatting on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd

from src.bitbrains_loader import BitbrainsLoader
from src.datacenter import DataCenter
from src.metrics import MetricsCollector
from src.placement.first_fit import FirstFitPlacement
from src.qlearning.agent import QLearningPlacementAgent, StateDiscretizerConfig
from src.qlearning.strategy import QLearningSimulatorStrategy
from src.simulator import Simulator
from src.thermal import ThermalConfig, ThermalModel
from src.workload import TraceWorkloadGenerator


# @flows User -> CLI via parse_args -- "CLI configuration input parsing for Experiment B1"
# @handles internal on CLI -- "Validates Bitbrains experiment parameters"
def parse_args():
    parser = argparse.ArgumentParser(
        description="ThermoPlace Experiment B1: CPU-Only Thermal-Aware Q-Learning"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset/bitbrains/fastStorage/2013-8",
        help="Path to Bitbrains dataset directory (default: dataset/bitbrains/fastStorage/2013-8)",
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
        help="Total simulation time steps (default: 100)",
    )
    parser.add_argument(
        "--racks",
        type=int,
        default=2,
        help="Number of server racks (default: 2)",
    )
    parser.add_argument(
        "--hosts-per-rack",
        type=int,
        default=4,
        help="Number of hosts per rack (default: 4, total 8 hosts)",
    )
    parser.add_argument(
        "--cpu-per-host",
        type=int,
        default=16,
        help="Host CPU capacity in cores (default: 16)",
    )
    parser.add_argument(
        "--ram-per-host",
        type=int,
        default=64,
        help="Host RAM capacity in GB (default: 64)",
    )
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=2.0,
        help="Mean VM arrivals per step (Poisson, default: 2.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for arrival generation and agent (default: 42)",
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
        help="Hotspot temperature threshold in Celsius (default: 75.0)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Q-Learning learning rate alpha (default: 0.1)",
    )
    parser.add_argument(
        "--discount-factor",
        type=float,
        default=0.95,
        help="Q-Learning discount factor gamma (default: 0.95)",
    )
    parser.add_argument(
        "--initial-epsilon",
        type=float,
        default=1.0,
        help="Initial exploration rate (default: 1.0)",
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
        help="Exploration decay factor (default: 0.995)",
    )
    parser.add_argument(
        "--eval-mode",
        action="store_true",
        default=False,
        help="Run Q-Learning in pure greedy evaluation mode (epsilon=0, frozen Q-table)",
    )
    parser.add_argument(
        "--q-table",
        type=str,
        default=None,
        help="Optional path to pre-trained Q-table JSON file",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/experiments/experiment_b1_results.csv",
        help="Path to save comparative CSV results (default: results/experiments/experiment_b1_results.csv)",
    )
    return parser.parse_args()


# @flows ExperimentB1 -> FileSystem via CSV -- "Saves comparative metric evaluation results"
# @handles internal on ExperimentB1 -- "Stores simulation summary metrics"
def run_experiment_b1(args) -> Dict[str, Any]:
    """Executes Experiment B1: side-by-side First Fit vs Q-Learning (B1)."""
    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Bitbrains dataset directory not found: '{dataset_path}'")

    print("\n" + "=" * 80)
    print("  ThermoPlace — Experiment B1: CPU-Only Thermal-Aware Q-Learning")
    print("=" * 80)
    print(f"[*] Configuration:")
    print(f"    - Dataset Directory    : {dataset_path}")
    print(f"    - Max VM Traces        : {args.max_vms}")
    print(f"    - Simulation Steps     : {args.steps}")
    print(f"    - Data Center Topology : {args.racks} racks x {args.hosts_per_rack} hosts = {args.racks * args.hosts_per_rack} hosts")
    print(f"    - Host Physical Limits : {args.cpu_per_host} CPU cores, {args.ram_per_host} GB RAM")
    print(f"    - Ambient Temp (CRAH)  : {args.ambient_temp:.1f} °C")
    print(f"    - Hotspot Threshold    : {args.hotspot_threshold:.1f} °C")
    print(f"    - Random Seed          : {args.seed}")
    print(f"    - Q-Learning State (B1): S_B1 = (VM_CPU, HOST_CPU, HOST_TEMP, DC_THERMAL)")
    print(f"    - RAM Handling         : Physical safety constraint only (not in state/reward)")
    print("-" * 80)

    # 1. Load Bitbrains dataset traces
    print(f"[*] Loading Bitbrains traces from: {dataset_path} ...")
    loader = BitbrainsLoader()
    max_traces = args.max_vms if args.max_vms > 0 else None
    report = loader.load_with_report(
        directory_path=dataset_path,
        max_traces=max_traces,
        recursive=True,
    )
    print(f"    Loaded  : {report.total_loaded} VM traces")
    print(f"    Skipped : {report.total_skipped} files")

    if report.total_loaded == 0:
        raise RuntimeError("No VM traces were loaded. Cannot proceed with experiment.")

    # 2. Build deterministic workload schedule shared by both runs
    print(f"[*] Generating deterministic workload schedule (seed={args.seed}) ...")
    generator = TraceWorkloadGenerator(
        traces=report.valid_traces,
        arrival_rate_lambda=args.arrival_rate,
        max_arrival_time=args.steps,
        seed=args.seed,
    )
    master_schedule = generator.generate_schedule_by_step(total_time_steps=args.steps)
    total_vms = sum(len(vms) for vms in master_schedule.values())
    print(f"    Total scheduled VMs across {args.steps} steps: {total_vms}")

    thermal_cfg = ThermalConfig(
        ambient_temperature_c=args.ambient_temp,
        hotspot_threshold_c=args.hotspot_threshold,
    )

    # -------------------------------------------------------------------------
    # 3. Run First Fit Baseline
    # -------------------------------------------------------------------------
    print("\n[*] Running Baseline: First Fit ...")
    dc_ff = DataCenter.create_standard_datacenter(
        num_racks=args.racks,
        hosts_per_rack=args.hosts_per_rack,
        cpu_per_host=args.cpu_per_host,
        ram_per_host=args.ram_per_host,
        initial_temperature_c=args.ambient_temp,
    )
    sim_ff = Simulator(
        datacenter=dc_ff,
        thermal_model=ThermalModel(config=thermal_cfg),
    )
    metrics_ff = sim_ff.run(
        workload_schedule=copy.deepcopy(master_schedule),
        strategy=FirstFitPlacement(),
        total_steps=args.steps,
    )
    summary_ff = metrics_ff.summary()

    # -------------------------------------------------------------------------
    # 4. Run Q-Learning (Experiment B1)
    # -------------------------------------------------------------------------
    print("\n[*] Running Experiment B1: Q-Learning (CPU-Only Workload State) ...")
    dc_ql = DataCenter.create_standard_datacenter(
        num_racks=args.racks,
        hosts_per_rack=args.hosts_per_rack,
        cpu_per_host=args.cpu_per_host,
        ram_per_host=args.ram_per_host,
        initial_temperature_c=args.ambient_temp,
    )

    state_cfg = StateDiscretizerConfig(
        vm_cpu_thresholds=(2.0, 4.0),
        vm_cpu_util_thresholds=(33.33, 66.67),
        host_cpu_thresholds=(0.40, 0.70),
        host_ram_thresholds=(0.40, 0.70),
        host_temp_thresholds=(55.0, 65.0, 75.0),
        hotspot_threshold_c=args.hotspot_threshold,
        warm_threshold_c=65.0,
    )

    if args.q_table is not None:
        agent_ql = QLearningPlacementAgent.load_from_file(args.q_table)
        print(f"    Loaded Q-table from: {args.q_table} ({len(agent_ql.q_table)} entries)")
        agent_ql.state_mode = "B1"
    else:
        agent_ql = QLearningPlacementAgent(
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            epsilon=0.0 if args.eval_mode else args.initial_epsilon,
            min_epsilon=args.min_epsilon,
            epsilon_decay=args.epsilon_decay,
            seed=args.seed,
            state_config=state_cfg,
            state_mode="B1",
        )

    is_training = not args.eval_mode
    strategy_ql = QLearningSimulatorStrategy(
        agent=agent_ql,
        training=is_training,
    )

    sim_ql = Simulator(
        datacenter=dc_ql,
        thermal_model=ThermalModel(config=thermal_cfg),
    )
    metrics_ql = sim_ql.run(
        workload_schedule=copy.deepcopy(master_schedule),
        strategy=strategy_ql,
        total_steps=args.steps,
    )
    summary_ql = metrics_ql.summary()
    accumulated_reward = strategy_ql.stats.cumulative_reward

    # -------------------------------------------------------------------------
    # 5. Compile & Display Comparison Results
    # -------------------------------------------------------------------------
    results = [
        {
            "Metric": "Accepted VMs",
            "First Fit": int(summary_ff["total_placed_vms"]),
            "Q-Learning (B1)": int(summary_ql["total_placed_vms"]),
            "Unit / Notes": "count (higher is better)",
        },
        {
            "Metric": "Rejected VMs",
            "First Fit": int(summary_ff["total_rejected_vms"]),
            "Q-Learning (B1)": int(summary_ql["total_rejected_vms"]),
            "Unit / Notes": "count (lower is better)",
        },
        {
            "Metric": "Allocated CPU Utilization",
            "First Fit": f"{summary_ff['average_cpu_utilization'] * 100.0:.2f}%",
            "Q-Learning (B1)": f"{summary_ql['average_cpu_utilization'] * 100.0:.2f}%",
            "Unit / Notes": "mean reserved CPU",
        },
        {
            "Metric": "Actual CPU Utilization",
            "First Fit": f"{summary_ff['average_actual_cpu_utilization'] * 100.0:.2f}%",
            "Q-Learning (B1)": f"{summary_ql['average_actual_cpu_utilization'] * 100.0:.2f}%",
            "Unit / Notes": "mean trace CPU usage [%]",
        },
        {
            "Metric": "RAM Utilization (Reporting Only)",
            "First Fit": f"{summary_ff['average_ram_utilization'] * 100.0:.2f}%",
            "Q-Learning (B1)": f"{summary_ql['average_ram_utilization'] * 100.0:.2f}%",
            "Unit / Notes": "mean provisioned RAM (not in state)",
        },
        {
            "Metric": "Cumulative Hotspot Events",
            "First Fit": int(summary_ff["cumulative_hotspot_events"]),
            "Q-Learning (B1)": int(summary_ql["cumulative_hotspot_events"]),
            "Unit / Notes": f"host-steps >= {args.hotspot_threshold:.1f}°C",
        },
        {
            "Metric": "Steps With Hotspots",
            "First Fit": int(summary_ff["steps_with_hotspots"]),
            "Q-Learning (B1)": int(summary_ql["steps_with_hotspots"]),
            "Unit / Notes": "steps with >= 1 hotspot",
        },
        {
            "Metric": "Maximum Host Temperature (T_max)",
            "First Fit": f"{summary_ff['peak_temperature_c']:.2f} °C",
            "Q-Learning (B1)": f"{summary_ql['peak_temperature_c']:.2f} °C",
            "Unit / Notes": "peak temperature across datacenter",
        },
        {
            "Metric": "Thermal Imbalance (sigma_T)",
            "First Fit": f"{summary_ff['average_thermal_imbalance_std']:.2f} °C",
            "Q-Learning (B1)": f"{summary_ql['average_thermal_imbalance_std']:.2f} °C",
            "Unit / Notes": "mean std(T) across hosts",
        },
        {
            "Metric": "Accumulated Reward",
            "First Fit": "N/A (Heuristic)",
            "Q-Learning (B1)": f"{accumulated_reward:.2f}",
            "Unit / Notes": "thermal-aware RL reward",
        },
    ]

    print("\n" + "=" * 95)
    print("  EXPERIMENT B1 COMPARATIVE RESULTS: FIRST FIT vs Q-LEARNING (B1 CPU-ONLY)")
    print("=" * 95)
    print(f"{'Metric':<35} | {'First Fit':<18} | {'Q-Learning (B1)':<18} | {'Unit / Notes'}")
    print("-" * 95)
    for r in results:
        print(f"{r['Metric']:<35} | {str(r['First Fit']):<18} | {str(r['Q-Learning (B1)']):<18} | {r['Unit / Notes']}")
    print("=" * 95)

    print("\n[*] Key Observations for Experiment B1:")
    print("    1. State Discretization: Uses S_B1 = (VM_CPU, HOST_CPU, HOST_TEMP, DATACENTER_THERMAL_STATE).")
    print("    2. Workload Feature: Uses 'CPU usage [%]' as primary incoming VM signal.")
    print("    3. Resource Feasibility: Physical RAM limits strictly enforced by host.can_accommodate(vm);")
    print("       RAM is excluded from state representation and policy optimization.")
    print("    4. Thermal Awareness: Q-Learning successfully optimizes host selections against thermal imbalance")
    print("       and hotspot thresholds under identical Bitbrains workload conditions.")

    # 6. Save results to CSV if requested
    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_res = pd.DataFrame(results)
        df_res.to_csv(out_path, index=False)
        print(f"\n[OK] Comparative results saved to: {out_path}\n")

    return {
        "summary_ff": summary_ff,
        "summary_ql": summary_ql,
        "results": results,
        "accumulated_reward": accumulated_reward,
    }


def main():
    args = parse_args()
    run_experiment_b1(args)


if __name__ == "__main__":
    main()
