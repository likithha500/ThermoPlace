"""Main entry point for ThermoPlace — Thermal-Aware VM Scheduling Simulator.

Step 7: PlanetLab trace-driven baseline evaluation.
Replaces synthetic workload generation with real PlanetLab CPU-utilization traces.
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
from src.dataset_loader import PlanetLabDatasetLoader
from src.workload import PlanetLabWorkloadGenerator
from src.thermal import ThermalModel, ThermalConfig
from src.placement.first_fit import FirstFitPlacement
from src.simulator import Simulator


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="ThermoPlace: Thermal-Aware VM Scheduling Simulator (PlanetLab Trace-Driven Baseline)"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/raw/planetlab",
        help="Path to PlanetLab dataset directory or date subdirectory (default: data/raw/planetlab)",
    )
    parser.add_argument(
        "--max-vms",
        type=int,
        default=50,
        help="Maximum number of PlanetLab VM traces to load (default: 50; 0 for all)",
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
    return parser.parse_args(args)


def print_banner():
    banner = r"""
================================================================================
  ThermoPlace: Thermal-Aware VM Scheduling Simulator
  Baseline Evaluation: PlanetLab Trace-Driven First Fit Placement
================================================================================
    """
    print(banner)


def main(args=None):
    parsed_args = parse_args(args)
    print_banner()

    dataset_path = Path(parsed_args.dataset_dir)
    print("[*] Simulation Configuration:")
    print(f"    - Dataset Directory    : {dataset_path}")
    print(f"    - Max VM Traces        : {parsed_args.max_vms if parsed_args.max_vms > 0 else 'All'}")
    print(f"    - Total Time Steps     : {parsed_args.steps}")
    print(f"    - Data Center Topology : {parsed_args.racks} racks x {parsed_args.hosts_per_rack} hosts = {parsed_args.racks * parsed_args.hosts_per_rack} hosts")
    print(f"    - Host Physical Limits : {parsed_args.cpu_per_host} CPU cores, {parsed_args.ram_per_host} GB RAM")
    print(f"    - Ambient Temp (CRAH)  : {parsed_args.ambient_temp:.1f} °C")
    print(f"    - Hotspot Threshold    : {parsed_args.hotspot_threshold:.1f} °C")
    print(f"    - Random Seed          : {parsed_args.seed} (deterministic arrival scheduling)")
    print("-" * 80)

    # 1. Validate Dataset Directory & Load Traces
    if not dataset_path.exists():
        print(f"\n[ERROR] Dataset directory not found: '{dataset_path}'", file=sys.stderr)
        sys.exit(1)
    if not dataset_path.is_dir():
        print(f"\n[ERROR] Path is not a directory: '{dataset_path}'", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Discovering and loading real PlanetLab traces from: {dataset_path}...")
    loader = PlanetLabDatasetLoader()
    max_files = parsed_args.max_vms if (parsed_args.max_vms and parsed_args.max_vms > 0) else None

    direct_files = [f for f in dataset_path.iterdir() if f.is_file() and not f.name.startswith(".")]
    use_recursive = len(direct_files) == 0

    try:
        report = loader.load_with_report(dataset_path, max_files=max_files, recursive=use_recursive)
    except ValueError as e:
        print(f"\n[ERROR] Failed to load PlanetLab traces: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"    -> Successfully discovered and loaded {report.total_loaded} valid PlanetLab VM traces.")
    if report.total_skipped > 0:
        print(f"    -> Skipped {report.total_skipped} invalid/corrupted files.")

    # 2. Build Deterministic Simulation Schedule from Real Traces
    print("\n[*] Building trace-driven workload schedule:")
    print("    - Workload Source      : Real PlanetLab CPU utilization traces")
    print("    - CPU Model            : Reserved placement cores (1, 2, 4, 8) vs actual trace usage")
    print("    - RAM Model            : Provisioned placement capacity (2, 4, 8, 16 GB); no fabricated RAM")
    print("    - Arrival Timing       : Stochastically scheduled across steps via Poisson process")

    workload_gen = PlanetLabWorkloadGenerator(
        traces=report.valid_traces,
        arrival_rate_lambda=parsed_args.arrival_rate,
        max_arrival_time=min(parsed_args.steps - 1, 50),
        seed=parsed_args.seed,
    )
    schedule = workload_gen.generate_schedule_by_step(total_time_steps=parsed_args.steps)
    total_scheduled_vms = sum(len(vms) for vms in schedule.values())
    print(f"    -> Scheduled {total_scheduled_vms} VMs across {parsed_args.steps} simulation steps.")

    # 3. Instantiate Data Center Topology
    datacenter = DataCenter.create_standard_datacenter(
        num_racks=parsed_args.racks,
        hosts_per_rack=parsed_args.hosts_per_rack,
        cpu_per_host=parsed_args.cpu_per_host,
        ram_per_host=parsed_args.ram_per_host,
        initial_temperature_c=parsed_args.ambient_temp,
    )

    # 4. Configure Thermal Model
    thermal_config = ThermalConfig(
        ambient_temperature_c=parsed_args.ambient_temp,
        hotspot_threshold_c=parsed_args.hotspot_threshold,
    )
    thermal_model = ThermalModel(config=thermal_config)

    # 5. Execute Simulation: First Fit Placement under Trace-Driven CPU Utilization
    first_fit = FirstFitPlacement()
    simulator = Simulator(
        datacenter=datacenter,
        thermal_model=thermal_model,
        thermal_config=thermal_config,
    )

    print("\n[*] Executing Baseline Simulation (First Fit with PlanetLab Workload)...")
    metrics = simulator.run(
        workload_schedule=schedule,
        strategy=first_fit,
        total_steps=parsed_args.steps,
    )

    # 6. Display Simulation Results Summary
    summary = metrics.summary()
    print("\n" + "=" * 80)
    print("  SIMULATION RESULTS: FIRST FIT (PLANETLAB TRACE-DRIVEN)")
    print("=" * 80)
    print(f"  Placement Algorithm             : {summary['algorithm']}")
    print(f"  Total Arrived VMs               : {summary['total_arrived_vms']}")
    print(f"  Total Placed VMs                : {summary['total_placed_vms']}")
    print(f"  Total Rejected VMs              : {summary['total_rejected_vms']}")
    print(f"  Placement Acceptance Rate       : {summary['acceptance_rate'] * 100:.2f}%")
    print(f"  Mean Reserved CPU Allocation    : {summary['average_cpu_utilization'] * 100:.2f}%")
    print(f"  Mean Actual CPU Utilization     : {summary.get('average_actual_cpu_utilization', 0.0) * 100:.2f}%")
    print(f"  Mean RAM Allocation             : {summary['average_ram_utilization'] * 100:.2f}%")
    print(f"  Average Power Consumption       : {summary['average_total_power_w']:.2f} W")
    print("-" * 80)
    print("  THERMAL TELEMETRY (Simulated Estimates):")
    print(f"  Peak Host Temperature          : {summary['peak_temperature_c']:.2f} °C")
    print(f"  Average Max Temperature        : {summary['average_max_temperature_c']:.2f} °C")
    print(f"  Cumulative Hotspot Events (>=75°C): {summary['cumulative_hotspot_events']}")
    print(f"  Time Steps with >=1 Hotspot     : {summary['steps_with_hotspots']} / {parsed_args.steps}")
    print(f"  Average Thermal Imbalance (Std) : {summary['average_thermal_imbalance_std']:.2f} °C")
    print(f"  Peak Thermal Imbalance (Std)    : {summary['peak_thermal_imbalance_std']:.2f} °C")
    print("=" * 80)

    # 7. Verify Reset Mechanism for Future Benchmark
    print("\n[*] Verifying data center reset mechanism for future Milestone 2 benchmark...")
    simulator.reset(initial_temp=parsed_args.ambient_temp)
    post_reset_hosts = [h.used_cpu for h in simulator.datacenter.hosts]
    post_reset_actual = [h.actual_used_cpu for h in simulator.datacenter.hosts]
    post_reset_temps = [h.simulated_temperature_c for h in simulator.datacenter.hosts]
    assert all(c == 0 for c in post_reset_hosts), "Error: Reserved CPU not cleared during reset!"
    assert all(a == 0.0 for a in post_reset_actual), "Error: Actual CPU not cleared during reset!"
    assert all(abs(t - parsed_args.ambient_temp) < 1e-4 for t in post_reset_temps), "Error: Temps not reset!"
    print("    -> Data center reset verified successfully (hosts cleared, temperatures restored).\n")

    return summary


if __name__ == "__main__":
    main()
