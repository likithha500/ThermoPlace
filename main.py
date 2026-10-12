"""Main entry point for ThermoPlace — Thermal-Aware VM Scheduling Simulator.

Milestone 1: Baseline First Fit evaluation under reproducible synthetic workload,
RC thermal dynamics, and capacity enforcement.
"""

import argparse
import sys

# Ensure UTF-8 stdout formatting for degree symbols on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.datacenter import DataCenter
from src.workload import WorkloadGenerator
from src.thermal import ThermalModel, ThermalConfig
from src.placement.first_fit import FirstFitPlacement
from src.simulator import Simulator


def parse_args():
    parser = argparse.ArgumentParser(
        description="ThermoPlace: Thermal-Aware VM Scheduling Simulator (Milestone 1: First Fit Baseline)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable workload generation (default: 42)",
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
    return parser.parse_args()


def print_banner():
    banner = r"""
================================================================================
  ThermoPlace: Thermal-Aware VM Scheduling Simulator (Milestone 1)
  Baseline Evaluation: Traditional First Fit Placement
================================================================================
    """
    print(banner)


def main():
    args = parse_args()
    print_banner()

    print("[*] Configuration:")
    print(f"    - Random Seed          : {args.seed} (deterministic reproducibility)")
    print(f"    - Total Time Steps     : {args.steps}")
    print(f"    - Data Center Topology : {args.racks} racks x {args.hosts_per_rack} hosts = {args.racks * args.hosts_per_rack} hosts")
    print(f"    - Host Capacity        : {args.cpu_per_host} CPU cores, {args.ram_per_host} GB RAM")
    print(f"    - Ambient Temp (CRAH)  : {args.ambient_temp:.1f} °C")
    print(f"    - Hotspot Threshold    : {args.hotspot_threshold:.1f} °C (simulated estimate)")
    print("-" * 80)

    # 1. Instantiate Data Center Topology
    datacenter = DataCenter.create_standard_datacenter(
        num_racks=args.racks,
        hosts_per_rack=args.hosts_per_rack,
        cpu_per_host=args.cpu_per_host,
        ram_per_host=args.ram_per_host,
        initial_temperature_c=args.ambient_temp,
    )

    # 2. Configure Thermal Model
    # Explicit scientific notice: Simulated approximations, not physical sensor readings
    thermal_config = ThermalConfig(
        ambient_temperature_c=args.ambient_temp,
        hotspot_threshold_c=args.hotspot_threshold,
    )
    thermal_model = ThermalModel(config=thermal_config)

    # 3. Generate Repeatable Synthetic Workload
    print(f"[*] Generating synthetic workload trace with seed={args.seed}...")
    workload_gen = WorkloadGenerator(
        seed=args.seed,
        total_time_steps=args.steps,
        arrival_rate_lambda=2.0,
    )
    schedule = workload_gen.generate_schedule_by_step()
    total_vms = sum(len(vms) for vms in schedule.values())
    print(f"    -> Generated {total_vms} VM arrival requests across {args.steps} steps.")

    # 4. Instantiate First Fit Placement Strategy & Simulator
    first_fit = FirstFitPlacement()
    simulator = Simulator(
        datacenter=datacenter,
        thermal_model=thermal_model,
        thermal_config=thermal_config,
    )

    print("\n[*] Executing Baseline Simulation: First Fit Placement...")
    metrics = simulator.run(
        workload_schedule=schedule,
        strategy=first_fit,
        total_steps=args.steps,
    )

    # 5. Display Comprehensive Summary
    summary = metrics.summary()
    print("\n" + "=" * 80)
    print("  SIMULATION RESULTS SUMMARY: FIRST FIT BASELINE")
    print("=" * 80)
    print(f"  Placement Algorithm             : {summary['algorithm']}")
    print(f"  Total Arrived VMs               : {summary['total_arrived_vms']}")
    print(f"  Total Placed VMs                : {summary['total_placed_vms']}")
    print(f"  Total Rejected VMs              : {summary['total_rejected_vms']}")
    print(f"  Placement Acceptance Rate       : {summary['acceptance_rate'] * 100:.2f}%")
    print(f"  Mean CPU Utilization            : {summary['average_cpu_utilization'] * 100:.2f}%")
    print(f"  Mean RAM Utilization            : {summary['average_ram_utilization'] * 100:.2f}%")
    print(f"  Average Power Consumption       : {summary['average_total_power_w']:.2f} W")
    print("-" * 80)
    print("  THERMAL TELEMETRY (Simulated Estimates):")
    print(f"  Peak Host Temperature          : {summary['peak_temperature_c']:.2f} °C")
    print(f"  Average Max Temperature        : {summary['average_max_temperature_c']:.2f} °C")
    print(f"  Cumulative Hotspot Events (>=75°C): {summary['cumulative_hotspot_events']}")
    print(f"  Time Steps with >=1 Hotspot     : {summary['steps_with_hotspots']} / {args.steps}")
    print(f"  Average Thermal Imbalance (Std) : {summary['average_thermal_imbalance_std']:.2f} °C")
    print(f"  Peak Thermal Imbalance (Std)    : {summary['peak_thermal_imbalance_std']:.2f} °C")
    print("=" * 80)

    # 6. Verify Reset Capability
    print("\n[*] Verifying data center reset mechanism for future Milestone 2 benchmark...")
    simulator.reset(initial_temp=args.ambient_temp)
    post_reset_hosts = [h.used_cpu for h in simulator.datacenter.hosts]
    post_reset_temps = [h.simulated_temperature_c for h in simulator.datacenter.hosts]
    assert all(c == 0 for c in post_reset_hosts), "Error: Hosts not cleared during reset!"
    assert all(abs(t - args.ambient_temp) < 1e-4 for t in post_reset_temps), "Error: Temps not reset!"
    print("    -> Data center reset verified successfully (all hosts zeroed, temperatures restored).")
    print("    -> Ready for Milestone 2: Q-Learning implementation and fair comparative evaluation.\n")


if __name__ == "__main__":
    main()
