"""Main entry point for ThermoPlace — Thermal-Aware VM Scheduling Simulator.

Provides command-line interface for running thermal-aware VM placement simulations.
Note: Dataset integration is currently pending (previous PlanetLab integration has been removed).
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


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="ThermoPlace: Thermal-Aware VM Scheduling Simulator"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to dataset directory (no dataset is currently configured; integration pending)",
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
    return parser.parse_args(args)


def print_banner():
    banner = r"""
================================================================================
  ThermoPlace: Thermal-Aware VM Scheduling Simulator
================================================================================
    """
    print(banner)


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
    print("-" * 80)

    # Validate dataset configuration
    if parsed_args.dataset_dir is None:
        print(
            "\n[ERROR] No dataset is currently configured.\n"
            "The previous PlanetLab dataset integration has been removed, and the replacement\n"
            "workload dataset integration is pending.\n"
            "Please specify a valid dataset path via --dataset-dir once a dataset loader is configured.",
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

    print(
        f"\n[ERROR] No dataset loader is currently configured to load traces from '{dataset_path}'.\n"
        "The previous PlanetLab dataset integration has been removed, and dataset integration is pending.\n"
        "Please configure a valid dataset loader before running the simulation.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
