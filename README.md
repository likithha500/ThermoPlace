# ThermoPlace — Thermal-Aware VM Scheduling Simulator

A modular research simulation framework designed to evaluate and compare virtual machine (VM) scheduling algorithms in data centers with explicit thermal modeling.

---

## 🎯 Project Objectives

1. **Compare Heuristic vs. Reinforcement Learning Placement**: Evaluate a traditional baseline (First Fit) against an intelligent thermal-aware tabular Q-Learning scheduler.
2. **Strict Fair Comparison**: Benchmark algorithms under **identical workloads and initial physical data center states** using reproducible random seeds.
3. **Thermal & Operational Metrics**: Quantify performance via:
   - **Hotspot count** (number of thermal threshold breaches)
   - **Maximum host temperature** ($T_{\max}$)
   - **Thermal imbalance** (spatial standard deviation of temperatures across hosts)
   - **Resource utilization & VM rejection count**
4. **Student-Friendly Architecture**: Designed for a collaborative team of three students with clear module boundaries, object-oriented datacenter entities, and extensive documentation.

---

## 🏗️ Project Architecture & File Structure

```text
ThermoPlace/
├── README.md                      # Project documentation and guide
├── requirements.txt               # Dependencies (NumPy, Pandas, Matplotlib)
├── main.py                        # Main CLI entry point to run simulations
├── src/
│   ├── __init__.py                # Package initialization
│   ├── dataset_loader.py          # Reusable VM trace structures and reporting (dataset pending)
│   ├── workload.py                # VM model, synthetic & trace-driven workload generators
│   ├── datacenter.py              # Physical datacenter model (Host, Rack, DataCenter)
│   ├── thermal.py                 # Thermal model (power-to-heat, RC inertia, recirculation)
│   ├── placement/
│   │   ├── __init__.py
│   │   ├── base.py                # Base placement strategy interface
│   │   └── first_fit.py           # Traditional First Fit placement algorithm
│   ├── qlearning/
│   │   ├── __init__.py
│   │   ├── agent.py               # Tabular Q-Learning agent (Phase 1)
│   │   └── reward.py              # Thermal-aware reward function calculator (Phase 2)
│   ├── metrics.py                 # Telemetry & metric collection (hotspots, imbalance, logs)
│   └── simulator.py               # Discrete-time simulation engine and execution loop
└── tests/
    ├── __init__.py
    ├── test_dataset_loader.py     # Unit tests for VMTrace and report data structures
    ├── test_workload.py           # Unit tests for workload determinism, lifecycle, and traces
    ├── test_datacenter.py         # Unit tests for host capacity, allocation, and resets
    ├── test_thermal.py            # Unit tests for thermal inertia and power models
    ├── test_first_fit.py          # Unit tests for First Fit placement decisions
    ├── test_metrics.py            # Unit tests for metric calculation accuracy
    ├── test_simulator.py          # End-to-end integration tests for simulation runs
    ├── test_main.py               # Unit tests for CLI parsing and dataset validation
    ├── test_qlearning.py          # Unit tests for Tabular Q-Learning agent
    └── test_reward.py             # Unit tests for thermal reward function and decomposition
```

---

## 🔬 Thermal Modeling: Simulated Estimates vs. Real Measurements

> **Important Scientific Distinction:**
> Real data centers rely on physical thermistors, server BMCs, ambient temperature sensors, and complex 3D Computational Fluid Dynamics (CFD). 
> In **ThermoPlace**, all temperature values are **simulated estimates** derived from analytical RC thermal equations and empirical linear power dissipation models widely used in research literature (e.g., Tang et al., Moore et al.):
>
> 1. **Server Power Dissipation**:
>    $$P_i(t) = P_{\text{idle}} + (P_{\text{max}} - P_{\text{idle}}) \times u_i(t)$$
>    where $u_i(t) = \frac{\text{used\_cpu}}{\text{total\_cpu}} \in [0, 1]$ represents the host's current CPU utilization ratio.
>
> 2. **Rack Inlet & Recirculation Approximations**:
>    $$T_{\text{inlet}, i}(t) = T_{\text{ambient}} + \alpha \cdot P_{\text{rack}}(t)$$
>    Accounts for hot-air recirculation effects where servers located higher up or in densely loaded racks ingest warmer air.
>
> 3. **Thermal Inertia (RC Model)**:
>    Servers do not instantly heat up or cool down. Temperature dynamics follow Newton's Law of Cooling / lumped RC capacitance:
>    $$T_i(t + \Delta t) = T_i(t) + \frac{\Delta t}{\tau} \left( T_{\text{steady-state}, i}(t) - T_i(t) \right)$$
>    where $T_{\text{steady-state}, i} = T_{\text{inlet}, i} + R_{\text{th}} \cdot P_i(t)$, and $\tau = R_{\text{th}} \cdot C_{\text{th}}$ is the thermal time constant.
>
> 4. **Hotspot Definition**:
>    Any host whose simulated temperature $T_i(t) \ge T_{\text{threshold}}$ (default: $75.0^\circ\text{C}$).

---

## 👥 Student Team Responsibilities (Team of 3)

| Member | Primary Focus Area | Key Modules |
| :--- | :--- | :--- |
| **Student 1: Workload & Infrastructure** | Physical topology, VM constraints, workload generation, reset mechanisms | `src/workload.py`, `src/datacenter.py` |
| **Student 2: Thermal Modeling & Baseline** | Power-to-temperature physics, thermal inertia, First Fit scheduler | `src/thermal.py`, `src/placement/` |
| **Student 3: RL & Evaluation (Milestone 2)** | Tabular Q-Learning policy, reward engineering, metric reporting | `src/qlearning/`, `src/metrics.py`, `src/simulator.py` |

---

## 🚀 Getting Started

### 1. Requirements & Setup
The project requires Python 3.10+ and standard numerical libraries:
```bash
pip install -r requirements.txt
```

### 2. Dataset Setup
The PlanetLab dataset integration has been removed from ThermoPlace. Currently:
- No dataset is installed in `data/raw/`.
- No replacement dataset has been integrated yet.
- Dataset integration is pending (e.g. integration of the Bitbrains GWA-T-12 dataset is planned as a separate future task).

The codebase maintains reusable trace data structures (`VMTrace`, `DatasetLoadReport` in [`src/dataset_loader.py`](file:///c:/Users/likitha%20s/ThermoPlace/src/dataset_loader.py)) and a generic trace workload generator (`TraceWorkloadGenerator` in [`src/workload.py`](file:///c:/Users/likitha%20s/ThermoPlace/src/workload.py)) ready for future dataset integration.

### 3. Running the Simulation
Because the former dataset has been removed and replacement dataset integration is pending, running `main.py` directly without a configured dataset loader will inform the user:
```bash
python main.py
```
This produces an actionable message indicating that no dataset is currently configured.

CLI options available:
```bash
# Available CLI options:
#   --dataset-dir PATH        Path to dataset directory (integration pending)
#   --max-vms N               Maximum VM traces to load (default: 50; 0 for all)
#   --steps N                 Total simulation steps to run (default: 100)
#   --racks N                 Number of server racks (default: 2)
#   --hosts-per-rack N        Number of hosts per rack (default: 4)
#   --cpu-per-host N          Host CPU capacity in cores (default: 16)
#   --ram-per-host N          Host RAM capacity in GB (default: 64)
#   --arrival-rate FLOAT      Mean arrivals per step (default: 2.0; 0 for all at t=0)
#   --seed INT                Deterministic seed for arrival & flavor assignment (default: 42)
#   --ambient-temp FLOAT      Ambient CRAH temperature in °C (default: 25.0)
#   --hotspot-threshold FLOAT Critical hotspot temperature in °C (default: 75.0)
```

### 4. Running Unit Tests
All unit tests across the test suite can be executed with:
```bash
python -m unittest discover -s tests -v
```

---

## 📌 Implementation Status

- [x] **Core Simulation Architecture**:
  - Object-oriented Host, Rack, and DataCenter models ([`src/datacenter.py`](file:///c:/Users/likitha%20s/ThermoPlace/src/datacenter.py)) enforcing CPU and RAM capacity limits while tracking dynamic utilized CPU.
  - Analytical thermal model ([`src/thermal.py`](file:///c:/Users/likitha%20s/ThermoPlace/src/thermal.py)) driving power dissipation, lumped RC thermal inertia, and rack recirculation.
  - Baseline First Fit VM placement scheduler ([`src/placement/first_fit.py`](file:///c:/Users/likitha%20s/ThermoPlace/src/placement/first_fit.py)).
  - Discrete-time simulation engine ([`src/simulator.py`](file:///c:/Users/likitha%20s/ThermoPlace/src/simulator.py)) and telemetry reporting ([`src/metrics.py`](file:///c:/Users/likitha%20s/ThermoPlace/src/metrics.py)).
  - Reusable trace structures ([`src/dataset_loader.py`](file:///c:/Users/likitha%20s/ThermoPlace/src/dataset_loader.py)) and trace-driven workload models ([`src/workload.py`](file:///c:/Users/likitha%20s/ThermoPlace/src/workload.py)).
  - CLI runner ([`main.py`](file:///c:/Users/likitha%20s/ThermoPlace/main.py)) with topology and thermal parameter configuration.
- [ ] **Workload Dataset Integration (Pending)**:
  - PlanetLab dataset and dependencies removed.
  - Integration of replacement workload dataset (e.g., Bitbrains GWA-T-12) to be implemented in a subsequent task.
- [x] **Milestone 2 (Reinforcement Learning Scheduler)**:
  - [x] **Phase 1**: Tabular Q-Learning placement agent with sparse dictionary Q-table, capacity filtering, epsilon-greedy action selection, and geometric decay ([`src/qlearning/agent.py`](file:///c:/Users/likitha%20s/ThermoPlace/src/qlearning/agent.py)).
  - [x] **Phase 2**: State representation discretization (VM demands, host utilization, 4-tier host temperature, 3-tier datacenter thermal condition) and modular thermal reward function calculator ([`src/qlearning/reward.py`](file:///c:/Users/likitha%20s/ThermoPlace/src/qlearning/reward.py)) penalizing temperature elevation, hotspot creation, and thermal variance.
  - [ ] **Phase 3 (Future)**: Simulator training loop, comparative benchmark (First Fit vs. Q-Learning under identical workload traces), and comparative plotting.
