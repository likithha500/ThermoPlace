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
├── dataset/
│   └── bitbrains/fastStorage/2013-8/   # Bitbrains GWA-T-12 VM trace CSVs
├── src/
│   ├── __init__.py                # Package initialization
│   ├── dataset_loader.py          # Reusable VM trace structures (VMTrace, DatasetLoadReport)
│   ├── bitbrains_loader.py        # Bitbrains GWA-T-12 CSV dataset loader
│   ├── workload.py                # VM model, synthetic & trace-driven workload generators
│   ├── datacenter.py              # Physical datacenter model (Host, Rack, DataCenter)
│   ├── thermal.py                 # Thermal model (power-to-heat, RC inertia, recirculation)
│   ├── placement/
│   │   ├── __init__.py
│   │   ├── base.py                # Base placement strategy interface (with lifecycle hooks)
│   │   └── first_fit.py           # Traditional First Fit placement algorithm
│   ├── qlearning/
│   │   ├── __init__.py
│   │   ├── agent.py               # Tabular Q-Learning agent (Phase 1)
│   │   ├── reward.py              # Thermal-aware reward function calculator (Phase 2)
│   │   └── strategy.py            # Q-Learning simulator strategy integration (Phase 3)
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
    ├── test_reward.py             # Unit tests for thermal reward function and decomposition
    └── test_phase3.py             # Integration tests for Q-Learning simulator integration
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
| **Student 3: RL & Evaluation (Milestone 2)** | Tabular Q-Learning policy, reward engineering, simulator integration | `src/qlearning/`, `src/metrics.py`, `src/simulator.py` |

---

## 🚀 Getting Started

### 1. Requirements & Setup
The project requires Python 3.10+ and standard numerical libraries:
```bash
pip install -r requirements.txt
```

### 2. Dataset Setup

The Bitbrains GWA-T-12 dataset is located at:
```
dataset/bitbrains/fastStorage/2013-8/
```
Each file (e.g. `1.csv`, `2.csv`, …) is a VM monitoring trace sampled at 5-minute intervals with columns including `CPU usage [%]`.

The dataset loader ([`src/bitbrains_loader.py`](src/bitbrains_loader.py)) reads these CSV files and converts them to `VMTrace` objects used by the `TraceWorkloadGenerator`.

### 3. Running the Simulation

#### Run with First Fit (default)
```bash
python main.py --dataset-dir dataset/bitbrains/fastStorage/2013-8 --strategy first_fit
```

#### Run with Q-Learning (training mode)
```bash
python main.py --dataset-dir dataset/bitbrains/fastStorage/2013-8 --strategy q_learning
```

#### Run with Q-Learning (evaluation mode — greedy, no Q-table updates)
```bash
python main.py --dataset-dir dataset/bitbrains/fastStorage/2013-8 --strategy q_learning --eval-mode
```

#### Full CLI reference
```
Options:
  --dataset-dir PATH        Path to Bitbrains CSV directory
  --max-vms N               Maximum VM traces to load (default: 50; 0 = all)
  --steps N                 Total simulation time steps (default: 100)
  --racks N                 Number of server racks (default: 2)
  --hosts-per-rack N        Number of hosts per rack (default: 4)
  --cpu-per-host N          Host CPU capacity in cores (default: 16)
  --ram-per-host N          Host RAM capacity in GB (default: 64)
  --arrival-rate FLOAT      Mean arrivals per step — Poisson (default: 2.0)
  --seed INT                Deterministic seed (default: 42)
  --ambient-temp FLOAT      Ambient CRAH temperature in °C (default: 25.0)
  --hotspot-threshold FLOAT Hotspot threshold in °C (default: 75.0)
  --strategy STR            Placement strategy: first_fit | q_learning (default: first_fit)
  --eval-mode               Q-Learning evaluation mode (greedy, no Q-updates)
```

### 4. Running Unit Tests

Run the complete test suite:
```bash
python -m unittest discover -s tests -v
```

Run only Phase 3 integration tests:
```bash
python -m unittest tests/test_phase3.py -v
```

---

## 🔄 Simulator Lifecycle (Phase 3)

The `Simulator.step()` method follows this lifecycle per time step:

```
1. Tick & release finished VMs
2. strategy.on_step_begin(...)       ← strategy caches arriving VMs
3. For each VM:
   a. strategy.select_host(vm, dc)  ← epsilon-greedy (or greedy in eval)
   b. host.allocate(vm)             ← CPU + RAM reserved
   c. strategy.on_placement_outcome(...)  ← records pending transition
4. thermal_model.update_datacenter(...)   ← one thermal update per step
5. strategy.on_step_end(...)         ← computes rewards + writes Q-updates
6. metrics.record_step(...)         ← telemetry snapshot
```

**Reward calculation** occurs in `on_step_end` so that it observes **post-thermal** temperatures — the physically correct moment to evaluate the thermal impact of a placement.

**Epsilon decay** is applied once in `on_simulation_end` — once per episode, not once per step.

---

## 📌 Implementation Status

- [x] **Core Simulation Architecture**:
  - Object-oriented Host, Rack, and DataCenter models ([`src/datacenter.py`](src/datacenter.py)) enforcing CPU and RAM capacity limits while tracking dynamic utilized CPU.
  - Analytical thermal model ([`src/thermal.py`](src/thermal.py)) driving power dissipation, lumped RC thermal inertia, and rack recirculation.
  - Baseline First Fit VM placement scheduler ([`src/placement/first_fit.py`](src/placement/first_fit.py)).
  - Discrete-time simulation engine ([`src/simulator.py`](src/simulator.py)) and telemetry reporting ([`src/metrics.py`](src/metrics.py)).
  - Reusable trace structures ([`src/dataset_loader.py`](src/dataset_loader.py)) and trace-driven workload models ([`src/workload.py`](src/workload.py)).
  - CLI runner ([`main.py`](main.py)) with topology, thermal, and strategy configuration.
- [x] **Bitbrains Dataset Integration**:
  - Bitbrains GWA-T-12 CSV loader ([`src/bitbrains_loader.py`](src/bitbrains_loader.py)) parsing `CPU usage [%]` column.
  - Dataset located at `dataset/bitbrains/fastStorage/2013-8/`.
- [x] **Milestone 2 (Reinforcement Learning Scheduler)**:
  - [x] **Phase 1**: Tabular Q-Learning placement agent with sparse dictionary Q-table, capacity filtering, epsilon-greedy action selection, and geometric decay ([`src/qlearning/agent.py`](src/qlearning/agent.py)).
  - [x] **Phase 2**: State representation discretization (VM demands, host utilization, 4-tier host temperature, 3-tier datacenter thermal condition) and modular thermal reward function calculator ([`src/qlearning/reward.py`](src/qlearning/reward.py)) penalizing temperature elevation, hotspot creation, and thermal variance.
  - [x] **Phase 3**: Q-Learning simulator strategy integration ([`src/qlearning/strategy.py`](src/qlearning/strategy.py)) with correct thermal-aware reward timing, training/evaluation mode switching, CLI strategy selection, and 35 integration tests.
  - [ ] **Phase 4 (Future)**: Multi-episode agent training, comparative benchmark (First Fit vs. Q-Learning under identical workload traces), and result plotting.
