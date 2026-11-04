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
├── main.py                        # Main CLI entry point to run simulations & training
├── train.py                       # Dedicated CLI entry point for Q-Learning training
├── dataset/
│   └── bitbrains/fastStorage/2013-8/   # Bitbrains GWA-T-12 VM trace CSVs
├── models/
│   └── qlearning/
│       └── q_table.json           # Serialized trained Q-table model & metadata
├── results/
│   └── training/
│       ├── training_history.csv   # Episode-by-episode telemetry records
│       └── plots/                 # Generated Matplotlib performance graphs
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
│   │   ├── agent.py               # Tabular Q-Learning agent, QTable, save/load (Phase 1 & 4)
│   │   ├── reward.py              # Thermal-aware reward function calculator (Phase 2)
│   │   ├── strategy.py            # Q-Learning simulator strategy integration (Phase 3)
│   │   └── trainer.py             # Multi-episode training loop, CSV & plots (Phase 4)
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
    ├── test_phase3.py             # Integration tests for Q-Learning simulator integration
    └── test_phase4.py             # Comprehensive test suite for Phase 4 training & persistence
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
>    $$T_{\text{inlet}, i}(t) = T_{\text{ambient}} + \alpha \cdot P_{\text{rack}}(t) + \beta \cdot \text{slot}_i$$
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
| **Student 3: RL & Evaluation (Milestone 2)** | Tabular Q-Learning policy, reward engineering, simulator integration, training loop | `src/qlearning/`, `src/metrics.py`, `src/simulator.py` |

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

---

## 🎓 Phase 4: Training the Q-Learning Agent

In Phase 4, the Tabular Q-Learning agent is trained across multiple simulation episodes using real Bitbrains workload traces.

### 1. Starting Training

You can initiate training via the dedicated training CLI (`train.py`) or through `main.py`:

```bash
# Option A: Dedicated training script (recommended)
python train.py --dataset-dir dataset/bitbrains/fastStorage/2013-8 --episodes 500

# Option B: Via main CLI
python main.py --dataset-dir dataset/bitbrains/fastStorage/2013-8 --strategy q_learning --train --episodes 500
```

### 2. Configurable Training Hyperparameters

The training process is fully configurable with the following initial experimental settings:

| Parameter | Flag | Default Value | Description |
| :--- | :--- | :---: | :--- |
| **Episodes** | `--episodes` | `500` | Total simulation episodes to train |
| **Learning Rate ($\alpha$)** | `--learning-rate`, `--alpha` | `0.1` | Step size for Bellman error updates |
| **Discount Factor ($\gamma$)** | `--discount-factor`, `--gamma` | `0.95` | Importance weight for future returns |
| **Initial Epsilon ($\epsilon$)** | `--initial-epsilon` | `1.0` | Initial exploration probability |
| **Minimum Epsilon ($\epsilon_{\min}$)** | `--min-epsilon` | `0.05` | Exploration probability floor |
| **Epsilon Decay** | `--epsilon-decay` | `0.995` | Multiplicative decay per episode: $\epsilon \leftarrow \max(\epsilon_{\min}, \epsilon \cdot \text{decay})$ |
| **Random Seed** | `--seed` | `42` | Random seed for arrival times, VM flavors, and action exploration |
| **Max VM Traces** | `--max-vms` | `50` | Maximum Bitbrains traces to load |
| **Time Steps / Episode** | `--steps` | `100` | Discrete time steps simulated per episode |
| **Topology** | `--racks`, `--hosts-per-rack` | `2`, `4` | 2 server racks $\times$ 4 hosts = 8 physical servers |
| **Host Limits** | `--cpu-per-host`, `--ram-per-host` | `16`, `64` | 16 CPU cores, 64 GB RAM per physical host |
| **Ambient CRAH Temp** | `--ambient-temp` | `25.0` | Background computer room supply air temperature in °C |
| **Hotspot Threshold** | `--hotspot-threshold` | `75.0` | Thermal safety threshold in °C |
| **Overwrite** | `--overwrite` | `False` | Overwrite existing CSV and model files (creates timestamped backups if False) |

**Example with custom parameters:**
```bash
python train.py --dataset-dir dataset/bitbrains/fastStorage/2013-8 \
                --episodes 500 \
                --learning-rate 0.1 \
                --discount-factor 0.95 \
                --initial-epsilon 1.0 \
                --min-epsilon 0.05 \
                --epsilon-decay 0.995 \
                --seed 42 \
                --overwrite
```

### 3. Training Output & Persistence

All training artifacts are automatically created in structured directories:

- **Episode-by-Episode History (CSV)**:
  `results/training/training_history.csv`
  Contains 13 performance metrics recorded at each episode boundary:
  1. `episode`: Episode index (1 to $N$).
  2. `cumulative_reward`: Total composite reward accumulated in the episode.
  3. `avg_reward_per_decision`: Average reward received per VM placement attempt.
  4. `successful_placements`: Number of VMs placed on eligible hosts.
  5. `rejected_vms`: Number of VMs rejected due to capacity limits.
  6. `total_hotspots`: Sum of hotspot events across all simulation steps.
  7. `max_host_temperature`: Peak host temperature observed (°C).
  8. `avg_host_temperature`: Mean host temperature across all hosts and steps (°C).
  9. `thermal_imbalance`: Spatial standard deviation $\sigma_T$ across hosts (°C).
  10. `avg_cpu_utilization`: Average CPU allocation ratio $[0, 1]$.
  11. `avg_ram_utilization`: Average RAM allocation ratio $[0, 1]$.
  12. `epsilon`: Exploration rate at the end of the episode.
  13. `q_table_entries`: Count of distinct state-action pairs stored in the Q-table.

- **Trained Q-Table Model (JSON)**:
  `models/qlearning/q_table.json`
  Stores all non-zero state-action pairs, hyperparameters, state discretization bin thresholds, training metadata, and entry count. Includes schema validation to prevent loading incompatible models.

- **Training Progress Visualizations**:
  `results/training/plots/`
  1. `cumulative_reward.png`: Cumulative reward vs. training episodes.
  2. `average_reward.png`: Average reward per decision vs. training episodes.
  3. `epsilon_decay.png`: Epsilon decay vs. training episodes.
  4. `hotspot_count.png`: Hotspot count vs. training episodes.
  5. `max_host_temperature.png`: Peak host temperature vs. training episodes.
  6. `thermal_imbalance.png`: Thermal imbalance vs. training episodes.
  7. `successful_placements.png`: Successful VM placements vs. training episodes.
  8. `training_dashboard.png`: Combined 8-panel dashboard summarizing all metrics.

---

## 🔍 Evaluation Mode (Evaluating Saved Q-Table)

Strict separation is maintained between training and evaluation:

- **Evaluation Mode Rules:**
  - The saved Q-table is loaded from JSON.
  - Epsilon ($\epsilon$) is forced to `0.0` (pure greedy exploitation).
  - Q-table updates are strictly disabled (model remains frozen).
  - Epsilon decay does not occur.
  - Evaluation results are never used to update Q-values.

**Run evaluation using a trained Q-table:**
```bash
python main.py --dataset-dir dataset/bitbrains/fastStorage/2013-8 \
               --strategy q_learning \
               --eval-mode \
               --q-table models/qlearning/q_table.json
```

---

## 🏃 Baseline Simulation (First Fit)

To run a simulation run with First Fit:
```bash
python main.py --dataset-dir dataset/bitbrains/fastStorage/2013-8 --strategy first_fit
```

---

## 🧪 Running Unit & Integration Tests

Run the complete test suite (115 tests):
```bash
python -m unittest discover -s tests -v
```

Run Phase 4 training, persistence, and reset tests:
```bash
python -m unittest tests/test_phase4.py -v
```

Run Phase 3 simulator integration tests:
```bash
python -m unittest tests/test_phase3.py -v
```

---

## 🔄 Simulator Lifecycle

The `Simulator.step()` method executes the following per-time-step sequence:

```
1. Tick & release finished VMs
2. strategy.on_step_begin(...)       ← strategy caches arriving VMs
3. For each arriving VM:
   a. strategy.select_host(vm, dc)  ← epsilon-greedy (or greedy in eval)
   b. host.allocate(vm)             ← CPU + RAM reserved
   c. strategy.on_placement_outcome(...)  ← records pending transition
4. thermal_model.update_datacenter(...)   ← analytical RC thermal step (1/step)
5. strategy.on_step_end(...)         ← computes rewards + updates Q-values
6. metrics.record_step(...)         ← telemetry snapshot
```

**Episode Reset Contract:**
Between episodes:
1. `simulator.reset(ambient_temp)` clears all host allocations (`used_cpu=0`, `used_ram=0`, `allocated_vms.clear()`) and resets host temperatures to ambient.
2. `trainer.get_episode_schedule()` ensures all VM instances are reset (`remaining_duration = duration`, `current_sample_index = 0`).
3. `strategy.reset_episode()` clears transient buffers while preserving the learned Q-table.
4. Epsilon is decayed once per episode during `strategy.on_simulation_end()`.

---

## 📌 Implementation Status

- [x] **Core Simulation Architecture**:
  - Object-oriented Host, Rack, and DataCenter models ([`src/datacenter.py`](src/datacenter.py)).
  - Analytical RC thermal model with rack recirculation ([`src/thermal.py`](src/thermal.py)).
  - Baseline First Fit scheduler ([`src/placement/first_fit.py`](src/placement/first_fit.py)).
  - Discrete-time simulation engine ([`src/simulator.py`](src/simulator.py)).
  - Telemetry and metrics reporting ([`src/metrics.py`](src/metrics.py)).
  - Reusable trace structures ([`src/dataset_loader.py`](src/dataset_loader.py)).
- [x] **Bitbrains Dataset Integration**:
  - Bitbrains GWA-T-12 CSV loader ([`src/bitbrains_loader.py`](src/bitbrains_loader.py)).
  - Trace workload generator with deterministic arrival and flavor assignment ([`src/workload.py`](src/workload.py)).
- [x] **Milestone 2 (Reinforcement Learning Scheduler)**:
  - [x] **Phase 1**: Tabular Q-Learning placement agent with sparse dictionary Q-table, capacity filtering, epsilon-greedy action selection, and geometric decay ([`src/qlearning/agent.py`](src/qlearning/agent.py)).
  - [x] **Phase 2**: State representation discretization (VM demands, host utilization, 4-tier host temperature, 3-tier datacenter thermal condition) and modular thermal reward function calculator ([`src/qlearning/reward.py`](src/qlearning/reward.py)).
  - [x] **Phase 3**: Q-Learning simulator strategy integration ([`src/qlearning/strategy.py`](src/qlearning/strategy.py)) with correct post-thermal reward timing, training/evaluation mode switching, and CLI strategy selection.
  - [x] **Phase 4**: Multi-episode Q-Learning training loop using Bitbrains traces ([`src/qlearning/trainer.py`](src/qlearning/trainer.py), [`train.py`](train.py)), episode reset mechanism, CSV history logging, Matplotlib progress visualization, Q-table JSON persistence, strict evaluation separation, and 15 unit/integration tests.
