# ThermoPlace — Thermal-Aware VM Scheduling Simulator

A modular research simulation framework designed to evaluate and compare virtual machine (VM) scheduling algorithms in data centers with explicit thermal modeling.

---

## 🎯 Project Objectives

1. **Compare Heuristic vs. Reinforcement Learning Placement**: Evaluate a traditional baseline (First Fit) against an intelligent thermal-aware tabular Q-Learning scheduler.
2. **Strict Fair Comparison**: Benchmark both algorithms under **identical workloads and initial physical data center states** using reproducible random seeds.
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
│   ├── workload.py                # Synthetic VM workload generator with fixed seeds
│   ├── datacenter.py              # Physical datacenter model (Host, Rack, DataCenter)
│   ├── thermal.py                 # Thermal model (power-to-heat, RC inertia, recirculation)
│   ├── placement/
│   │   ├── __init__.py
│   │   ├── base.py                # Base placement strategy interface
│   │   └── first_fit.py           # Traditional First Fit placement algorithm
│   ├── qlearning/
│   │   ├── __init__.py
│   │   └── agent.py               # Tabular Q-Learning agent scaffold (Milestone 2)
│   ├── metrics.py                 # Telemetry & metric collection (hotspots, imbalance, logs)
│   └── simulator.py               # Discrete-time simulation engine and execution loop
└── tests/
│   ├── __init__.py
│   ├── test_workload.py           # Unit tests for workload determinism and schema
│   ├── test_datacenter.py         # Unit tests for host capacity, allocation, and resets
│   ├── test_thermal.py            # Unit tests for thermal inertia and power models
│   ├── test_first_fit.py          # Unit tests for First Fit placement decisions
│   ├── test_metrics.py            # Unit tests for metric calculation accuracy
│   └── test_simulator.py          # End-to-end integration test for repeatable simulation runs
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

### 2. Running the Simulation
To execute the baseline First Fit simulation run:
```bash
python main.py
```
Optional CLI flags:
```bash
# Run with custom seed, steps, and hosts
python main.py --seed 42 --steps 100 --num-racks 2 --hosts-per-rack 4
```

### 3. Running Unit Tests
ThermoPlace uses Python's built-in `unittest` framework:
```bash
python -m unittest discover tests
```

---

## 📌 Implementation Status

- [x] **Milestone 1 (Current)**:
  - Repeatable synthetic workload generator with fixed random seeds.
  - Object-oriented Host, Rack, and DataCenter models with strict CPU/RAM capacity enforcement.
  - Complete host reset functionality between runs.
  - Analytical thermal model with power curves, thermal inertia, and rack recirculation.
  - Baseline First Fit VM placement strategy.
  - Metric collection (hotspots, peak temperature, thermal imbalance, resource usage).
  - Main runner and comprehensive unit test suite.
- [ ] **Milestone 2 (Next)**:
  - Tabular Q-Learning placement agent with state discretization (thermal & utilization bins).
  - Reward function penalizing hotspot creation and thermal variance.
  - Comparative benchmark: First Fit vs. Q-Learning under identical workload traces.
  - Comparative plotting and report generation.
