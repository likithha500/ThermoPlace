# PlanetLab Dataset Integration — ThermoPlace

This document explains how the PlanetLab workload dataset is integrated into ThermoPlace for thermal-aware VM scheduling research.

---

## 1. Dataset Source and Citation

**Dataset:** PlanetLab Workload Traces (CPU Utilization)  
**Repository:** https://github.com/beloglazov/planetlab-workload-traces  
**Collected by:** Anton Beloglazov, The University of Melbourne  
**Collection period:** March–April 2011 (10 date-based snapshots)

**Citation:**
```
Beloglazov, A., & Buyya, R. (2012). Optimal Online Deterministic Algorithms and
Adaptive Heuristics for Energy and Performance Efficient Dynamic Consolidation of
Virtual Machines in Cloud Data Centers. Concurrency and Computation: Practice and
Experience, 24(13), 1397–1420.
```

This dataset is distinct from the **GWA-T-12 Bitbrains** dataset (also hosted at TU Delft). They share a common research context but are separate collections with different file formats.

---

## 2. Dataset Structure and File Format

### Directory layout

```
data/planetlab/
  20110303/     ← date-based subdirectory (YYYYMMDD)
    146-179_surfsnel_dsl_internl_net_colostate_557.txt
    146-179_surfsnel_dsl_internl_net_rnp_dcc_ufjf.txt
    ...          ← 1,000 files per day
  20110306/
    ...
  20110309/
    ...
  (10 date directories total)
```

### File format

Each file is a plain-text file with:
- **No header row**
- **One CPU utilization value per line** (integer or float)
- **288 lines per file** — one reading per 5-minute interval over a 24-hour period (288 × 5 min = 1440 min = 24 h)
- **Values are CPU utilization percentages** in the range [0, 100]

**Example (first 5 lines of a real file):**
```
24
34
29
26
26
```

### What is measured vs. simulated

| Feature | Source | Notes |
|---|---|---|
| **CPU utilization %** | ✅ Measured | Directly from PlanetLab node monitoring |
| **RAM demand** | ⚠️ Simulated | Assigned from flavor profiles `[4, 8, 16, 32]` GB at schedule generation time |
| **VM arrival time** | ⚠️ Simulated | Poisson process with configurable `--arrival-rate` |
| **VM duration** | ⚠️ Simulated | Derived from trace length and `--steps` |

> **Important:** RAM consumption is **not** measured in PlanetLab traces. It is randomly assigned from standard VM flavor profiles by `TraceWorkloadGenerator`, using the same approach as the Bitbrains integration. This is a documented assumption, not a measurement.

---

## 3. Download Instructions

The dataset is freely available from GitHub. To download 50 representative traces:

```powershell
# PowerShell — download 50 traces from 20110303
$wc = New-Object System.Net.WebClient
$wc.Headers.Add("User-Agent", "ThermoPlace-Research")
$apiUrl = "https://api.github.com/repos/beloglazov/planetlab-workload-traces/contents/20110303"
$resp = $wc.DownloadString($apiUrl)
$items = $resp | ConvertFrom-Json
$top50 = $items[0..49]
$dir = "data/planetlab/20110303"
New-Item -ItemType Directory -Path $dir -Force | Out-Null
foreach ($item in $top50) {
    $wc2 = New-Object System.Net.WebClient
    $wc2.Headers.Add("User-Agent", "ThermoPlace-Research")
    $content = $wc2.DownloadString($item.download_url)
    [System.IO.File]::WriteAllText("$dir/$($item.name).txt", $content)
}
```

Or download via `git clone` for the full dataset:

```bash
git clone https://github.com/beloglazov/planetlab-workload-traces data/planetlab-raw
# Then rename/reorganize as needed
```

**Note:** The ThermoPlace repository ships with 50 pre-downloaded traces in `data/planetlab/20110303/`.

---

## 4. Preprocessing

The `PlanetLabLoader` applies the following preprocessing automatically:

1. **Empty lines** — skipped silently
2. **Non-numeric lines** — skipped with a debug-level log message
3. **Out-of-range values** — clamped: values < 0 → 0.0, values > 100 → 100.0
4. **Empty files** — recorded in `DatasetLoadReport.skipped_files` with the reason
5. **File naming** — the `vm_id` is set to the filename stem (without extension)

No separate preprocessing script is required.

---

## 5. Running Experiments

### First Fit Simulation (baseline)

```bash
python main.py \
  --dataset-dir data/planetlab/20110303 \
  --dataset-type planetlab \
  --strategy first_fit \
  --max-vms 50 \
  --steps 100 \
  --seed 42
```

### Q-Learning Training

```bash
python main.py \
  --dataset-dir data/planetlab/20110303 \
  --dataset-type planetlab \
  --strategy q_learning \
  --train \
  --episodes 500 \
  --max-vms 50 \
  --steps 100 \
  --seed 42 \
  --model-path models/qlearning/q_table_planetlab.json \
  --output-dir results/training_planetlab \
  --overwrite
```

Or using the dedicated training CLI:

```bash
python train.py \
  --dataset-dir data/planetlab/20110303 \
  --dataset-type planetlab \
  --episodes 500 \
  --max-vms 50 \
  --steps 100 \
  --seed 42 \
  --model-path models/qlearning/q_table_planetlab.json
```

### Q-Learning Evaluation (greedy, no updates)

```bash
python main.py \
  --dataset-dir data/planetlab/20110303 \
  --dataset-type planetlab \
  --strategy q_learning \
  --eval-mode \
  --q-table models/qlearning/q_table_planetlab.json \
  --max-vms 50 \
  --steps 100 \
  --seed 42
```

### Fair Comparison: First Fit vs Q-Learning

For a fair comparison, use **identical** parameters for both runs:

```bash
# First Fit baseline
python main.py --dataset-dir data/planetlab/20110303 --dataset-type planetlab \
  --strategy first_fit --max-vms 50 --steps 100 --seed 42

# Q-Learning evaluation (after training)
python main.py --dataset-dir data/planetlab/20110303 --dataset-type planetlab \
  --strategy q_learning --eval-mode --q-table models/qlearning/q_table_planetlab.json \
  --max-vms 50 --steps 100 --seed 42
```

---

## 6. Architecture Changes

### New files

| File | Description |
|---|---|
| `src/planetlab_loader.py` | PlanetLab trace loader; produces `VMTrace` objects via `PlanetLabLoader.load_with_report()` |
| `tests/test_planetlab_loader.py` | 37 unit and integration tests |
| `data/planetlab/20110303/` | 50 pre-downloaded PlanetLab trace files |
| `docs/planetlab_integration.md` | This document |

### Modified files

| File | Change |
|---|---|
| `main.py` | Added `--dataset-type {bitbrains,planetlab}` argument; dispatches to correct loader |
| `train.py` | Added `--dataset-type {bitbrains,planetlab}` argument; passes through to `TrainingConfig` |
| `src/qlearning/trainer.py` | Added `dataset_type` field to `TrainingConfig`; `_load_dataset_workload()` dispatches by type; `_load_bitbrains_workload()` preserved as legacy alias |

### Unchanged files (confirmed)

`src/dataset_loader.py`, `src/workload.py`, `src/simulator.py`, `src/datacenter.py`,  
`src/placement/base.py`, `src/placement/first_fit.py`, `src/qlearning/agent.py`,  
`src/qlearning/reward.py`, `src/qlearning/strategy.py`, `src/thermal.py`, `src/metrics.py`

---

## 7. Experimental Assumptions and Limitations

1. **RAM is not measured.** PlanetLab traces contain CPU utilization only. RAM demand is randomly assigned from flavor profiles `[4, 8, 16, 32]` GB. This matches the approach used for Bitbrains.

2. **Fixed 24-hour trace window.** Each PlanetLab trace covers exactly 24 hours at 5-minute resolution. The simulator replays traces cyclically if the simulation duration exceeds 288 steps.

3. **Arrival times are synthetic.** VM arrivals follow a Poisson process with configurable `--arrival-rate`. The original PlanetLab data does not capture VM arrival/departure events.

4. **Single-day snapshot.** The pre-downloaded data covers `20110303` only. Using multiple days (e.g., `20110306`) would require downloading additional directories.

5. **Q-table portability.** A Q-table trained on PlanetLab data can be evaluated on PlanetLab workloads and vice versa for Bitbrains, but cross-dataset evaluation may show degraded performance since state distributions differ.

6. **No thermal measurement.** Temperatures are computed analytically by the RC thermal model using CPU utilization as input. No actual server temperatures are measured or reproduced.

---

## 8. Test Coverage

Run the PlanetLab test suite:

```bash
python -m pytest tests/test_planetlab_loader.py -v
```

37 tests covering:
- File parsing (valid, clamped, empty, non-numeric)
- Directory loading (recursive, flat, max_traces)
- DatasetLoadReport counts
- TraceWorkloadGenerator compatibility
- First Fit simulation integration
- Q-Learning simulation integration
- CLI argument parsing (`--dataset-type`)
- TrainingConfig defaults and dispatch
- Independence from BitbrainsLoader and pandas

Run the full suite (all tests):

```bash
python -m pytest tests/ -v
```

