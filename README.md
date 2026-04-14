# phys-MCP Prototype

This repository contains the prototype for the **phys-MCP** paper. The prototype validates a
**control plane for heterogeneous physical neural network (PNN) backends**.

It does **not** validate real physical substrates. Instead, it implements:

- a substrate-aware descriptor model,
- a common orchestration layer,
- lightweight digital-twin/mock backends for representative substrate classes,
- runnable demo workflows,
- and a small quantitative evaluation.

---

## What this prototype demonstrates

The prototype currently supports:

- **Discovery** of heterogeneous backend descriptors
- **Explainable backend matching**
- **Task orchestration** through one common control plane
- **Lifecycle handling**, including reset and recalibration
- **Telemetry collection**
- **Fallback routing** when a primary backend fails
- **Evaluation scripts** for overhead, portability, and matching quality

The three representative backend classes are:

- **Chemical backend**  
  concentration-driven, slow, explicit flush/recharge semantics

- **Wetware backend**  
  stimulation/observation semantics, viability-sensitive state, rest/recalibration lifecycle

- **Fast edge backend**  
  vector/tensor-oriented, low latency, device-like drift and recovery semantics

---

## Requirements

- Python **3.11** recommended
- `pip`
- `venv`

The prototype is designed to run on a single machine in a local virtual environment.

---

## Create a virtual environment

### Windows PowerShell

Check that Python is available:

```powershell
python --version
```

Create the virtual environment in a local `.venv` directory:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, allow local scripts for the current user:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

Upgrade `pip`, `setuptools`, and `wheel`:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

Install the required libraries:

```powershell
python -m pip install -r requirements.txt
```

Verify the installation:

```powershell
python -m pip list
```

Deactivate when finished:

```powershell
deactivate
```

---

### Linux / macOS / bash

Check that Python is available:

```bash
python3 --version
```

Create the virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Upgrade `pip`, `setuptools`, and `wheel`:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install the required libraries:

```bash
python -m pip install -r requirements.txt
```

Verify the installation:

```bash
python -m pip list
```

Deactivate when finished:

```bash
deactivate
```

---

## Install from scratch: command summary

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### Linux / bash

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

---

## Installed libraries

The current prototype environment uses:

- **pydantic** for typed descriptor models
- **numpy** for numerical utilities
- **scipy** for lightweight ODE-based chemical twin behavior
- **matplotlib** for evaluation plots
- **pytest** for later test support

The current version intentionally avoids a web/API stack. The prototype is implemented as an
**in-process orchestration system** rather than a networked service.

---

## Repository layout

```text
phys-mcp-prototype/
├── README.md
├── requirements.txt
├── descriptors/
│   ├── __init__.py
│   └── capability_schema.py
├── core/
│   ├── __init__.py
│   ├── matcher.py
│   ├── orchestrator.py
│   ├── task_model.py
│   └── twin_registry.py
├── adapters/
│   ├── __init__.py
│   ├── base_adapter.py
│   ├── chemical_adapter.py
│   ├── wetware_adapter.py
│   └── edge_adapter.py
├── twins/
│   ├── __init__.py
│   ├── chemical_twin.py
│   ├── wetware_twin.py
│   └── edge_twin.py
├── demos/
│   ├── __init__.py
│   ├── common.py
│   ├── demo_discovery_and_matching.py
│   ├── demo_invocation_and_telemetry.py
│   └── demo_fallback_and_recalibration.py
├── evaluation/
│   ├── __init__.py
│   ├── common.py
│   ├── plots.py
│   ├── evaluate_overhead.py
│   ├── evaluate_portability.py
│   ├── evaluate_matching.py
│   └── results/
└── tests/
```

---

## Initial sanity checks

After installation, test that the environment is usable:

### Linux / bash

```bash
python -c "import numpy, scipy, matplotlib, pydantic; print('Environment OK')"
```

### Windows PowerShell

```powershell
python -c "import numpy, scipy, matplotlib, pydantic; print('Environment OK')"
```

---

# Running the prototype

All commands below assume that:

- you are inside the project root `phys-mcp-prototype/`
- the virtual environment is activated

---

## Run the demos

The demos live in `demos/` and are intended to be run directly as standalone scripts.

### 1. Discovery and matching demo

This demo shows:

- backend discovery,
- explainable ranking,
- and task-to-substrate selection.

#### Linux / bash

```bash
python demos/demo_discovery_and_matching.py
```

#### Windows PowerShell

```powershell
python demos\demo_discovery_and_matching.py
```

---

### 2. Invocation, telemetry, and lifecycle demo

This demo shows:

- end-to-end orchestration,
- repeated chemical backend invocations,
- telemetry collection,
- and lifecycle-triggered recovery.

#### Linux / bash

```bash
python demos/demo_invocation_and_telemetry.py
```

#### Windows PowerShell

```powershell
python demos\demo_invocation_and_telemetry.py
```

---

### 3. Fallback and recalibration demo

This demo shows:

- drift-triggered recalibration,
- and fallback from a failing primary backend to a compatible backup backend.

#### Linux / bash

```bash
python demos/demo_fallback_and_recalibration.py
```

#### Windows PowerShell

```powershell
python demos\demo_fallback_and_recalibration.py
```

---

## Run the evaluation scripts

The evaluation scripts live in `evaluation/` and write result files into:

```text
evaluation/results/
```

### 1. Overhead evaluation

Measures **wall-clock control-plane overhead**, comparing direct backend access with orchestrated access.

#### Linux / bash

```bash
python evaluation/evaluate_overhead.py
```

#### Windows PowerShell

```powershell
python evaluation\evaluate_overhead.py
```

Outputs:
- `evaluation/results/overhead_results.json`
- `evaluation/results/overhead_results.csv`
- `evaluation/results/overhead_bar_chart.png`

---

### 2. Portability evaluation

Measures how consistently the abstraction behaves across heterogeneous backends.

#### Linux / bash

```bash
python evaluation/evaluate_portability.py
```

#### Windows PowerShell

```powershell
python evaluation\evaluate_portability.py
```

Outputs:
- `evaluation/results/portability_results.json`
- `evaluation/results/portability_runs.csv`
- `evaluation/results/portability_metadata_bar_chart.png`

---

### 3. Matching evaluation

Measures matcher behavior on a curated task suite.

#### Linux / bash

```bash
python evaluation/evaluate_matching.py
```

#### Windows PowerShell

```powershell
python evaluation\evaluate_matching.py
```

Outputs:
- `evaluation/results/matching_results.json`
- `evaluation/results/matching_results.csv`
- `evaluation/results/matching_accuracy_bar_chart.png`

---

## Run everything in sequence

### Linux / bash

```bash
python demos/demo_discovery_and_matching.py
python demos/demo_invocation_and_telemetry.py
python demos/demo_fallback_and_recalibration.py
python evaluation/evaluate_overhead.py
python evaluation/evaluate_portability.py
python evaluation/evaluate_matching.py
```

### Windows PowerShell

```powershell
python demos\demo_discovery_and_matching.py
python demos\demo_invocation_and_telemetry.py
python demos\demo_fallback_and_recalibration.py
python evaluation\evaluate_overhead.py
python evaluation\evaluate_portability.py
python evaluation\evaluate_matching.py
```

---

## Interpreting the results

### Overhead
The overhead evaluation reports **wall-clock runtime overhead** added by the control plane.
This is intentionally different from the **simulated substrate latency** reported by the twins.

### Portability
The portability evaluation reports how consistently descriptors and invocation results remain
structured across different backend classes, and how much backend-specific metadata each task needs.

### Matching
The matching evaluation reports the behavior of the rule-based matcher on a curated task set.
It should be interpreted as a **plausibility check**, not as a broad statistical claim of optimality.

---

## Notes on scientific scope

This prototype is intentionally modest.

It does **not** claim:
- experimental validation of real physical neural substrates,
- realistic wet-lab or device-physics fidelity,
- or optimal matching across a large empirical benchmark.

It **does** claim:
- a working control-plane abstraction,
- substrate-aware descriptors,
- semantically distinct backend classes,
- demonstrable lifecycle and telemetry handling,
- and a small but reproducible evaluation.

That is the intended contribution of the paper prototype.

---

## Troubleshooting

### PowerShell execution policy error
If activation fails, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Import/path issues
Always run the commands from the **project root** `phys-mcp-prototype/`.

### Missing packages
Reinstall dependencies:

```bash
python -m pip install -r requirements.txt
```

or in PowerShell:

```powershell
python -m pip install -r requirements.txt
```

### Regenerate evaluation outputs
Delete the files in `evaluation/results/` and rerun the evaluation scripts.

---

## Possible next extensions

Possible future additions include:

- `fastapi` / `uvicorn` for an HTTP-facing orchestrator
- `pandas` for richer tabular evaluation
- formal tests in `tests/`
- serialization examples for backend descriptors
- tighter integration of prototype outputs into the paper figures and tables
