# AI-Powered Quantum Circuit Optimization Advisor
## Interactive Streamlit Dashboard

This dashboard is built for the revised project pipeline.

It accepts a quantum circuit and shows:

- raw circuit depth
- total gates and multi-qubit gates
- optimization levels 0, 1, 2 and 3
- optimized depth and gate count
- estimated noise exposure
- the depth layers where noise-sensitive operations occur
- gate-level noise hotspots
- predicted Success Probability using the trained Random Forest model
- recommended optimization level
- an optimized circuit view
- dataset analytics

## Important scientific wording

The dashboard displays a **noise exposure map**.

It does NOT claim that a random physical error definitely occurred at a specific gate.
The map shows which operations and depth layers are exposed to the project's modeled
noise channels:

- one-qubit gate error
- two-qubit gate error
- T1/T2 exposure
- readout error

That is the correct interpretation for this simulator-based project.

---

## Required project files

The dashboard itself is complete, but for live ML prediction you need the model
already saved by your notebook:

`random_forest_success_probability.joblib`

For the Dataset page, optionally place:

`quantum_circuit_dataset_clean.xlsx`

in the same folder as `app.py`.

You may also upload either file from the dashboard sidebar.

---

## Run in Windows / VS Code

Open a terminal in this folder.

### 1. Activate your Qiskit environment

```bash
conda activate myQisk
```

### 2. Install dashboard packages

```bash
pip install -r requirements.txt
```

### 3. Start dashboard

```bash
streamlit run app.py
```

The browser should open automatically.

You can also double-click `run_dashboard.bat`.

---

## Circuit input options

### Built-in Algorithm
Choose one of:

- Bell
- GHZ
- Deutsch-Jozsa
- Bernstein-Vazirani
- Grover
- QFT
- VQE
- QAOA
- Simon

### Manual Gate Builder
Create your own circuit using:

- H
- X
- Y
- Z
- RX
- RY
- RZ
- CX
- CZ
- SWAP
- CCX
- MEASURE_ALL

### OpenQASM 2
Paste or upload a `.qasm` circuit.

---

## What happens after you click Analyze

1. The circuit is converted into the canonical feature basis.
2. Raw circuit depth and structural features are extracted.
3. The selected backend and noise profile are applied.
4. The circuit is transpiled at optimization levels 0–3.
5. The dashboard compares depth, gates and noise exposure.
6. If the trained model is loaded, Success Probability is predicted for all four levels.
7. The best level is recommended.
8. The selected optimized circuit is shown.
9. Noise-sensitive depth layers and gate hotspots are displayed.

---

## Current model results

These values are pre-filled from the latest result reported during development:

- MAE: 0.045087
- MSE: 0.011213
- RMSE: 0.105893
- R²: 0.901976

If you retrain the model, edit `LATEST_METRICS` near the top of `app.py`.
