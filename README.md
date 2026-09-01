# AI-Powered Quantum Circuit Optimization Advisor

An interactive quantum circuit analysis and optimization dashboard built using **Qiskit, Machine Learning, and Streamlit**.

The system accepts a quantum circuit, analyzes its structure, compares Qiskit optimization levels `0–3`, predicts `Success_Probability` using a trained Random Forest model, estimates noise exposure, and recommends a suitable optimization level.

## Live Dashboard

👉 [Open the Quantum Circuit Optimization Advisor](https://quantum-circuit-optimization-advisor-lqppos4z5vycedajsyflwh.streamlit.app/)

---

## Project Workflow

```text
Quantum Circuit
      ↓
Feature Extraction
      ↓
Backend + Noise Conditions
      ↓
Optimization Levels 0–3
      ↓
Random Forest Prediction
      ↓
Optimization Recommendation
      ↓
Interactive Dashboard
```

---

## Features

- Built-in quantum circuits
- Manual gate builder
- OpenQASM 2 input
- Circuit depth and gate analysis
- Single- and multi-qubit gate counts
- Backend and noise-profile selection
- Qiskit optimization levels `0–3`
- Optimized depth and gate count
- Depth and gate reduction
- Noise-exposure analysis
- Noise-sensitive depth layers
- Random Forest Success Probability prediction
- Automatic optimization recommendation
- Optimized circuit visualization
- Dataset analytics
- Model-performance dashboard

---

## Supported Built-in Circuits

- Bell
- GHZ
- Deutsch-Jozsa
- Bernstein-Vazirani
- Grover
- QFT
- VQE
- QAOA
- Simon

Custom circuits can also be created using the **Manual Gate Builder** or supplied through **OpenQASM 2**.

---

## Machine Learning Model

The current model is a **Random Forest Regressor** trained to predict:

```text
Success_Probability
```

Current evaluation results:

| Metric | Value |
|---|---:|
| MAE | 0.045087 |
| MSE | 0.011213 |
| RMSE | 0.105893 |
| R² | 0.901976 |

The trained model is stored as:

```text
random_forest_success_probability.joblib
```

---

## Benchmark Dataset

The project uses an automatically generated benchmark dataset containing:

```text
1500 quantum circuit execution samples
```

The dataset includes variations in:

- Algorithm
- Number of qubits
- Backend profile
- Optimization level
- Noise level
- Shots

Dataset file:

```text
quantum_circuit_dataset_clean.xlsx
```

---

## Noise Exposure

The dashboard displays a **noise exposure map**.

It does not claim that a physical error definitely occurred at a specific gate. Instead, it identifies operations and circuit depth layers that are more exposed to modeled noise effects such as:

- One-qubit gate error
- Two-qubit gate error
- T1/T2 exposure
- Readout error

---

## Repository Structure

```text
Quantum-Circuit-Optimization-Advisor/
├── app.py
├── Quantum_Circuit_Optimization_Advisor.ipynb
├── quantum_circuit_dataset_clean.xlsx
├── random_forest_success_probability.joblib
├── metrics.json
├── requirements.txt
└── README.md
```


## Technologies Used

- Python
- Qiskit
- Qiskit Aer
- Scikit-learn
- Random Forest Regression
- Pandas
- NumPy
- SciPy
- Joblib
- Streamlit
- OpenQASM
- Git
- GitHub





##Scope

The model was trained primarily on the benchmark circuit families included in the dataset.

Custom circuits can still be analyzed and transpiled by Qiskit, but predictions for circuits very different from the training data should be interpreted as model estimates rather than guaranteed hardware performance.

---

## Links

**Live Dashboard:**  
https://quantum-circuit-optimization-advisor-lqppos4z5vycedajsyflwh.streamlit.app/

