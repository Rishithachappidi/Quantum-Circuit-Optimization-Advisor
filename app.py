
import io
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate, efficient_su2, qaoa_ansatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.converters import circuit_to_dag


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Quantum Circuit Optimization Advisor",
    page_icon="⚛️",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        .hero {
            padding: 1.3rem 1.5rem;
            border-radius: 18px;
            border: 1px solid rgba(128,128,128,0.25);
            margin-bottom: 1rem;
        }

        .hero h1 {
            margin-bottom: 0.2rem;
        }

        .soft-box {
            padding: 1rem;
            border-radius: 14px;
            border: 1px solid rgba(128,128,128,0.20);
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,0.20);
            border-radius: 14px;
            padding: 0.75rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PROJECT CONSTANTS
# ============================================================

SEED = 42

ALGORITHMS = [
    "Bell",
    "GHZ",
    "DeutschJozsa",
    "BernsteinVazirani",
    "Grover",
    "QFT",
    "VQE",
    "QAOA",
    "Simon",
]

ALGORITHM_QUBITS = {
    "Bell": [2],
    "GHZ": list(range(3, 11)),
    "DeutschJozsa": list(range(3, 11)),
    "BernsteinVazirani": list(range(3, 11)),
    "Grover": list(range(2, 6)),
    "QFT": list(range(2, 11)),
    "VQE": list(range(2, 11)),
    "QAOA": list(range(3, 11)),
    "Simon": [4, 6, 8, 10],
}

OPT_LEVELS = [0, 1, 2, 3]
SHOT_LEVELS = [256, 512, 1024, 2048, 4096]
NOISE_LEVELS = ["Ideal", "Low", "Medium", "High"]
BACKEND_PROFILES = ["Aer_FullyConnected", "Aer_LinearNISQ"]

FEATURE_BASIS = ["h", "x", "y", "z", "rx", "ry", "rz", "cx", "swap", "ccx"]
HARDWARE_BASIS = ["rz", "sx", "x", "cx"]

BACKEND_BASE = {
    "Aer_FullyConnected": {
        "p1": 0.00035,
        "p2": 0.0035,
        "readout": 0.008,
        "t1_us": 130.0,
        "t2_us": 105.0,
    },
    "Aer_LinearNISQ": {
        "p1": 0.00065,
        "p2": 0.0075,
        "readout": 0.016,
        "t1_us": 95.0,
        "t2_us": 75.0,
    },
}

NOISE_SCALE = {
    "Ideal": 0.0,
    "Low": 0.60,
    "Medium": 1.00,
    "High": 1.80,
}

LATEST_METRICS = {
    "MAE": 0.045087,
    "MSE": 0.011213,
    "RMSE": 0.105893,
    "R²": 0.901976,
}

DEFAULT_MODEL_NAMES = [
    "random_forest_success_probability.joblib",
    "random_forest_success_probability.pkl",
]

DEFAULT_DATASET_NAMES = [
    "quantum_circuit_dataset_clean.xlsx",
    "quantum_circuit_dataset_clean.csv",
]


# ============================================================
# FILE / MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_local_model(path_str):
    return joblib.load(path_str)


@st.cache_data(show_spinner=False)
def read_dataset_from_path(path_str):
    path = Path(path_str)
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path)


def find_first_existing(names):
    for name in names:
        path = Path(name)
        if path.exists():
            return path
    return None


def load_uploaded_model(uploaded):
    return joblib.load(io.BytesIO(uploaded.getvalue()))


def read_uploaded_dataset(uploaded):
    data = uploaded.getvalue()
    if uploaded.name.lower().endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data))


# ============================================================
# QUANTUM CIRCUIT GENERATORS
# ============================================================

def generate_bell(total_qubits, rng):
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def generate_ghz(total_qubits, rng):
    n = total_qubits
    qc = QuantumCircuit(n, n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure(range(n), range(n))
    return qc


def generate_deutsch_jozsa(total_qubits, rng):
    n_inputs = total_qubits - 1
    anc = n_inputs
    qc = QuantumCircuit(total_qubits, n_inputs)

    qc.x(anc)
    qc.h(range(total_qubits))

    oracle_type = rng.choice(["constant", "balanced"])

    if oracle_type == "constant":
        if int(rng.integers(0, 2)) == 1:
            qc.x(anc)
    else:
        mask = rng.integers(0, 2, size=n_inputs)
        if not mask.any():
            mask[int(rng.integers(0, n_inputs))] = 1
        for q, bit in enumerate(mask):
            if bit:
                qc.cx(q, anc)

    qc.h(range(n_inputs))
    qc.measure(range(n_inputs), range(n_inputs))
    return qc


def generate_bernstein_vazirani(total_qubits, rng):
    n_inputs = total_qubits - 1
    anc = n_inputs
    secret = "".join(str(int(x)) for x in rng.integers(0, 2, size=n_inputs))

    qc = QuantumCircuit(total_qubits, n_inputs)
    qc.x(anc)
    qc.h(range(total_qubits))

    for q in range(n_inputs):
        if secret[n_inputs - 1 - q] == "1":
            qc.cx(q, anc)

    qc.h(range(n_inputs))
    qc.measure(range(n_inputs), range(n_inputs))
    return qc


def grover_phase_oracle(qc, marked_count_order):
    n = len(marked_count_order)
    marked_qorder = marked_count_order[::-1]

    for q, bit in enumerate(marked_qorder):
        if bit == "0":
            qc.x(q)

    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)

    for q, bit in enumerate(marked_qorder):
        if bit == "0":
            qc.x(q)


def grover_diffuser(qc):
    n = qc.num_qubits
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))


def generate_grover(total_qubits, rng):
    n = total_qubits
    marked = format(int(rng.integers(0, 2**n)), f"0{n}b")
    qc = QuantumCircuit(n, n)
    qc.h(range(n))

    iterations = max(
        1,
        int(math.floor((math.pi / 4) * math.sqrt(2**n))),
    )

    for _ in range(iterations):
        grover_phase_oracle(qc, marked)
        grover_diffuser(qc)

    qc.measure(range(n), range(n))
    return qc


def generate_qft(total_qubits, rng):
    n = total_qubits
    value = int(rng.integers(0, 2**n))
    qc = QuantumCircuit(n, n)

    for q in range(n):
        if (value >> q) & 1:
            qc.x(q)

    qft_gate = QFTGate(n)
    qc.append(qft_gate, range(n))
    qc.append(qft_gate.inverse(), range(n))
    qc.measure(range(n), range(n))
    return qc


def generate_vqe(total_qubits, rng):
    n = total_qubits

    ansatz = efficient_su2(
        n,
        su2_gates=["ry", "rz"],
        entanglement="linear",
        reps=2,
    )

    values = rng.normal(
        loc=0.0,
        scale=0.30,
        size=len(ansatz.parameters),
    )
    bound = ansatz.assign_parameters(
        dict(zip(ansatz.parameters, values))
    )

    qc = QuantumCircuit(n, n)
    qc.compose(bound, inplace=True)
    qc.measure(range(n), range(n))
    return qc


def generate_qaoa(total_qubits, rng):
    n = total_qubits
    edges = [(i, (i + 1) % n) for i in range(n)]

    pauli_terms = []
    for i, j in edges:
        label = ["I"] * n
        label[i] = "Z"
        label[j] = "Z"
        pauli_terms.append(("".join(label[::-1]), 1.0))

    cost_operator = SparsePauliOp.from_list(pauli_terms)
    ansatz = qaoa_ansatz(cost_operator, reps=1)

    values = rng.uniform(
        0.15,
        1.25,
        size=len(ansatz.parameters),
    )
    bound = ansatz.assign_parameters(
        dict(zip(ansatz.parameters, values))
    )

    qc = QuantumCircuit(n, n)
    qc.compose(bound, inplace=True)
    qc.measure(range(n), range(n))
    return qc


def generate_simon(total_qubits, rng):
    n = total_qubits // 2
    qc = QuantumCircuit(2 * n, n)

    secret = rng.integers(0, 2, size=n)
    if not secret.any():
        secret[int(rng.integers(0, n))] = 1

    pivot = int(np.where(secret == 1)[0][0])

    rows = []
    for j in range(n):
        if j == pivot:
            continue

        row = np.zeros(n, dtype=int)
        row[j] = 1
        row[pivot] = int(secret[j])
        rows.append(row)

    qc.h(range(n))

    for out_index, row in enumerate(rows):
        output_qubit = n + out_index
        for input_index, bit in enumerate(row):
            if bit:
                qc.cx(input_index, output_qubit)

    qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc


GENERATORS = {
    "Bell": generate_bell,
    "GHZ": generate_ghz,
    "DeutschJozsa": generate_deutsch_jozsa,
    "BernsteinVazirani": generate_bernstein_vazirani,
    "Grover": generate_grover,
    "QFT": generate_qft,
    "VQE": generate_vqe,
    "QAOA": generate_qaoa,
    "Simon": generate_simon,
}


# ============================================================
# MANUAL CIRCUIT BUILDER
# ============================================================

def default_gate_editor():
    return pd.DataFrame(
        [
            {"Gate": "H", "Q0": 0, "Q1": -1, "Q2": -1, "Angle": 0.0},
            {"Gate": "CX", "Q0": 0, "Q1": 1, "Q2": -1, "Angle": 0.0},
            {"Gate": "MEASURE_ALL", "Q0": -1, "Q1": -1, "Q2": -1, "Angle": 0.0},
        ]
    )


def build_manual_circuit(n_qubits, operations):
    qc = QuantumCircuit(n_qubits, n_qubits)

    one_qubit_gates = {
        "H": qc.h,
        "X": qc.x,
        "Y": qc.y,
        "Z": qc.z,
    }

    rotation_gates = {
        "RX": qc.rx,
        "RY": qc.ry,
        "RZ": qc.rz,
    }

    for row_number, row in operations.iterrows():
        gate = str(row["Gate"]).strip().upper()

        if gate in {"", "NONE", "NAN"}:
            continue

        q0 = int(row["Q0"])
        q1 = int(row["Q1"])
        q2 = int(row["Q2"])
        angle = float(row["Angle"])

        def validate_qubit(q):
            if q < 0 or q >= n_qubits:
                raise ValueError(
                    f"Row {row_number + 1}: qubit {q} is outside 0..{n_qubits - 1}"
                )

        if gate in one_qubit_gates:
            validate_qubit(q0)
            one_qubit_gates[gate](q0)

        elif gate in rotation_gates:
            validate_qubit(q0)
            rotation_gates[gate](angle, q0)

        elif gate == "CX":
            validate_qubit(q0)
            validate_qubit(q1)
            qc.cx(q0, q1)

        elif gate == "CZ":
            validate_qubit(q0)
            validate_qubit(q1)
            qc.cz(q0, q1)

        elif gate == "SWAP":
            validate_qubit(q0)
            validate_qubit(q1)
            qc.swap(q0, q1)

        elif gate == "CCX":
            validate_qubit(q0)
            validate_qubit(q1)
            validate_qubit(q2)
            qc.ccx(q0, q1, q2)

        elif gate == "MEASURE_ALL":
            qc.measure(range(n_qubits), range(n_qubits))

        else:
            raise ValueError(
                f"Row {row_number + 1}: unsupported gate '{gate}'."
            )

    if qc.count_ops().get("measure", 0) == 0:
        qc.measure(range(n_qubits), range(n_qubits))

    return qc


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def coupling_map_for(backend_name, n_qubits):
    if backend_name == "Aer_FullyConnected":
        return None

    edges = []
    for i in range(n_qubits - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])

    return edges


def hardware_parameters(backend_name, noise_level):
    base = BACKEND_BASE[backend_name]
    scale = NOISE_SCALE[noise_level]

    if noise_level == "Ideal":
        return {
            "p1": 0.0,
            "p2": 0.0,
            "readout": 0.0,
            "t1_us": 1_000_000.0,
            "t2_us": 1_000_000.0,
        }

    p1 = base["p1"] * scale
    p2 = base["p2"] * scale
    readout = base["readout"] * scale
    t1_us = base["t1_us"] / scale
    t2_us = min(
        base["t2_us"] / scale,
        2.0 * t1_us,
    )

    return {
        "p1": float(p1),
        "p2": float(p2),
        "readout": float(readout),
        "t1_us": float(t1_us),
        "t2_us": float(t2_us),
    }


def canonical_feature_circuit(qc):
    return transpile(
        qc,
        basis_gates=FEATURE_BASIS,
        optimization_level=0,
        seed_transpiler=SEED,
    )


def is_quantum_gate(instruction):
    return instruction.operation.name not in {
        "measure",
        "barrier",
        "delay",
        "reset",
    }


def quantum_gate_count(qc):
    return sum(
        1
        for instruction in qc.data
        if is_quantum_gate(instruction)
    )


def circuit_features(qc):
    ops = qc.count_ops()

    single = 0
    multi = 0

    for instruction in qc.data:
        if not is_quantum_gate(instruction):
            continue

        n_operands = instruction.operation.num_qubits

        if n_operands == 1:
            single += 1
        elif n_operands > 1:
            multi += 1

    no_measure = qc.remove_final_measurements(inplace=False)
    depth = int(no_measure.depth() or 0)
    total = int(single + multi)

    density = total / max(
        depth * qc.num_qubits,
        1,
    )

    return {
        "Circuit_Depth": depth,
        "Circuit_Width": int(qc.width()),
        "Total_Gates": total,
        "Single_Qubit_Gates": int(single),
        "Multi_Qubit_Gates": int(multi),
        "Gate_Density": float(density),
        "H_Gates": int(ops.get("h", 0)),
        "X_Gates": int(ops.get("x", 0)),
        "Y_Gates": int(ops.get("y", 0)),
        "Z_Gates": int(ops.get("z", 0)),
        "RX_Gates": int(ops.get("rx", 0)),
        "RY_Gates": int(ops.get("ry", 0)),
        "RZ_Gates": int(ops.get("rz", 0)),
        "CX_Gates": int(ops.get("cx", 0)),
        "SWAP_Gates": int(ops.get("swap", 0)),
        "CCX_Gates": int(ops.get("ccx", 0)),
        "Measure_Gates": int(ops.get("measure", 0)),
    }


# ============================================================
# NOISE EXPOSURE PROFILING
# ============================================================

def combined_risk(base_error, t1_us, t2_us, duration_ns, n_qubits=1):
    if base_error <= 0:
        base_error = 0.0

    t1_ns = max(t1_us * 1000.0, 1.0)
    t2_ns = max(t2_us * 1000.0, 1.0)

    relaxation_loss = 1.0 - math.exp(-duration_ns / t1_ns)
    coherence_loss = 1.0 - math.exp(-duration_ns / t2_ns)

    survival = (
        (1.0 - base_error)
        * ((1.0 - relaxation_loss) ** n_qubits)
        * ((1.0 - coherence_loss) ** n_qubits)
    )

    return float(np.clip(1.0 - survival, 0.0, 1.0))


def gate_risk(gate_name, params):
    gate_name = gate_name.lower()

    if gate_name in {"x", "sx"}:
        risk = combined_risk(
            params["p1"],
            params["t1_us"],
            params["t2_us"],
            50.0,
            n_qubits=1,
        )
        reason = "1-qubit gate error + T1/T2 exposure"

    elif gate_name == "cx":
        risk = combined_risk(
            params["p2"],
            params["t1_us"],
            params["t2_us"],
            300.0,
            n_qubits=2,
        )
        reason = "2-qubit gate error + T1/T2 exposure"

    elif gate_name == "measure":
        risk = float(params["readout"])
        reason = "readout error"

    elif gate_name == "rz":
        risk = 0.0
        reason = "no explicit RZ noise in the current project noise model"

    else:
        risk = 0.0
        reason = "no explicit error channel assigned in the current model"

    return risk, reason


def risk_label(risk):
    if risk >= 0.03:
        return "High"
    if risk >= 0.01:
        return "Medium"
    if risk > 0:
        return "Low"
    return "None"


def noise_exposure_table(transpiled_qc, params):
    dag = circuit_to_dag(transpiled_qc)
    records = []
    operation_index = 0

    for layer_index, layer in enumerate(dag.layers(), start=1):
        for node in layer["graph"].op_nodes():
            operation_index += 1

            gate_name = node.op.name
            qubits = [
                transpiled_qc.find_bit(qarg).index
                for qarg in node.qargs
            ]

            risk, reason = gate_risk(
                gate_name,
                params,
            )

            records.append(
                {
                    "Operation": operation_index,
                    "Depth Layer": layer_index,
                    "Gate": gate_name.upper(),
                    "Qubits": ", ".join(map(str, qubits)) if qubits else "—",
                    "Estimated Local Noise Risk": risk,
                    "Risk Level": risk_label(risk),
                    "Why": reason,
                }
            )

    table = pd.DataFrame(records)

    if table.empty:
        return table, pd.DataFrame()

    layer_rows = []

    for layer_value, group in table.groupby("Depth Layer"):
        risks = group["Estimated Local Noise Risk"].to_numpy(dtype=float)
        layer_survival = float(np.prod(1.0 - risks))
        layer_risk = float(
            np.clip(
                1.0 - layer_survival,
                0.0,
                1.0,
            )
        )

        layer_rows.append(
            {
                "Depth Layer": int(layer_value),
                "Layer Noise Exposure": layer_risk,
                "Noisy Operations": int((risks > 0).sum()),
                "Total Operations": len(group),
            }
        )

    layers = pd.DataFrame(layer_rows)

    return table, layers


def total_exposure_from_table(table):
    if table.empty:
        return 0.0

    risks = table[
        "Estimated Local Noise Risk"
    ].to_numpy(dtype=float)

    return float(
        np.clip(
            1.0 - np.prod(1.0 - risks),
            0.0,
            1.0,
        )
    )


# ============================================================
# OPTIMIZATION CANDIDATES
# ============================================================

def build_candidate_rows(
    qc,
    algorithm_label,
    backend,
    noise,
    shots,
):
    feature_qc = canonical_feature_circuit(qc)
    raw = circuit_features(feature_qc)
    params = hardware_parameters(
        backend,
        noise,
    )

    coupling_map = coupling_map_for(
        backend,
        feature_qc.num_qubits,
    )

    model_rows = []
    display_rows = []
    transpiled_versions = {}

    for level in OPT_LEVELS:
        optimized = transpile(
            feature_qc,
            basis_gates=HARDWARE_BASIS,
            coupling_map=coupling_map,
            optimization_level=level,
            seed_transpiler=SEED + level,
        )

        transpiled_versions[level] = optimized

        no_measure = optimized.remove_final_measurements(
            inplace=False
        )

        optimized_depth = int(
            no_measure.depth() or 0
        )

        optimized_total = int(
            quantum_gate_count(optimized)
        )

        depth_reduction = (
            100.0
            * (
                raw["Circuit_Depth"]
                - optimized_depth
            )
            / max(
                raw["Circuit_Depth"],
                1,
            )
        )

        gate_reduction = (
            100.0
            * (
                raw["Total_Gates"]
                - optimized_total
            )
            / max(
                raw["Total_Gates"],
                1,
            )
        )

        exposure_table, layer_table = noise_exposure_table(
            optimized,
            params,
        )

        total_noise_exposure = total_exposure_from_table(
            exposure_table
        )

        row = {
            "Algorithm": algorithm_label,
            "Num_Qubits": int(feature_qc.num_qubits),
            **raw,
            "Backend": backend,
            "Optimization_Level": int(level),
            "Noise_Model": noise,
            "Shots": int(shots),
            "Gate_Error_Rate": float(params["p2"]),
            "Readout_Error": float(params["readout"]),
            "Average_T1": float(params["t1_us"]),
            "Average_T2": float(params["t2_us"]),
            "Optimized_Depth": optimized_depth,
            "Optimized_Total_Gates": optimized_total,
            "Depth_Reduction": depth_reduction,
            "Gate_Reduction": gate_reduction,
        }

        model_rows.append(row)

        display_rows.append(
            {
                "Optimization Level": int(level),
                "Optimized Depth": optimized_depth,
                "Optimized Gates": optimized_total,
                "Estimated Circuit Noise Exposure": total_noise_exposure,
                "Depth Reduction vs Raw (%)": round(
                    depth_reduction,
                    2,
                ),
                "Gate Reduction vs Raw (%)": round(
                    gate_reduction,
                    2,
                ),
            }
        )

    display_df = pd.DataFrame(display_rows)

    baseline_depth = int(
        display_df.loc[
            display_df["Optimization Level"] == 0,
            "Optimized Depth",
        ].iloc[0]
    )

    baseline_gates = int(
        display_df.loc[
            display_df["Optimization Level"] == 0,
            "Optimized Gates",
        ].iloc[0]
    )

    display_df["Depth Improvement vs Level 0 (%)"] = (
        100.0
        * (
            baseline_depth
            - display_df["Optimized Depth"]
        )
        / max(
            baseline_depth,
            1,
        )
    )

    display_df["Gate Improvement vs Level 0 (%)"] = (
        100.0
        * (
            baseline_gates
            - display_df["Optimized Gates"]
        )
        / max(
            baseline_gates,
            1,
        )
    )

    return (
        pd.DataFrame(model_rows),
        display_df,
        feature_qc,
        transpiled_versions,
        params,
    )


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_candidates(model, feature_rows):
    x = feature_rows.copy()

    expected_columns = None

    try:
        preprocessor = model.named_steps.get(
            "preprocess"
        )

        if hasattr(
            preprocessor,
            "feature_names_in_",
        ):
            expected_columns = list(
                preprocessor.feature_names_in_
            )

    except Exception:
        expected_columns = None

    if expected_columns is not None:
        missing = [
            column
            for column in expected_columns
            if column not in x.columns
        ]

        if missing:
            raise ValueError(
                "The dashboard could not create these model inputs: "
                + ", ".join(missing)
            )

        x = x[expected_columns]

    predictions = np.asarray(
        model.predict(x),
        dtype=float,
    )

    return np.clip(
        predictions,
        0.0,
        1.0,
    )


def recommend_with_model(scored):
    best_prediction = scored[
        "Predicted Success Probability"
    ].max()

    candidates = scored[
        scored["Predicted Success Probability"]
        >= best_prediction - 0.01
    ].copy()

    best = candidates.sort_values(
        by=[
            "Estimated Circuit Noise Exposure",
            "Optimized Depth",
            "Optimized Gates",
            "Optimization Level",
        ],
        ascending=True,
    ).iloc[0]

    return int(
        best["Optimization Level"]
    )


def recommend_without_model(scored):
    normalized = scored.copy()

    for column in [
        "Estimated Circuit Noise Exposure",
        "Optimized Depth",
        "Optimized Gates",
    ]:
        min_value = normalized[column].min()
        max_value = normalized[column].max()

        if max_value == min_value:
            normalized[f"N_{column}"] = 0.0
        else:
            normalized[f"N_{column}"] = (
                normalized[column]
                - min_value
            ) / (
                max_value
                - min_value
            )

    normalized["Heuristic Score"] = (
        0.60
        * normalized[
            "N_Estimated Circuit Noise Exposure"
        ]
        + 0.25
        * normalized[
            "N_Optimized Depth"
        ]
        + 0.15
        * normalized[
            "N_Optimized Gates"
        ]
    )

    best = normalized.sort_values(
        [
            "Heuristic Score",
            "Optimization Level",
        ]
    ).iloc[0]

    return int(
        best["Optimization Level"]
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚛️ Advisor Controls")

uploaded_model = st.sidebar.file_uploader(
    "Upload trained Random Forest model",
    type=["joblib", "pkl"],
    help=(
        "Use random_forest_success_probability.joblib "
        "saved by the project notebook."
    ),
)

uploaded_dataset = st.sidebar.file_uploader(
    "Upload clean benchmark dataset",
    type=["xlsx", "csv"],
    help=(
        "Optional. Used by the Dataset Analytics page."
    ),
)

local_model_path = find_first_existing(
    DEFAULT_MODEL_NAMES
)

local_dataset_path = find_first_existing(
    DEFAULT_DATASET_NAMES
)

model = None
model_source = None

try:
    if uploaded_model is not None:
        model = load_uploaded_model(
            uploaded_model
        )
        model_source = uploaded_model.name

    elif local_model_path is not None:
        model = load_local_model(
            str(local_model_path)
        )
        model_source = local_model_path.name

except Exception as exc:
    st.sidebar.error(
        f"Could not load model: {exc}"
    )

dataset = None
dataset_source = None

try:
    if uploaded_dataset is not None:
        dataset = read_uploaded_dataset(
            uploaded_dataset
        )
        dataset_source = uploaded_dataset.name

    elif local_dataset_path is not None:
        dataset = read_dataset_from_path(
            str(local_dataset_path)
        )
        dataset_source = local_dataset_path.name

except Exception as exc:
    st.sidebar.error(
        f"Could not load dataset: {exc}"
    )

if model is not None:
    st.sidebar.success(
        f"ML model ready: {model_source}"
    )
else:
    st.sidebar.warning(
        "ML model not loaded. "
        "The dashboard will still analyze depth and noise, "
        "but recommendation will use a clearly labeled heuristic."
    )

if dataset is not None:
    st.sidebar.success(
        f"Dataset ready: {dataset_source}"
    )
else:
    st.sidebar.info(
        "Dataset is optional for circuit analysis."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>AI-Powered Quantum Circuit Optimization Advisor</h1>
        <p>
            Give the system a quantum circuit and it will analyze
            <b>depth</b>, <b>gate structure</b>,
            <b>noise-exposed depth layers</b>,
            <b>optimization levels 0–3</b>,
            and recommend where optimization is required.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "🧪 Analyze Circuit",
        "📊 Dataset",
        "🤖 Model",
        "🧭 System Flow",
    ]
)

tab_analyze, tab_dataset, tab_model, tab_flow = tabs


# ============================================================
# ANALYZE CIRCUIT
# ============================================================

with tab_analyze:
    st.subheader(
        "Create or Give a Quantum Circuit"
    )

    source = st.radio(
        "Circuit input method",
        [
            "Built-in Algorithm",
            "Manual Gate Builder",
            "Paste / Upload OpenQASM 2",
        ],
        horizontal=True,
    )

    algorithm_label = "Custom"
    qc = None

    if source == "Built-in Algorithm":
        c1, c2 = st.columns(2)

        with c1:
            algorithm_label = st.selectbox(
                "Algorithm",
                ALGORITHMS,
            )

        with c2:
            qubits = st.selectbox(
                "Number of qubits",
                ALGORITHM_QUBITS[
                    algorithm_label
                ],
            )

        rng = np.random.default_rng(SEED)
        qc = GENERATORS[algorithm_label](
            int(qubits),
            rng,
        )

    elif source == "Manual Gate Builder":
        number_of_qubits = st.number_input(
            "Number of qubits",
            min_value=1,
            max_value=12,
            value=3,
            step=1,
        )

        algorithm_label = st.selectbox(
            "Circuit family label for the ML model",
            ["Custom"] + ALGORITHMS,
            help=(
                "Choose Custom if it is not one of the "
                "9 benchmark families."
            ),
        )

        st.caption(
            "Use -1 for unused Q1/Q2 fields. "
            "Angles are in radians."
        )

        operations = st.data_editor(
            default_gate_editor(),
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Gate": st.column_config.SelectboxColumn(
                    "Gate",
                    options=[
                        "H",
                        "X",
                        "Y",
                        "Z",
                        "RX",
                        "RY",
                        "RZ",
                        "CX",
                        "CZ",
                        "SWAP",
                        "CCX",
                        "MEASURE_ALL",
                    ],
                    required=True,
                )
            },
            key="manual_builder",
        )

        try:
            qc = build_manual_circuit(
                int(number_of_qubits),
                operations,
            )
        except Exception as exc:
            st.error(
                f"Manual circuit error: {exc}"
            )
            qc = None

    else:
        algorithm_label = st.selectbox(
            "Closest benchmark family",
            ["Custom"] + ALGORITHMS,
            help=(
                "For an arbitrary circuit use Custom. "
                "The model was trained only on the benchmark families, "
                "so arbitrary-circuit ML predictions should be treated as indicative."
            ),
        )

        input_mode = st.radio(
            "OpenQASM input",
            [
                "Paste QASM",
                "Upload .qasm",
            ],
            horizontal=True,
        )

        qasm_text = ""

        if input_mode == "Paste QASM":
            qasm_text = st.text_area(
                "Paste OpenQASM 2",
                value=(
                    'OPENQASM 2.0;\n'
                    'include "qelib1.inc";\n'
                    'qreg q[2];\n'
                    'creg c[2];\n'
                    'h q[0];\n'
                    'cx q[0],q[1];\n'
                    'measure q -> c;\n'
                ),
                height=220,
            )

        else:
            qasm_file = st.file_uploader(
                "Upload QASM circuit",
                type=["qasm"],
                key="qasm_upload",
            )

            if qasm_file is not None:
                qasm_text = qasm_file.getvalue().decode(
                    "utf-8"
                )

        if qasm_text.strip():
            try:
                qc = QuantumCircuit.from_qasm_str(
                    qasm_text
                )
            except Exception as exc:
                st.error(
                    f"Could not parse QASM: {exc}"
                )
                qc = None

    st.markdown("### Execution Conditions")

    c1, c2, c3 = st.columns(3)

    with c1:
        backend = st.selectbox(
            "Backend profile",
            BACKEND_PROFILES,
        )

    with c2:
        noise = st.selectbox(
            "Noise level",
            NOISE_LEVELS,
            index=2,
        )

    with c3:
        shots = st.selectbox(
            "Shots",
            SHOT_LEVELS,
            index=2,
        )

    analyze = st.button(
        "Analyze Circuit & Find Optimization",
        type="primary",
        use_container_width=True,
        disabled=(qc is None),
    )

    if analyze and qc is not None:
        try:
            (
                model_rows,
                comparison,
                feature_qc,
                transpiled_versions,
                params,
            ) = build_candidate_rows(
                qc,
                algorithm_label,
                backend,
                noise,
                shots,
            )

            raw = circuit_features(
                feature_qc
            )

            st.divider()
            st.subheader(
                "1. Circuit Structure"
            )

            m1, m2, m3, m4, m5 = st.columns(5)

            m1.metric(
                "Qubits",
                feature_qc.num_qubits,
            )
            m2.metric(
                "Raw Depth",
                raw["Circuit_Depth"],
            )
            m3.metric(
                "Total Gates",
                raw["Total_Gates"],
            )
            m4.metric(
                "Multi-Qubit Gates",
                raw["Multi_Qubit_Gates"],
            )
            m5.metric(
                "Gate Density",
                f"{raw['Gate_Density']:.3f}",
            )

            with st.expander(
                "View circuit text diagram",
                expanded=True,
            ):
                st.code(
                    str(
                        qc.draw(
                            output="text"
                        )
                    ),
                    language="text",
                )

            st.subheader(
                "2. Optimization Levels 0–3"
            )

            if model is not None:
                predictions = predict_candidates(
                    model,
                    model_rows,
                )

                comparison[
                    "Predicted Success Probability"
                ] = predictions

                recommended_level = recommend_with_model(
                    comparison
                )

                recommendation_type = (
                    "ML-based recommendation"
                )

            else:
                recommended_level = recommend_without_model(
                    comparison
                )

                recommendation_type = (
                    "Structural/noise heuristic fallback"
                )

            baseline = comparison[
                comparison[
                    "Optimization Level"
                ] == 0
            ].iloc[0]

            recommended = comparison[
                comparison[
                    "Optimization Level"
                ] == recommended_level
            ].iloc[0]

            st.dataframe(
                comparison.round(5),
                use_container_width=True,
                hide_index=True,
            )

            o1, o2, o3, o4 = st.columns(4)

            o1.metric(
                "Recommended Level",
                f"Level {recommended_level}",
            )

            o2.metric(
                "Depth",
                int(
                    recommended[
                        "Optimized Depth"
                    ]
                ),
                delta=(
                    int(
                        recommended[
                            "Optimized Depth"
                        ]
                    )
                    - int(
                        baseline[
                            "Optimized Depth"
                        ]
                    )
                ),
                delta_color="inverse",
            )

            o3.metric(
                "Gate Count",
                int(
                    recommended[
                        "Optimized Gates"
                    ]
                ),
                delta=(
                    int(
                        recommended[
                            "Optimized Gates"
                        ]
                    )
                    - int(
                        baseline[
                            "Optimized Gates"
                        ]
                    )
                ),
                delta_color="inverse",
            )

            o4.metric(
                "Estimated Noise Exposure",
                f"{recommended['Estimated Circuit Noise Exposure']:.3%}",
            )

            st.caption(
                recommendation_type
            )

            if model is not None:
                st.metric(
                    "Predicted Success Probability",
                    f"{recommended['Predicted Success Probability']:.4f}",
                )

            depth_chart = comparison[
                [
                    "Optimization Level",
                    "Optimized Depth",
                ]
            ].set_index(
                "Optimization Level"
            )

            st.markdown(
                "#### Depth by Optimization Level"
            )

            st.bar_chart(
                depth_chart
            )

            st.subheader(
                "3. Where Is Noise Most Likely to Affect the Circuit?"
            )

            recommended_circuit = transpiled_versions[
                recommended_level
            ]

            exposure_table, layer_table = noise_exposure_table(
                recommended_circuit,
                params,
            )

            if exposure_table.empty:
                st.info(
                    "No operations found in the transpiled circuit."
                )
            else:
                noisy_operations = exposure_table[
                    exposure_table[
                        "Estimated Local Noise Risk"
                    ] > 0
                ].copy()

                noisy_layers = layer_table[
                    layer_table[
                        "Layer Noise Exposure"
                    ] > 0
                ].copy()

                n1, n2, n3, n4 = st.columns(4)

                n1.metric(
                    "Transpiled Depth",
                    int(
                        recommended_circuit
                        .remove_final_measurements(
                            inplace=False
                        )
                        .depth()
                        or 0
                    ),
                )

                n2.metric(
                    "Noise-Exposed Layers",
                    len(noisy_layers),
                )

                n3.metric(
                    "Noise-Exposed Operations",
                    len(noisy_operations),
                )

                high_risk_ops = int(
                    (
                        noisy_operations[
                            "Risk Level"
                        ] == "High"
                    ).sum()
                )

                n4.metric(
                    "High-Risk Operations",
                    high_risk_ops,
                )

                st.info(
                    "Important: this is a **noise exposure map**, "
                    "not a claim that a random error definitely occurred at that exact gate. "
                    "It identifies the depth layers and operations exposed to the "
                    "depolarizing, T1/T2, and readout noise used by your project model."
                )

                if not noisy_layers.empty:
                    st.markdown(
                        "#### Noise Exposure by Circuit Depth Layer"
                    )

                    layer_chart = noisy_layers[
                        [
                            "Depth Layer",
                            "Layer Noise Exposure",
                        ]
                    ].set_index(
                        "Depth Layer"
                    )

                    st.bar_chart(
                        layer_chart
                    )

                    top_layers = noisy_layers.sort_values(
                        "Layer Noise Exposure",
                        ascending=False,
                    ).head(10)

                    st.markdown(
                        "#### Most Noise-Sensitive Depths"
                    )

                    st.dataframe(
                        top_layers.round(6),
                        use_container_width=True,
                        hide_index=True,
                    )

                st.markdown(
                    "#### Gate-Level Noise Hotspots"
                )

                st.dataframe(
                    noisy_operations.sort_values(
                        "Estimated Local Noise Risk",
                        ascending=False,
                    ).head(50).round(6),
                    use_container_width=True,
                    hide_index=True,
                )

            st.subheader(
                "4. Where Is Optimization Required?"
            )

            messages = []

            depth_improvement = float(
                recommended[
                    "Depth Improvement vs Level 0 (%)"
                ]
            )

            gate_improvement = float(
                recommended[
                    "Gate Improvement vs Level 0 (%)"
                ]
            )

            baseline_noise = float(
                baseline[
                    "Estimated Circuit Noise Exposure"
                ]
            )

            recommended_noise = float(
                recommended[
                    "Estimated Circuit Noise Exposure"
                ]
            )

            if recommended_level == 0:
                messages.append(
                    "Level 0 is already competitive under the selected conditions; "
                    "aggressive transpiler optimization is not required."
                )
            else:
                messages.append(
                    f"Optimization is recommended at **Level {recommended_level}**."
                )

            if depth_improvement > 0.1:
                messages.append(
                    f"Depth is reduced by approximately **{depth_improvement:.2f}%** "
                    "relative to hardware Level 0."
                )
            elif depth_improvement < -0.1:
                messages.append(
                    "The recommended candidate does not reduce hardware depth; "
                    "its advantage comes from the overall predicted/noise trade-off."
                )

            if gate_improvement > 0.1:
                messages.append(
                    f"Gate count is reduced by approximately **{gate_improvement:.2f}%** "
                    "relative to hardware Level 0."
                )

            if recommended_noise < baseline_noise:
                noise_drop = (
                    100.0
                    * (
                        baseline_noise
                        - recommended_noise
                    )
                    / max(
                        baseline_noise,
                        1e-12,
                    )
                )

                messages.append(
                    f"Estimated cumulative noise exposure decreases by "
                    f"approximately **{noise_drop:.2f}%**."
                )

            if raw["Multi_Qubit_Gates"] > 0:
                messages.append(
                    "Multi-qubit operations are a key optimization target because "
                    "the current backend profiles assign larger error to two-qubit gates."
                )

            for message in messages:
                st.markdown(
                    f"- {message}"
                )

            st.markdown(
                "#### Recommended Circuit"
            )

            st.code(
                str(
                    recommended_circuit.draw(
                        output="text"
                    )
                ),
                language="text",
            )

            export = comparison.copy()
            export[
                "Recommended"
            ] = (
                export[
                    "Optimization Level"
                ]
                == recommended_level
            )

            st.download_button(
                "Download optimization analysis (CSV)",
                data=export.to_csv(
                    index=False
                ).encode(
                    "utf-8"
                ),
                file_name=(
                    "quantum_optimization_analysis.csv"
                ),
                mime="text/csv",
            )

        except Exception as exc:
            st.exception(exc)


# ============================================================
# DATASET PAGE
# ============================================================

with tab_dataset:
    st.subheader(
        "Benchmark Dataset Analytics"
    )

    if dataset is None:
        st.info(
            "Upload the clean `.xlsx` dataset in the sidebar "
            "or place `quantum_circuit_dataset_clean.xlsx` "
            "in the same folder as app.py."
        )

    else:
        d = dataset.copy()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Rows",
            f"{len(d):,}",
        )

        c2.metric(
            "Columns",
            len(d.columns),
        )

        c3.metric(
            "Algorithms",
            d["Algorithm"].nunique()
            if "Algorithm" in d.columns
            else "—",
        )

        c4.metric(
            "Missing Values",
            int(
                d.isna().sum().sum()
            ),
        )

        st.dataframe(
            d.head(30),
            use_container_width=True,
            hide_index=True,
        )

        left, right = st.columns(2)

        with left:
            if "Algorithm" in d.columns:
                st.markdown(
                    "#### Algorithm Distribution"
                )
                st.bar_chart(
                    d[
                        "Algorithm"
                    ].value_counts().sort_index()
                )

            if "Optimization_Level" in d.columns:
                st.markdown(
                    "#### Optimization Levels"
                )
                st.bar_chart(
                    d[
                        "Optimization_Level"
                    ].value_counts().sort_index()
                )

        with right:
            if "Noise_Model" in d.columns:
                st.markdown(
                    "#### Noise Distribution"
                )
                st.bar_chart(
                    d[
                        "Noise_Model"
                    ].value_counts()
                )

            if "Shots" in d.columns:
                st.markdown(
                    "#### Shots Distribution"
                )
                st.bar_chart(
                    d[
                        "Shots"
                    ].value_counts().sort_index()
                )

        constant_columns = [
            column
            for column in d.columns
            if d[column].nunique(
                dropna=False
            ) <= 1
        ]

        st.markdown(
            "#### Zero-Variance / Constant Columns"
        )

        if constant_columns:
            st.warning(
                "These columns contain no variation and should not be "
                "used as informative ML features: "
                + ", ".join(
                    constant_columns
                )
            )
        else:
            st.success(
                "No constant columns detected."
            )

        if {
            "Algorithm",
            "Success_Probability",
        }.issubset(
            d.columns
        ):
            st.markdown(
                "#### Mean Success Probability by Algorithm"
            )

            st.bar_chart(
                d.groupby(
                    "Algorithm"
                )[
                    "Success_Probability"
                ].mean().sort_values(
                    ascending=False
                )
            )


# ============================================================
# MODEL PAGE
# ============================================================

with tab_model:
    st.subheader(
        "Random Forest Regression"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "MAE",
        f"{LATEST_METRICS['MAE']:.6f}",
    )

    c2.metric(
        "MSE",
        f"{LATEST_METRICS['MSE']:.6f}",
    )

    c3.metric(
        "RMSE",
        f"{LATEST_METRICS['RMSE']:.6f}",
    )

    c4.metric(
        "R²",
        f"{LATEST_METRICS['R²']:.6f}",
    )

    st.markdown(
        """
        **Target:** `Success_Probability`

        The target is continuous, so the prediction stage is a
        **regression problem**.

        The model used by this dashboard should be the exact
        `random_forest_success_probability.joblib`
        saved by the revised project notebook.
        """
    )

    if model is not None:
        st.success(
            "The trained model is loaded and live prediction is enabled."
        )
    else:
        st.warning(
            "Upload the trained `.joblib` model in the sidebar "
            "to activate ML prediction."
        )


# ============================================================
# SYSTEM FLOW PAGE
# ============================================================

with tab_flow:
    st.subheader(
        "Complete Project Flow"
    )

    st.markdown(
        """
        ### Quantum Circuit
        Built-in benchmark, manually created circuit, or OpenQASM circuit.

        ↓

        ### Structural Analysis
        Number of qubits, raw circuit depth, gate count,
        single/multi-qubit gates and gate density.

        ↓

        ### Backend + Noise Profile
        Backend connectivity, gate error, readout error,
        T1/T2 and selected noise level.

        ↓

        ### Noise Exposure Map
        The dashboard maps the transpiled circuit into depth layers
        and identifies gates exposed to the project's one-qubit,
        two-qubit and readout noise channels.

        ↓

        ### Optimization Search
        The same circuit is transpiled at Qiskit optimization
        levels 0, 1, 2 and 3.

        ↓

        ### Success Prediction
        The Random Forest regressor predicts success probability
        for each candidate.

        ↓

        ### Recommendation
        The dashboard recommends an optimization level and shows
        where depth, gate count and estimated noise exposure improve.
        """
    )
