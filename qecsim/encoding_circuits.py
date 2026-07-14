"""Statevector simulation of stabilizer-code encoding.

`encode_bit_flip_circuit` is a 2-CNOT gate circuit for the
3-qubit bit-flip code. `encode_general` handles any code (Steane, Shor)
by projecting |0...0> onto the codespace via the stabilizer group.
"""

import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI_MATRIX = {(0, 0): I2, (1, 0): X, (0, 1): Z, (1, 1): Y}

KET0 = np.array([1, 0], dtype=complex)
KET1 = np.array([0, 1], dtype=complex)


def _embed_single_qubit_gate(gate_2x2: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """2**n x 2**n matrix applying `gate_2x2` to `qubit`, identity elsewhere."""
    mats = [I2] * n
    mats[qubit] = gate_2x2
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def apply_single_qubit_gate(state: np.ndarray, gate_2x2: np.ndarray, qubit: int,
                             n: int) -> np.ndarray:
    """Apply a single-qubit gate to `qubit` of an n-qubit statevector."""
    return _embed_single_qubit_gate(gate_2x2, qubit, n) @ state


def apply_cnot(state: np.ndarray, control: int, target: int, n: int) -> np.ndarray:
    """Apply CNOT(control -> target) by permuting basis-state amplitudes:
    flip the `target` bit of every basis index whose `control` bit is 1."""
    dim = 2 ** n
    new_state = np.zeros(dim, dtype=complex)
    for idx in range(dim):
        amp = state[idx]
        if amp == 0:
            continue
        # Qubit 0 is the MSB of idx (matches kron ordering above).
        control_bit = (idx >> (n - 1 - control)) & 1
        if control_bit:
            new_idx = idx ^ (1 << (n - 1 - target))
        else:
            new_idx = idx
        new_state[new_idx] += amp
    return new_state


def encode_bit_flip_circuit(logical_bit: int) -> np.ndarray:
    """Prepare |logical_bit>|0>|0>, then CNOT(0,1), CNOT(0,2), giving
    |000> or |111> for logical 0 / 1."""
    n = 3
    if logical_bit == 0:
        state = np.zeros(2 ** n, dtype=complex)
        state[0] = 1.0  # |000>
    elif logical_bit == 1:
        state = np.zeros(2 ** n, dtype=complex)
        state[0b100] = 1.0  # |1>|0>|0>, qubit 0 = MSB
    else:
        raise ValueError("logical_bit must be 0 or 1")

    state = apply_cnot(state, control=0, target=1, n=n)
    state = apply_cnot(state, control=0, target=2, n=n)
    return state


def _pauli_vector_to_matrix(vec: np.ndarray) -> np.ndarray:
    """Convert a length-2n binary symplectic Pauli vector into its
    2**n x 2**n matrix via a kron product of single-qubit Pauli matrices.

    Assumes a +1-phase convention per qubit (true for all generators in codes.py).
    """
    n = len(vec) // 2
    mats = []
    for i in range(n):
        x, z = int(vec[i]), int(vec[n + i])
        mats.append(_PAULI_MATRIX[(x, z)])
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def stabilizer_projector(code, n_full: int) -> np.ndarray:
    """Build the 2**n x 2**n projector onto the codespace:
    Pi = prod_j (I + G_j) / 2 over the generator matrices G_j.

    Multiplying the literal generator matrices (rather than averaging over
    the whole stabilizer group) keeps relative phase correct for CSS codes,
    where X-type * Z-type generators pick up a factor of i (X*Z = -iY).
    """
    dim = 2 ** n_full
    identity = np.eye(dim, dtype=complex)
    proj = identity.copy()
    for vec in code.stabilizers:
        g_mat = _pauli_vector_to_matrix(vec)
        proj = proj @ ((identity + g_mat) / 2)
    return proj


def encode_general(code, logical_bit: int) -> np.ndarray:
    """Encode |logical_bit> by projecting |0...0> onto the codespace to
    get |0_L>, then (for logical_bit=1) applying logical X_bar and
    renormalizing."""
    if logical_bit not in (0, 1):
        raise ValueError("logical_bit must be 0 or 1")

    n = code.n
    proj = stabilizer_projector(code, n)
    zero_state = np.zeros(2 ** n, dtype=complex)
    zero_state[0] = 1.0

    zero_l = proj @ zero_state
    norm = np.linalg.norm(zero_l)
    if norm < 1e-12:
        raise RuntimeError(
            "Projector applied to |0...0> vanished -- |0...0> is orthogonal "
            "to the codespace, which should not happen for these codes."
        )
    zero_l = zero_l / norm

    if logical_bit == 0:
        return zero_l

    x_bar_matrix = _pauli_vector_to_matrix(code.logical_x[0])
    one_l = x_bar_matrix @ zero_l
    one_l = one_l / np.linalg.norm(one_l)
    return one_l


def verify_encoded_state(code, state: np.ndarray, n: int) -> dict:
    """Check that `state` is a +1 eigenstate of every stabilizer generator.
    Returns {label: ||S|psi> - |psi>||}, which should be near 0 for each."""
    results = {}
    for label, vec in zip(code.stabilizer_labels, code.stabilizers):
        mat = _pauli_vector_to_matrix(vec)
        diff = mat @ state - state
        results[label] = float(np.linalg.norm(diff))
    return results
