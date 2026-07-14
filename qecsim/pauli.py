"""Binary symplectic (Gottesman) representation of n-qubit Pauli operators.

Each Pauli string is a length-2n uint8 vector v = (x_1..x_n, z_1..z_n),
where qubit i carries (x_i, z_i): (0,0)->I, (1,0)->X, (0,1)->Z, (1,1)->Y.

Note: this discards the +-1/+-i phase of the operator
but it cannot distinguish +XZ from -XZ.
"""

import numpy as np

_LABEL_TO_XZ = {
    "I": (0, 0),
    "X": (1, 0),
    "Z": (0, 1),
    "Y": (1, 1),
}
_XZ_TO_LABEL = {v: k for k, v in _LABEL_TO_XZ.items()}


def pauli_from_label(label: str) -> np.ndarray:
    """Parse a string like "XIZY" into a length-2n binary symplectic vector."""
    n = len(label)
    vec = np.zeros(2 * n, dtype=np.uint8)
    for i, ch in enumerate(label.upper()):
        if ch not in _LABEL_TO_XZ:
            raise ValueError(f"Invalid Pauli character {ch!r} in label {label!r}")
        x, z = _LABEL_TO_XZ[ch]
        vec[i] = x
        vec[n + i] = z
    return vec


def pauli_to_label(vec: np.ndarray) -> str:
    """Inverse of pauli_from_label."""
    vec = np.asarray(vec)
    n = len(vec) // 2
    chars = []
    for i in range(n):
        x = int(vec[i])
        z = int(vec[n + i])
        chars.append(_XZ_TO_LABEL[(x, z)])
    return "".join(chars)


def symplectic_product(a: np.ndarray, b: np.ndarray) -> int:
    """sum_i (a_x[i]*b_z[i] + a_z[i]*b_x[i]) mod 2; 0 iff a and b commute."""
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a) // 2
    ax, az = a[:n], a[n:]
    bx, bz = b[:n], b[n:]
    total = int(np.sum(ax.astype(np.int64) * bz.astype(np.int64) +
                        az.astype(np.int64) * bx.astype(np.int64)))
    return total % 2


def commutes(a: np.ndarray, b: np.ndarray) -> bool:
    """True if Pauli vectors a and b commute (up to phase)."""
    return symplectic_product(a, b) == 0


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose two Pauli operators up to phase: elementwise XOR (mod-2 sum)."""
    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)
    return np.bitwise_xor(a, b)


def weight(vec: np.ndarray) -> int:
    """Hamming weight: number of qubits where the Pauli isn't I."""
    vec = np.asarray(vec)
    n = len(vec) // 2
    x, z = vec[:n], vec[n:]
    return int(np.sum((x | z) != 0))