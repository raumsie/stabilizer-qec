"""Functions that build four classic small stabilizer codes.

Each one returns a `StabilizerCode`. Creating it validates that the
generators commute and the logical operators have the right commutation
relations, raising `ValueError` if not.
"""

from .stabilizer_code import StabilizerCode


def bit_flip_code() -> StabilizerCode:
    """[[3,1,1]] repetition code: |0>/|1> -> |000>/|111>, corrects a single
    X error via majority vote. Doesn't help with Z errors."""
    return StabilizerCode(
        name="bit_flip_3",
        n=3, k=1,
        stabilizer_labels=["ZZI", "IZZ"],
        logical_x_labels=["XXX"],
        logical_z_labels=["ZZZ"],
        distance=1,
    )


def phase_flip_code() -> StabilizerCode:
    """[[3,1,1]] phase-flip code:
    |0>/|1> -> |+++>/|--->, corrects a single Z error.
    Doesn't help with X errors."""
    return StabilizerCode(
        name="phase_flip_3",
        n=3, k=1,
        stabilizer_labels=["XXI", "IXX"],
        logical_x_labels=["XXX"],
        logical_z_labels=["ZZZ"],
        distance=1,
    )


def steane_code() -> StabilizerCode:
    """[[7,1,3]] Steane code: a CSS code built from the
    [7,4,3] Hamming code, using the same parity-check matrix for
    both X-type and Z-type stabilizers. Corrects any single-qubit error."""
    x_stabilizers = ["IIIXXXX", "IXXIIXX", "XIXIXIX"]
    z_stabilizers = ["IIIZZZZ", "IZZIIZZ", "ZIZIZIZ"]
    return StabilizerCode(
        name="steane_7_1_3",
        n=7, k=1,
        stabilizer_labels=x_stabilizers + z_stabilizers,
        logical_x_labels=["XXXXXXX"],
        logical_z_labels=["ZZZZZZZ"],
        distance=3,
    )


def shor_code() -> StabilizerCode:
    """[[9,1,3]] Shor code: concatenates the 3-qubit
    phase-flip code with the 3-qubit bit-flip code,
    correcting X and Z errors simultaneously. 9 qubits form three blocks
    of 3 ({1,2,3},{4,5,6},{7,8,9}). Z stabilizers catch bit
    flips, X stabilizers catch relative phase flips.

    Logical Z is all nine Z's. Logical X is X1 X2 X3."""
    z_stabilizers = [
        "ZZIIIIIII",  # Z1 Z2
        "IZZIIIIII",  # Z2 Z3
        "IIIZZIIII",  # Z4 Z5
        "IIIIZZIII",  # Z5 Z6
        "IIIIIIZZI",  # Z7 Z8
        "IIIIIIIZZ",  # Z8 Z9
    ]
    x_stabilizers = [
        "XXXXXXIII",  # X1 X2 X3 X4 X5 X6
        "IIIXXXXXX",  # X4 X5 X6 X7 X8 X9
    ]
    return StabilizerCode(
        name="shor_9_1_3",
        n=9, k=1,
        stabilizer_labels=z_stabilizers + x_stabilizers,
        logical_x_labels=["XXXIIIIII"],  # X1 X2 X3
        logical_z_labels=["ZZZZZZZZZ"],  # all nine Z
        distance=3,
    )
