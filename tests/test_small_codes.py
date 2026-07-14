"""Tests for the custom stabilizer-theory small-code toolkit."""

import itertools

import numpy as np
import pytest

from qecsim import pauli
from qecsim.codes import bit_flip_code, phase_flip_code, steane_code, shor_code
from qecsim.encoding_circuits import (
    encode_bit_flip_circuit,
    encode_general,
    verify_encoded_state,
)
from qecsim.simulate import simulate_logical_error_rate


ALL_CODE_BUILDERS = {
    "bit_flip_3": bit_flip_code,
    "phase_flip_3": phase_flip_code,
    "steane_7_1_3": steane_code,
    "shor_9_1_3": shor_code,
}


# ----------------------------------------------------------------------
# Building a code / validation
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name,build_fn", ALL_CODE_BUILDERS.items())
def test_code_constructs_without_raising(name, build_fn):
    code = build_fn()
    assert code.n > 0
    assert code.k == 1
    assert len(code.stabilizers) == code.n - code.k
    # Group size must be exactly 2**(n-k).
    assert len(code.stabilizer_group) == 2 ** (code.n - code.k)


@pytest.mark.parametrize("name,build_fn", ALL_CODE_BUILDERS.items())
def test_logical_operators_anticommute_with_their_partner(name, build_fn):
    code = build_fn()
    for i in range(code.k):
        assert not pauli.commutes(code.logical_x[i], code.logical_z[i])


# ----------------------------------------------------------------------
# Decoder: all weight-1 errors of the relevant type must be corrected
# ----------------------------------------------------------------------
def _weight1_errors(n, letters):
    """Yield all single-qubit Pauli error vectors using the given letters
    (subset of X, Y, Z) on each of the n qubits."""
    for q in range(n):
        for letter in letters:
            label = ["I"] * n
            label[q] = letter
            yield pauli.pauli_from_label("".join(label))


def _assert_all_corrected(code, letters):
    for error in _weight1_errors(code.n, letters):
        synd = code.syndrome(error)
        correction = code.decode(synd)
        assert not code.is_logical_error(error, correction), (
            f"{code.name}: failed to correct error {pauli.pauli_to_label(error)} "
            f"(syndrome {synd}, correction {pauli.pauli_to_label(correction)})"
        )


def test_steane_corrects_all_weight1_errors():
    code = steane_code()
    _assert_all_corrected(code, "XYZ")


def test_shor_corrects_all_weight1_errors():
    code = shor_code()
    _assert_all_corrected(code, "XYZ")


def test_bit_flip_corrects_all_weight1_x_errors():
    code = bit_flip_code()
    _assert_all_corrected(code, "X")


def test_phase_flip_corrects_all_weight1_z_errors():
    code = phase_flip_code()
    _assert_all_corrected(code, "Z")


def test_bit_flip_code_cannot_correct_z_errors():
    """Documented limitation: bit-flip code stabilizers are all Z-type, so
    a Z error is invisible (trivial syndrome) and is misdiagnosed as "no
    error", leaving an undetected logical error."""
    code = bit_flip_code()
    error = pauli.pauli_from_label("ZII")
    synd = code.syndrome(error)
    assert synd == (0, 0)
    correction = code.decode(synd)
    assert code.is_logical_error(error, correction)


# ----------------------------------------------------------------------
# Encoding circuits / general projector-based encoding
# ----------------------------------------------------------------------
def test_encode_bit_flip_circuit_matches_repetition_states():
    s0 = encode_bit_flip_circuit(0)
    s1 = encode_bit_flip_circuit(1)
    expected0 = np.zeros(8, dtype=complex)
    expected0[0] = 1.0  # |000>
    expected1 = np.zeros(8, dtype=complex)
    expected1[0b111] = 1.0  # |111>
    assert np.allclose(s0, expected0)
    assert np.allclose(s1, expected1)


@pytest.mark.parametrize("name,build_fn", ALL_CODE_BUILDERS.items())
def test_encode_general_passes_verification(name, build_fn):
    code = build_fn()
    for logical_bit in (0, 1):
        state = encode_general(code, logical_bit)
        # Normalized.
        assert abs(np.linalg.norm(state) - 1.0) < 1e-8
        violations = verify_encoded_state(code, state, code.n)
        for label, viol in violations.items():
            assert viol < 1e-8, (
                f"{code.name} logical {logical_bit}: stabilizer {label} violated "
                f"by {viol}"
            )


def test_encode_general_logical_states_are_orthogonal():
    for name, build_fn in ALL_CODE_BUILDERS.items():
        code = build_fn()
        z = encode_general(code, 0)
        o = encode_general(code, 1)
        overlap = abs(np.vdot(z, o))
        assert overlap < 1e-8, f"{code.name}: |0_L> and |1_L> not orthogonal"


def test_encode_bit_flip_circuit_passes_verification():
    code = bit_flip_code()
    for logical_bit, state in ((0, encode_bit_flip_circuit(0)),
                                (1, encode_bit_flip_circuit(1))):
        violations = verify_encoded_state(code, state, code.n)
        for label, viol in violations.items():
            assert viol < 1e-8


# ----------------------------------------------------------------------
# Monte Carlo sanity check
# ----------------------------------------------------------------------
def test_steane_monte_carlo_below_physical_rate():
    code = steane_code()
    rng = np.random.default_rng(42)
    p = 0.01
    result = simulate_logical_error_rate(
        code, p, num_trials=5000, error_model="depolarizing", rng=rng
    )
    assert result["logical_error_rate"] < 0.3 * p, (
        f"Steane logical error rate {result['logical_error_rate']} not far enough "
        f"below physical rate {p}"
    )
