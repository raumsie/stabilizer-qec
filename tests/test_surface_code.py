"""Tests for the Stim + PyMatching rotated surface code pipeline."""

from __future__ import annotations

import stim

from qecsim import surface_code


def test_generate_circuit_valid_for_small_distances():
    for d in (3, 5):
        circuit = surface_code.generate_circuit(distance=d, rounds=d, p=0.005)
        assert isinstance(circuit, stim.Circuit)
        assert len(circuit) > 0
        assert circuit.num_detectors > 0
        assert circuit.num_observables >= 1


def test_no_noise_means_no_logical_errors():
    circuit = surface_code.generate_circuit(distance=3, rounds=3, p=1e-9)
    num_errors, num_shots = surface_code.count_logical_errors(
        circuit, num_shots=50, rng_seed=1
    )
    assert num_shots == 50
    assert num_errors == 0


def test_larger_distance_suppresses_errors_below_threshold():
    p = 0.001
    num_shots = 4000

    rates = {}
    for d in (3, 7):
        circuit = surface_code.generate_circuit(distance=d, rounds=d, p=p)
        num_errors, shots = surface_code.count_logical_errors(
            circuit, num_shots=num_shots, rng_seed=42
        )
        rates[d] = surface_code.logical_error_rate_per_round(num_errors, shots, d)

    # Below threshold, larger distance should give a lower (or at worst
    # comparable, allowing Monte Carlo slack) per-round logical error rate.
    assert rates[7] <= rates[3] + 1e-6


def test_count_logical_errors_runs_for_distance_7():
    circuit = surface_code.generate_circuit(distance=7, rounds=7, p=0.005)
    num_errors, num_shots = surface_code.count_logical_errors(
        circuit, num_shots=200, rng_seed=7
    )
    assert num_shots == 200
    assert 0 <= num_errors <= num_shots
