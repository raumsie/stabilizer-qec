"""Rotated surface code memory experiments, simulated with Stim and decoded
with PyMatching's minimum-weight perfect matching (MWPM) decoder. These are
standard QEC tools, used to compare against the custom code.
"""

from __future__ import annotations

import stim
import pymatching


def generate_circuit(distance: int, rounds: int, p: float) -> stim.Circuit:
    """Build a rotated surface code memory experiment under circuit-level noise.

    Uses Stim's built-in `surface_code:rotated_memory_z` generator: encodes a
    logical qubit in the Z basis, runs `rounds` rounds of syndrome
    measurement, then measures out the data qubits to read the final value.

    Noise is circuit-level depolarizing (applied after every gate, reset,
    and measurement, not just data qubits). Circuit-level thresholds in
    this noise regime are commonly cited around ~1% (Fowler et al., "Surface
    codes: Towards practical large-scale quantum computation," Phys. Rev. A
    86, 032324 (2012), page 2, arXiv:1208.0928). The exact value depends on
    the noise model and decoder.

    Args:
        distance: Code distance -- larger corrects more errors, roughly
            (distance - 1) / 2 per round.
        rounds: Number of syndrome-measurement rounds to simulate.
        p: Physical error probability, applied to all noise channels.

    Returns:
        A `stim.Circuit` ready for `detector_error_model()` and sampling.
    """
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )


def count_logical_errors(
    circuit: stim.Circuit, num_shots: int, rng_seed: int | None = None
) -> tuple[int, int]:
    """Sample a noisy circuit and decode it with MWPM, counting logical failures.

    Builds the circuit's detector error model (DEM), turns it into a
    PyMatching graph, samples `num_shots` shots, decodes each via MWPM, and
    counts shots where the decoded observable disagrees with the true one.

    Args:
        circuit: A noisy `stim.Circuit` (from `generate_circuit`).
        num_shots: Number of Monte Carlo shots to sample and decode.
        rng_seed: Optional seed for reproducibility.

    Returns:
        A tuple `(num_logical_errors, num_shots)`.
    """
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)

    sampler = circuit.compile_detector_sampler(seed=rng_seed)
    detection_events, observable_flips = sampler.sample(
        num_shots, separate_observables=True
    )

    predictions = matcher.decode_batch(detection_events)

    num_logical_errors = int((predictions != observable_flips).any(axis=1).sum())
    return num_logical_errors, num_shots


def logical_error_rate_per_round(
    num_logical_errors: int, num_shots: int, rounds: int
) -> float:
    """Convert a raw per-experiment logical error rate into a per-round rate,
    so experiments with different round counts are comparable.

    Treats each round as an independent trial with failure probability
    `p_round`, giving `p_round = 1 - (1-p_L)**(1/rounds)`.

    Args:
        num_logical_errors: Number of shots with a logical error.
        num_shots: Total number of shots.
        rounds: Number of syndrome-measurement rounds in the experiment.

    Returns:
        The estimated per-round logical error rate.
    """
    p_l = num_logical_errors / num_shots
    p_l = min(p_l, 1.0)  # guard float edge cases near 1
    return 1 - (1 - p_l) ** (1 / rounds)
