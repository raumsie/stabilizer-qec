"""Threshold sweeps for the rotated surface code: run many (distance, p)
combinations and collect logical-error-rate statistics suitable for the
canonical QEC threshold plot.
"""

from __future__ import annotations

import csv
import math
from typing import Callable, Optional

from qecsim import surface_code


def run_threshold_sweep(
    distances: list[int],
    ps: list[float],
    num_shots: int,
    rounds_fn: Optional[Callable[[int], int]] = None,
    seed: int = 0,
) -> list[dict]:
    """Run a full (distance x physical error rate) threshold sweep: for each
    combination, build a noisy circuit, decode `num_shots` shots via
    `surface_code.count_logical_errors`, and record the logical error
    rate (raw and per-round) with its Monte Carlo uncertainty.

    Args:
        distances: Code distances to sweep over (e.g. `[3, 5, 7]`).
        ps: Physical error rates to sweep over.
        num_shots: Number of shots to sample per (distance, p) combination.
        rounds_fn: Maps a distance to a number of syndrome rounds. Defaults
            to `lambda d: d`, the standard memory-experiment convention.
        seed: Base RNG seed, combined with `(d, p)` indices per combination.

    Returns:
        A list of dicts, one per (distance, p) combination, with keys:
        `distance, p, rounds, num_shots, num_logical_errors,
        logical_error_rate, logical_error_rate_per_round, std_error`.
    """
    if rounds_fn is None:
        rounds_fn = lambda d: d  # noqa: E731

    results = []
    for d in distances:
        rounds = rounds_fn(d)
        for i, p in enumerate(ps):
            shot_seed = seed + 1000 * d + i
            circuit = surface_code.generate_circuit(distance=d, rounds=rounds, p=p)
            num_errors, shots = surface_code.count_logical_errors(
                circuit, num_shots, rng_seed=shot_seed
            )

            p_hat = num_errors / shots
            std_error = math.sqrt(p_hat * (1 - p_hat) / shots)
            per_round = surface_code.logical_error_rate_per_round(
                num_errors, shots, rounds
            )

            print(
                f"d={d} p={p:.4f} -> logical_errors={num_errors}/{shots} "
                f"p_L={p_hat:.5f} p_L/round={per_round:.5f}"
            )

            results.append(
                {
                    "distance": d,
                    "p": p,
                    "rounds": rounds,
                    "num_shots": shots,
                    "num_logical_errors": num_errors,
                    "logical_error_rate": p_hat,
                    "logical_error_rate_per_round": per_round,
                    "std_error": std_error,
                }
            )
    return results


def save_results_csv(results: list[dict], path: str) -> None:
    """Write threshold sweep results to a CSV file with a header row."""
    if not results:
        raise ValueError("No results to save.")

    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
