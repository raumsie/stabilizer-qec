"""Monte Carlo logical-error-rate simulation for stabilizer codes.

For a fixed physical error rate p: sample a per-qubit Pauli error,
measure the syndrome, look up the decoder's correction,
and check whether applying the correction leaves a residual
logical operator (logical error). The fraction of trials with a
logical error estimates the code's logical error rate at that p.
"""

import csv
import os
from typing import Optional

import numpy as np

from . import pauli


def random_pauli_error(n: int, p: float, error_model: str = "depolarizing",
                        rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Sample an n-qubit Pauli error with iid per-qubit noise.

    depolarizing: I w.p. 1-p, else X/Y/Z each w.p. p/3.
    bit_flip: X w.p. p, else I. phase_flip: Z w.p. p, else I.
    """
    if rng is None:
        rng = np.random.default_rng()

    vec = np.zeros(2 * n, dtype=np.uint8)

    if error_model == "depolarizing":
        # Per-qubit draw in [0,1): [0,p/3)->X, [p/3,2p/3)->Y, [2p/3,p)->Z, [p,1)->I.
        draws = rng.random(n)
        is_x = draws < (p / 3)
        is_y = (draws >= (p / 3)) & (draws < (2 * p / 3))
        is_z = (draws >= (2 * p / 3)) & (draws < p)
        vec[:n] = (is_x | is_y).astype(np.uint8)  # X part: X or Y
        vec[n:] = (is_z | is_y).astype(np.uint8)  # Z part: Z or Y
    elif error_model == "bit_flip":
        flips = rng.random(n) < p
        vec[:n] = flips.astype(np.uint8)
    elif error_model == "phase_flip":
        flips = rng.random(n) < p
        vec[n:] = flips.astype(np.uint8)
    else:
        raise ValueError(f"Unknown error_model: {error_model!r}")

    return vec


def simulate_logical_error_rate(code, p: float, num_trials: int,
                                 error_model: str = "depolarizing",
                                 rng: Optional[np.random.Generator] = None) -> dict:
    """Run `num_trials` iid Monte Carlo trials at physical error rate `p`
    and return summary statistics of the resulting logical error rate."""
    if rng is None:
        rng = np.random.default_rng()

    n = code.n
    num_logical_errors = 0
    for _ in range(num_trials):
        error = random_pauli_error(n, p, error_model=error_model, rng=rng)
        synd = code.syndrome(error)
        correction = code.decode(synd)
        if code.is_logical_error(error, correction):
            num_logical_errors += 1

    p_hat = num_logical_errors / num_trials
    std_error = float(np.sqrt(p_hat * (1 - p_hat) / num_trials))

    return {
        "p": p,
        "logical_error_rate": p_hat,
        "num_trials": num_trials,
        "num_logical_errors": num_logical_errors,
        "std_error": std_error,
    }


def sweep(code, ps: list, num_trials: int, error_model: str = "depolarizing",
          rng: Optional[np.random.Generator] = None) -> list:
    """Run simulate_logical_error_rate for each p in `ps`."""
    if rng is None:
        rng = np.random.default_rng()
    return [
        simulate_logical_error_rate(code, p, num_trials, error_model=error_model, rng=rng)
        for p in ps
    ]


def save_results_csv(results: list, code_name: str, path: str) -> None:
    """Append result rows to a shared CSV, writing a header only if the
    file doesn't already exist."""
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    fieldnames = ["code_name", "p", "logical_error_rate", "num_trials",
                  "num_logical_errors", "std_error"]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in results:
            writer.writerow({
                "code_name": code_name,
                "p": r["p"],
                "logical_error_rate": r["logical_error_rate"],
                "num_trials": r["num_trials"],
                "num_logical_errors": r["num_logical_errors"],
                "std_error": r["std_error"],
            })
