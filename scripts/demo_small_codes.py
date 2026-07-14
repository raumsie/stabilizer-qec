#!/usr/bin/env python
"""Readable CLI demo of the custom stabilizer toolkit.

For each of the four small codes (bit-flip, phase-flip, Steane, Shor):
  1. Print stabilizers and logical operators in human-readable form.
  2. Encode logical 0 and 1, verify against every stabilizer generator.
  3. Inject an illustrative single-qubit error, show syndrome + decoder
     correction, and confirm no residual logical error.
Then sweep physical error rate for all four codes, save results to
data/small_codes_results.csv, and plot logical vs. physical error rate to
figures/small_codes_logical_vs_physical.png.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qecsim import pauli
from qecsim.codes import bit_flip_code, phase_flip_code, steane_code, shor_code
from qecsim.encoding_circuits import (
    encode_bit_flip_circuit,
    encode_general,
    verify_encoded_state,
)
from qecsim.simulate import sweep, save_results_csv
from qecsim.viz_small_codes import plot_logical_vs_physical

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "data", "small_codes_results.csv")
FIG_PATH = os.path.join(ROOT, "figures", "small_codes_logical_vs_physical.png")

# Each entry: (function that builds the code, illustrative single-qubit
# error label, error model used in the Monte Carlo sweep for this code).
CODE_SPECS = [
    ("Bit-flip repetition code [[3,1,1]]", bit_flip_code, "XII", "bit_flip"),
    ("Phase-flip repetition code [[3,1,1]]", phase_flip_code, "ZII", "phase_flip"),
    ("Steane code [[7,1,3]]", steane_code, "IIIXIII", "depolarizing"),
    ("Shor code [[9,1,3]]", shor_code, "IIIIYIIII", "depolarizing"),
]


def header(text, char="="):
    print()
    print(char * 72)
    print(text)
    print(char * 72)


def section(text):
    print()
    print(f"--- {text} ---")


def demo_code(title, build_fn, illustrative_error_label):
    header(title)
    code = build_fn()

    section("Stabilizer generators")
    for lbl in code.stabilizer_labels:
        print(f"  {lbl}")

    section("Logical operators")
    for i in range(code.k):
        print(f"  X_bar_{i} = {code.logical_x_labels[i]}")
        print(f"  Z_bar_{i} = {code.logical_z_labels[i]}")

    section("Encoding: |0_L> and |1_L>")
    if code.name == "bit_flip_3":
        s0 = encode_bit_flip_circuit(0)
        s1 = encode_bit_flip_circuit(1)
        print("  Using explicit 2-CNOT gate circuit (CNOT(0,1), CNOT(0,2)).")
    else:
        s0 = encode_general(code, 0)
        s1 = encode_general(code, 1)
        print("  Using stabilizer-projector construction (product of (I+G_j)/2).")

    v0 = verify_encoded_state(code, s0, code.n)
    v1 = verify_encoded_state(code, s1, code.n)
    max_v0 = max(v0.values())
    max_v1 = max(v1.values())
    print(f"  |0_L>: max ||S|psi> - |psi>|| over all stabilizers = {max_v0:.2e}")
    print(f"  |1_L>: max ||S|psi> - |psi>|| over all stabilizers = {max_v1:.2e}")
    ok = max_v0 < 1e-8 and max_v1 < 1e-8
    print(f"  Verification: {'PASSED' if ok else 'FAILED'}")
    assert ok, "Encoded state failed stabilizer verification"

    section("Syndrome measurement + decoding demo")
    error = pauli.pauli_from_label(illustrative_error_label)
    synd = code.syndrome(error)
    correction = code.decode(synd)
    residual_is_logical = code.is_logical_error(error, correction)
    print(f"  Injected error:      {pauli.pauli_to_label(error)}")
    print(f"  Measured syndrome:   {synd}")
    print(f"  Decoder correction:  {pauli.pauli_to_label(correction)}")
    print(f"  Residual logical error after correction: {residual_is_logical}")
    assert not residual_is_logical, "Decoder failed to fix the illustrative error"
    print("  Result: error corrected successfully (no logical error).")

    return code


def main():
    codes_and_models = []
    for title, build_fn, err_label, error_model in CODE_SPECS:
        code = demo_code(title, build_fn, err_label)
        codes_and_models.append((code, error_model))

    header("Monte Carlo logical-error-rate sweep")
    ps = [0.001, 0.003, 0.01, 0.03, 0.1, 0.2]
    num_trials = 20000

    # Clear any pre-existing CSV so this run's results are the ones on disk
    # (save_results_csv appends; we want a clean, single-run file here).
    if os.path.exists(CSV_PATH):
        os.remove(CSV_PATH)

    results_by_code = {}
    rng = np.random.default_rng(42)
    t_total_start = time.time()
    for code, error_model in codes_and_models:
        t0 = time.time()
        results = sweep(code, ps, num_trials, error_model=error_model, rng=rng)
        dt = time.time() - t0
        print(f"  {code.name:16s} ({error_model:12s}): {num_trials} trials x "
              f"{len(ps)} p-values in {dt:.1f}s")
        for r in results:
            print(f"      p={r['p']:<7g} logical_error_rate={r['logical_error_rate']:.5f} "
                  f"+/- {r['std_error']:.5f}  ({r['num_logical_errors']}/{r['num_trials']})")
        save_results_csv(results, code.name, CSV_PATH)
        results_by_code[code.name] = results

    print(f"\nTotal sweep time: {time.time() - t_total_start:.1f}s")
    print(f"Results saved to: {CSV_PATH}")

    header("Plotting")
    plot_logical_vs_physical(results_by_code, FIG_PATH)
    print(f"Figure saved to: {FIG_PATH}")

    header("Done")


if __name__ == "__main__":
    main()
