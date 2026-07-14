"""CLI: run a rotated surface code threshold sweep with Stim + PyMatching,
save results to CSV, and generate the headline threshold plot (plus a
best-effort lattice diagram).
"""

from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from qecsim import surface_code, threshold, viz_surface_code

DATA_PATH = os.path.join(ROOT, "data", "surface_code_threshold.csv")
FIGURE_PATH = os.path.join(ROOT, "figures", "surface_code_threshold.png")
LATTICE_PATH = os.path.join(ROOT, "figures", "surface_code_lattice_d5.png")


def main():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(FIGURE_PATH), exist_ok=True)

    # --- Step 1: fast sanity check -------------------------------------------------
    print("=== Sanity check: confirming the pipeline runs end to end ===")
    sanity_results = threshold.run_threshold_sweep(
        distances=[3, 5], ps=[0.005, 0.01], num_shots=500, seed=0
    )
    for row in sanity_results:
        assert 0 <= row["logical_error_rate"] <= 1
    print("Sanity check passed.\n")

    # --- Step 2: quick timing check to size the real sweep ----------------------
    print("=== Timing check on one (d, p) pair to size num_shots ===")
    t0 = time.time()
    probe_circuit = surface_code.generate_circuit(distance=7, rounds=7, p=0.01)
    surface_code.count_logical_errors(probe_circuit, num_shots=2000, rng_seed=123)
    elapsed = time.time() - t0
    per_shot = elapsed / 2000
    print(f"d=7 probe: {elapsed:.2f}s for 2000 shots ({per_shot * 1000:.3f} ms/shot)")

    distances = [3, 5, 7]
    ps = [0.002, 0.003, 0.005, 0.007, 0.01, 0.013, 0.016, 0.02]
    num_combinations = len(distances) * len(ps)

    num_shots = 10000
    est_total = per_shot * num_shots * num_combinations
    if est_total > 240:  # keep the full sweep to a few minutes
        num_shots = 5000
        est_total = per_shot * num_shots * num_combinations
        if est_total > 240:
            num_shots = 3000
            est_total = per_shot * num_shots * num_combinations
    print(
        f"Using num_shots={num_shots} for the full sweep "
        f"(estimated ~{est_total:.0f}s across {num_combinations} combinations).\n"
    )

    # --- Step 3: full sweep -------------------------------------------------------
    print("=== Full threshold sweep ===")
    t0 = time.time()
    results = threshold.run_threshold_sweep(
        distances=distances, ps=ps, num_shots=num_shots, seed=1
    )
    print(f"Full sweep took {time.time() - t0:.1f}s.\n")

    # --- Step 4: save CSV ----------------------------------------------------------
    threshold.save_results_csv(results, DATA_PATH)
    print(f"Saved results to {DATA_PATH}")

    # --- Step 5: headline threshold plot --------------------------------------------
    viz_surface_code.plot_threshold(results, FIGURE_PATH)
    print(f"Saved threshold plot to {FIGURE_PATH}")

    # --- Step 6: best-effort lattice diagram ----------------------------------------
    try:
        viz_surface_code.plot_lattice(5, LATTICE_PATH)
        print(f"Saved lattice diagram to {LATTICE_PATH}")
    except Exception as e:
        print(f"Skipping lattice diagram: unreliable to generate correctly ({e})")

    # --- Step 7: summary -------------------------------------------------------------
    print("\n=== Summary ===")
    by_dp = {(r["distance"], r["p"]): r for r in results}

    lowest_p = min(ps)
    r3 = by_dp[(3, lowest_p)]
    r7 = by_dp[(7, lowest_p)]
    print(
        f"At lowest p tested (p={lowest_p}): "
        f"d=3 logical_error_rate_per_round={r3['logical_error_rate_per_round']:.5f}, "
        f"d=7 logical_error_rate_per_round={r7['logical_error_rate_per_round']:.5f}"
    )
    if r7["logical_error_rate_per_round"] < r3["logical_error_rate_per_round"]:
        print("-> d=7 outperforms d=3 below threshold, as expected.")
    else:
        print("-> WARNING: d=7 did not outperform d=3 at this p; check shot counts/noise.")

    # crude threshold estimate: find p where d=3 and d=7 curves cross
    crossing = viz_surface_code._find_crossing(
        {d: [r for r in results if r["distance"] == d] for d in distances},
        distances,
    )
    if crossing is not None:
        print(f"Apparent threshold (d=3 vs d=7 crossing): p ~= {crossing:.4f}")
    else:
        print("No clean crossing detected in the sampled p range.")


if __name__ == "__main__":
    main()
