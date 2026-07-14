"""Visualizations for the rotated surface code threshold study.

Uses a fixed colorblind-safe palette (blue -> aqua -> yellow) plus
distinct marker shapes per series.
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qecsim import surface_code

# Categorical palette, fixed order: blue, aqua, yellow.
_COLORS = ["#2a78d6", "#1baf7a", "#eda100"]
_MARKERS = ["o", "s", "^"]

_GRIDLINE = "#e1e0d9"
_AXIS = "#c3c2b7"
_TEXT_PRIMARY = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_TEXT_MUTED = "#898781"


def plot_threshold(results: list[dict], save_path: str) -> None:
    """Plot the headline threshold figure: one line per code distance,
    physical error rate on a log x-axis, per-round logical error rate on a
    log y-axis. If the curves visibly cross, that marks the approximate
    circuit-level threshold; annotated only when the data shows one cleanly.
    """
    by_distance: dict[int, list[dict]] = defaultdict(list)
    for row in results:
        by_distance[row["distance"]].append(row)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    distances = sorted(by_distance.keys())
    for i, d in enumerate(distances):
        rows = sorted(by_distance[d], key=lambda r: r["p"])
        ps = [r["p"] for r in rows]
        rates = [max(r["logical_error_rate_per_round"], 1e-12) for r in rows]
        errs = [r["std_error"] for r in rows]

        color = _COLORS[i % len(_COLORS)]
        marker = _MARKERS[i % len(_MARKERS)]

        ax.errorbar(
            ps,
            rates,
            yerr=errs,
            color=color,
            marker=marker,
            markersize=7,
            linewidth=2,
            markeredgecolor="white",
            markeredgewidth=0.6,
            capsize=3,
            label=f"d={d}",
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical error rate, $p$", color=_TEXT_PRIMARY, fontsize=11)
    ax.set_ylabel(
        "Logical error rate per round", color=_TEXT_PRIMARY, fontsize=11
    )
    ax.set_title(
        "Rotated Surface Code: Logical vs. Physical Error Rate",
        color=_TEXT_PRIMARY,
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    ax.grid(True, which="major", color=_GRIDLINE, linewidth=0.8, zorder=0)
    ax.grid(True, which="minor", color=_GRIDLINE, linewidth=0.4, alpha=0.6, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_AXIS)
    ax.tick_params(colors=_TEXT_MUTED, labelsize=9)

    legend = ax.legend(
        frameon=False, loc="lower right", fontsize=10, labelcolor=_TEXT_PRIMARY
    )

    # Annotate a crossing only if the smallest/largest-distance curves
    # visibly swap order somewhere in the sampled range.
    crossing_p = _find_crossing(by_distance, distances)
    if crossing_p is not None:
        ax.axvline(crossing_p, color=_TEXT_MUTED, linewidth=1, linestyle="--", zorder=1)
        ax.text(
            crossing_p,
            ax.get_ylim()[1],
            "  apparent threshold",
            color=_TEXT_SECONDARY,
            fontsize=8.5,
            rotation=90,
            va="top",
            ha="left",
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _find_crossing(by_distance: dict[int, list[dict]], distances: list[int]):
    """Approximate p where the smallest- and largest-distance curves cross
    (sign change of rate[d_max] - rate[d_min]), or None if no clean crossing."""
    if len(distances) < 2:
        return None
    d_min, d_max = distances[0], distances[-1]

    rows_min = {r["p"]: r["logical_error_rate_per_round"] for r in by_distance[d_min]}
    rows_max = {r["p"]: r["logical_error_rate_per_round"] for r in by_distance[d_max]}

    common_ps = sorted(set(rows_min) & set(rows_max))
    if len(common_ps) < 2:
        return None

    diffs = [rows_max[p] - rows_min[p] for p in common_ps]
    for i in range(len(diffs) - 1):
        if diffs[i] == 0:
            return common_ps[i]
        if (diffs[i] < 0) != (diffs[i + 1] < 0):
            # Interpolate in log-p space for a better estimate.
            p0, p1 = common_ps[i], common_ps[i + 1]
            f0, f1 = diffs[i], diffs[i + 1]
            frac = abs(f0) / (abs(f0) + abs(f1))
            import math

            log_p = math.log(p0) + frac * (math.log(p1) - math.log(p0))
            return math.exp(log_p)
    return None


def plot_lattice(distance: int, save_path: str) -> None:
    """Plot the rotated surface code qubit layout for a given distance.

    Stim has no matplotlib diagram output, so this builds the layout
    directly from qubit coordinates. Ancilla qubits are identified as
    those measured more than once across the (3-round) circuit; data
    qubits are measured only once, at the end.
    """
    circuit = surface_code.generate_circuit(distance=distance, rounds=3, p=0.001)

    coords = circuit.get_final_qubit_coordinates()
    if not coords:
        raise RuntimeError("No qubit coordinates available for this circuit.")

    measured_counts: dict[int, int] = defaultdict(int)
    for instruction in circuit.flattened():
        if instruction.name in ("M", "MR", "MX", "MY", "MZ"):
            for target in instruction.targets_copy():
                if target.is_qubit_target:
                    measured_counts[target.value] += 1

    ancilla_qubits = {q for q, count in measured_counts.items() if count > 1}
    data_qubits = set(coords.keys()) - ancilla_qubits

    fig, ax = plt.subplots(figsize=(6, 6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    if data_qubits:
        xs = [coords[q][0] for q in data_qubits]
        ys = [coords[q][1] for q in data_qubits]
        ax.scatter(
            xs, ys, s=140, color=_COLORS[0], marker="o", label="data qubit", zorder=3
        )
    if ancilla_qubits:
        xs = [coords[q][0] for q in ancilla_qubits]
        ys = [coords[q][1] for q in ancilla_qubits]
        ax.scatter(
            xs,
            ys,
            s=110,
            color=_COLORS[2],
            marker="s",
            label="ancilla qubit",
            zorder=3,
        )

    ax.set_title(
        f"Rotated Surface Code Layout (d={distance})",
        color=_TEXT_PRIMARY,
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("x", color=_TEXT_PRIMARY)
    ax.set_ylabel("y", color=_TEXT_PRIMARY)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="upper right", labelcolor=_TEXT_PRIMARY)
    ax.grid(True, color=_GRIDLINE, linewidth=0.6, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(_AXIS)
    ax.tick_params(colors=_TEXT_MUTED)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
