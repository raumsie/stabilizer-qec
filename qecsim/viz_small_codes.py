"""Visualization of logical vs. physical error rate for the small codes.

Colors come from a fixed, colorblind-safe palette (slots assigned by
series identity, not cycled); each series also gets a distinct marker
shape, and the break-even y = x line is a recessive dashed gray reference.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fixed slot per series so the same code always gets the same color.
_SERIES_STYLE = {
    "bit_flip_3": {"color": "#2a78d6", "marker": "o", "label": "Bit-flip [[3,1,1]]"},
    "phase_flip_3": {"color": "#1baf7a", "marker": "s", "label": "Phase-flip [[3,1,1]]"},
    "steane_7_1_3": {"color": "#eda100", "marker": "^", "label": "Steane [[7,1,3]]"},
    "shor_9_1_3": {"color": "#e34948", "marker": "D", "label": "Shor [[9,1,3]]"},
}
_MUTED_INK = "#898781"
_PRIMARY_INK = "#0b0b0b"


def plot_logical_vs_physical(results_by_code: dict, save_path: str) -> None:
    """Log-log plot of logical vs. physical error rate, one line+markers
    per code plus a dashed y = x break-even reference line.

    `results_by_code` maps code name -> list of result dicts as returned
    by simulate.sweep.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)

    all_ps = []
    any_zero_count = False
    for code_name, results in results_by_code.items():
        results = sorted(results, key=lambda r: r["p"])
        ps = [r["p"] for r in results]
        # Log scale can't show a rate of exactly 0 (no logical errors observed).
        # Floor such points at 1/(2*num_trials) and mark them hollow as an
        # upper-bound placeholder rather than a direct measurement.
        rates = []
        is_zero_count = []
        for r in results:
            if r["num_logical_errors"] == 0:
                rates.append(1.0 / (2 * r["num_trials"]))
                is_zero_count.append(True)
                any_zero_count = True
            else:
                rates.append(r["logical_error_rate"])
                is_zero_count.append(False)
        errs = [r["std_error"] for r in results]
        all_ps.extend(ps)

        style = _SERIES_STYLE.get(code_name, {"color": "#4a3aa7", "marker": "x",
                                                "label": code_name})
        ax.errorbar(
            ps, rates, yerr=errs,
            color=style["color"], marker=style["marker"], markersize=7,
            linewidth=2, linestyle="-", capsize=3,
            label=style["label"], zorder=3,
        )
        # Re-draw zero-count points hollow on top as "no events observed" markers.
        zero_ps = [p for p, z in zip(ps, is_zero_count) if z]
        zero_rates = [rt for rt, z in zip(rates, is_zero_count) if z]
        if zero_ps:
            ax.scatter(zero_ps, zero_rates, marker=style["marker"], s=55,
                       facecolors="white", edgecolors=style["color"],
                       linewidths=1.8, zorder=4)

    # Break-even reference: physical rate == logical rate (code gives no benefit).
    if all_ps:
        lo, hi = min(all_ps), max(all_ps)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5,
                 color=_MUTED_INK, label="Break-even (logical = physical)", zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical error rate p", color=_PRIMARY_INK, fontsize=11)
    ax.set_ylabel("Logical error rate", color=_PRIMARY_INK, fontsize=11)
    ax.set_title("Logical vs. physical error rate: small stabilizer codes",
                 color=_PRIMARY_INK, fontsize=13, fontweight="bold")

    ax.grid(True, which="both", linestyle="-", linewidth=0.5, color="#e1e0d9",
             alpha=0.7)
    ax.tick_params(colors=_MUTED_INK)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")

    ax.legend(frameon=False, fontsize=9, loc="upper left")

    if any_zero_count:
        fig.text(
            0.5, -0.02,
            "Hollow markers: zero logical errors observed in num_trials trials; "
            "plotted at 1/(2 x num_trials) as an upper-bound placeholder.",
            ha="center", fontsize=8, color=_MUTED_INK,
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
