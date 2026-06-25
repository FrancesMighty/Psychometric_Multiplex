"""
BCa_plots.py
============
Publication-ready figures for BCa bootstrap O-Information results.

Covers the journal statistics requirements:
  - Exact n values reported in titles / captions
  - Error bars explicitly defined (95 % BCa CI)
  - Significant and non-significant results both shown
  - Redundancy / synergy breakdown per family
  - CI method reported (BCa or percentile, adaptive)

Figures produced
----------------
Figure 1 – Ω distribution with BCa CIs (dot + whisker, per family, one panel each)
Figure 2 – Redundancy / synergy / non-significant counts per family (stacked bar)
Figure 3 – Bootstrap sampling distribution for a representative multiplet (histogram)
Figure 4 – Summary table: exact n, % significant, median |Ω|

Usage
-----
    from BCa_plots import plot_all
    plot_all(BCa_boot, BCa_selected, n_samples=200, output_dir="figures/")
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Global style – clean, journal-ready (no gridlines on data, minimal chrome)
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # embed fonts for Illustrator compatibility
    "ps.fonttype": 42,
})

# Palette – two accent colours (redundancy / synergy) + neutral for non-sig
_RED_COLOR = "#C0392B"  # redundancy
_SYN_COLOR = "#2471A3"  # synergy
_NS_COLOR = "#BDC3C7"  # non-significant
_OBS_COLOR = "#2C3E50"  # observed point estimate


# ===========================================================================
# Internal helpers
# ===========================================================================

def _unpack_family(
        BCa_boot: Dict[tuple, tuple],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unpack a single family's BCa_boot dict into aligned arrays.

    Parameters
    ----------
    BCa_boot : {multiplet: (observed_O, lo, hi, diag)}

    Returns
    -------
    observed : (M,) observed Ω values
    lo       : (M,) lower CI bounds
    hi       : (M,) upper CI bounds
    """
    observed, lo, hi = [], [], []
    for obs, l, h, *_ in BCa_boot.values():
        observed.append(obs)
        lo.append(l)
        hi.append(h)
    return np.array(observed), np.array(lo), np.array(hi)


def _classify(
        observed: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
) -> np.ndarray:
    """
    Classify each multiplet as 'redundancy', 'synergy', or 'ns'.

    A multiplet is significant when its CI does not overlap zero.

    Parameters
    ----------
    observed : (M,) observed Ω
    lo, hi   : (M,) BCa CI bounds

    Returns
    -------
    labels : (M,) array of strings
    """
    labels = np.full(len(observed), "ns", dtype=object)
    labels[(lo > 0)] = "redundancy"
    labels[(hi < 0)] = "synergy"
    return labels


def _sort_by_observed(
        observed: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return arrays sorted by observed Ω (ascending)."""
    order = np.argsort(observed)
    return observed[order], lo[order], hi[order], labels[order]


def _color_array(labels: np.ndarray) -> list:
    """Map label strings to hex colours."""
    cmap = {"redundancy": _RED_COLOR, "synergy": _SYN_COLOR, "ns": _NS_COLOR}
    return [cmap[l] for l in labels]


def _save(fig: plt.Figure, path: str) -> None:
    """Save figure as both PDF and PNG."""
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png")
    plt.close(fig)
    print(f"  saved → {path}.pdf / .png")


# ===========================================================================
# Figure 1 – Ω dot-and-whisker plot per family
# ===========================================================================

def plot_omega_ci(
        BCa_boot_all: Dict[int, Dict[tuple, tuple]],
        n_samples: int,
        output_dir: str = ".",
        alpha: float = 0.05,
        max_multiplets_shown: int = 300,
) -> None:
    """
    Dot-and-whisker plot of observed Ω ± 95 % BCa CI for each family.

    One subplot per family.  Multiplets are sorted by observed Ω.
    Significant multiplets (CI does not overlap zero) are coloured;
    non-significant ones are grey.  A horizontal dashed line marks Ω = 0.

    The panel title reports the exact n (multiplets) and n_samples used,
    as required by journal statistics guidelines.

    Parameters
    ----------
    BCa_boot_all        : full nested BCa dict {family: {multiplet: (obs, lo, hi, diag)}}
    n_samples           : number of observations in X (rows); reported in caption
    output_dir          : directory for saved figures
    alpha               : CI level used (reported in axis label)
    max_multiplets_shown: subsample for visibility when family is very large
    """
    families = sorted(BCa_boot_all.keys())
    n_fam = len(families)

    fig, axes = plt.subplots(
        1, n_fam,
        figsize=(4.5 * n_fam, 4.0),
        sharey=False,
    )
    if n_fam == 1:
        axes = [axes]

    ci_pct = int(100 * (1 - alpha))

    for ax, fam in zip(axes, families):
        observed, lo, hi = _unpack_family(BCa_boot_all[fam])
        labels = _classify(observed, lo, hi)
        observed, lo, hi, labels = _sort_by_observed(observed, lo, hi, labels)

        M = len(observed)

        # Sub-sample for readability if family is very large.
        if M > max_multiplets_shown:
            step = M // max_multiplets_shown
            shown = np.arange(0, M, step)
            observed_s, lo_s, hi_s, labels_s = (
                observed[shown], lo[shown], hi[shown], labels[shown]
            )
            note = f"(showing {len(shown)}/{M})"
        else:
            observed_s, lo_s, hi_s, labels_s = observed, lo, hi, labels
            note = ""

        x = np.arange(len(observed_s))
        colors = _color_array(labels_s)

        # Error bars (asymmetric: obs - lo, hi - obs).
        yerr = np.array([
            observed_s - lo_s,
            hi_s - observed_s,
        ])

        ax.errorbar(
            x, observed_s,
            yerr=yerr,
            fmt="none",
            ecolor=colors,
            elinewidth=0.6,
            capsize=0,
            alpha=0.6,
            zorder=1,
        )
        ax.scatter(
            x, observed_s,
            c=colors,
            s=6,
            linewidths=0,
            zorder=2,
        )

        # Zero reference line.
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=0)

        # Count significant multiplets for the title.
        n_red = int(np.sum(labels == "redundancy"))
        n_syn = int(np.sum(labels == "synergy"))
        n_ns = int(np.sum(labels == "ns"))

        ax.set_title(
            f"Family {fam}  |  n = {M} multiplets, {n_samples} observations {note}\n"
            f"redundancy = {n_red}, synergy = {n_syn}, n.s. = {n_ns}",
            fontsize=7, pad=4,
        )
        ax.set_xlabel("Multiplet rank (sorted by Ω)", fontsize=8)
        ax.set_ylabel(f"Ω  ({ci_pct} % BCa CI)", fontsize=8)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

        # Legend moved below all panels; hypothesis spelled out explicitly.
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=_RED_COLOR,
                   markersize=6, label='Redundancy: O-Info > 0, CI entirely above 0'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=_SYN_COLOR,
                   markersize=6, label='Synergy: O-Info < 0, CI entirely below 0'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=_NS_COLOR,
                   markersize=6, label='Non-significant: CI contains 0'),
        ]

    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=3,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.06),
        frameon=False,
    )
    fig.suptitle(
        f'O-Information with {ci_pct} % BCa confidence intervals\n'
        f'Error bars = {ci_pct} % BCa CI (adaptive: percentile when |skew| <= 0.5, '
        f'else BCa; DiCiccio & Efron, 1996)',
        fontsize=8, y=1.02,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    _save(fig, os.path.join(output_dir, 'fig1_omega_ci'))


# ===========================================================================
# Figure 2 – Stacked bar: redundancy / synergy / n.s. per family
# ===========================================================================

def plot_breakdown(
        BCa_boot_all: Dict[int, Dict[tuple, tuple]],
        n_samples: int,
        output_dir: str = ".",
) -> None:
    """
    Stacked horizontal bar chart showing the count and percentage of
    redundancy, synergy, and non-significant multiplets per family.

    Exact n values are annotated inside each bar segment.

    Parameters
    ----------
    BCa_boot_all : full nested BCa dict
    n_samples    : number of observations in X (reported in title)
    output_dir   : directory for saved figures
    """
    families = sorted(BCa_boot_all.keys())
    n_red_list, n_syn_list, n_ns_list, n_tot_list = [], [], [], []

    for fam in families:
        observed, lo, hi = _unpack_family(BCa_boot_all[fam])
        labels = _classify(observed, lo, hi)
        n_red_list.append(int(np.sum(labels == "redundancy")))
        n_syn_list.append(int(np.sum(labels == "synergy")))
        n_ns_list.append(int(np.sum(labels == "ns")))
        n_tot_list.append(len(labels))

    n_fam = len(families)
    y = np.arange(n_fam)
    totals = np.array(n_tot_list, dtype=float)

    # Convert to percentages for bar widths.
    pct_red = np.array(n_red_list) / totals * 100
    pct_syn = np.array(n_syn_list) / totals * 100
    pct_ns = np.array(n_ns_list) / totals * 100

    fig, ax = plt.subplots(figsize=(6.5, 2.2 + 0.5 * n_fam))

    bar_h = 0.5

    b1 = ax.barh(y, pct_red, height=bar_h, color=_RED_COLOR, label="Redundancy")
    b2 = ax.barh(y, pct_syn, height=bar_h, left=pct_red, color=_SYN_COLOR, label="Synergy")
    b3 = ax.barh(y, pct_ns, height=bar_h, left=pct_red + pct_syn,
                 color=_NS_COLOR, label="n.s.")

    # Annotate exact n inside each segment (only if wide enough).
    def _annotate_bar(bars, counts, lefts):
        for bar, n, left in zip(bars, counts, lefts):
            w = bar.get_width()
            if w > 3:  # skip annotation if segment < 3 % wide
                ax.text(
                    left + w / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(n),
                    ha="center", va="center",
                    fontsize=6.5, color="white", fontweight="bold",
                )

    _annotate_bar(b1, n_red_list, np.zeros(n_fam))
    _annotate_bar(b2, n_syn_list, pct_red)
    _annotate_bar(b3, n_ns_list, pct_red + pct_syn)

    # Total n label on the right.
    for i, (tot, pr, ps) in enumerate(zip(n_tot_list, pct_red, pct_syn)):
        ax.text(
            101, y[i],
            f"n = {tot}",
            va="center", fontsize=6.5, color=_OBS_COLOR,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([f"Family {f}" for f in families], fontsize=8)
    ax.set_xlabel("Percentage of multiplets (%)", fontsize=8)
    ax.set_xlim(0, 115)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter())
    ax.set_title(
        f"Multiplet classification per family  |  {n_samples} observations\n"
        "Error bars = 95 % BCa CI; multiplets with CI containing 0 are non-significant (n.s.)",
        fontsize=8,
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color=_RED_COLOR, linewidth=6,
                   label='Redundancy: O-Info > 0 (CI entirely above 0)'),
            Line2D([0], [0], color=_SYN_COLOR, linewidth=6,
                   label='Synergy: O-Info < 0 (CI entirely below 0)'),
            Line2D([0], [0], color=_NS_COLOR, linewidth=6,
                   label='Non-significant: CI contains 0'),
        ],
        loc='lower center',
        ncol=3,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    _save(fig, os.path.join(output_dir, 'fig2_breakdown'))


# ===========================================================================
# Figure 3 – Bootstrap sampling distribution for a representative multiplet
# ===========================================================================

def plot_bootstrap_distribution(
        boot_array: np.ndarray,
        observed_O: float,
        lo: float,
        hi: float,
        rep_label: str,
        multiplet: tuple,
        family: int,
        n_samples: int,
        ci_method: str = "adaptive (BCa or percentile)",
        alpha: float = 0.05,
        output_dir: str = ".",
) -> None:
    """
    Histogram of the bootstrap sampling distribution for a single multiplet,
    with the observed Ω and its BCa CI marked.

    This figure documents the bootstrap procedure for a representative case,
    as required by journal statistics guidelines.

    Parameters
    ----------
    boot_array  : (n_boot,) bootstrap replicates of Ω
    observed_O  : observed Ω on the full sample
    lo, hi      : BCa CI bounds
    param rep_label: type of representative multiplet (most_typical, etc)
    multiplet   : tuple of variable indices (for labelling)
    family      : family/order key
    n_samples   : number of observations in X
    ci_method   : string describing the CI method used (from diagnostics)
    alpha       : significance level
    output_dir  : directory for saved figures
    """
    ci_pct = int(100 * (1 - alpha))
    n_boot = len(boot_array)
    is_sig = not (lo <= 0 <= hi)
    sig_str = "significant" if is_sig else "non-significant"

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    # Histogram of bootstrap replicates.
    ax.hist(
        boot_array,
        bins=50,
        color=_NS_COLOR,
        edgecolor="white",
        linewidth=0.3,
        label=f"Bootstrap replicates (n = {n_boot})",
        zorder=1,
    )

    # Shade the CI region.
    ci_color = _RED_COLOR if observed_O > 0 else _SYN_COLOR
    ax.axvspan(lo, hi, alpha=0.15, color=ci_color, zorder=2,
               label=f"{ci_pct} % BCa CI [{lo:.4f}, {hi:.4f}]")

    # Observed value.
    ax.axvline(observed_O, color=ci_color, linewidth=1.5, linestyle="-",
               zorder=3, label=f"Observed Ω = {observed_O:.4f}")

    # CI bounds.
    ax.axvline(lo, color=ci_color, linewidth=1.0, linestyle=":",
               zorder=3, alpha=0.8)
    ax.axvline(hi, color=ci_color, linewidth=1.0, linestyle=":",
               zorder=3, alpha=0.8)

    # Zero reference.
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--',
               zorder=0, label='O-Info = 0 (no interaction)')

    # Hypothesis label derived from observed sign and CI.
    hypothesis_str = (
        'Redundancy (O-Info > 0, CI entirely above 0)' if observed_O > 0 and lo > 0
        else 'Synergy (O-Info < 0, CI entirely below 0)' if observed_O < 0 and hi < 0
        else 'Non-significant (CI contains 0)'
    )

    ax.set_xlabel('O-Information', fontsize=8)
    ax.set_ylabel('Bootstrap frequency', fontsize=8)
    ax.set_title(
        f'Bootstrap distribution  |  Type {rep_label}, Family {family}, multiplet {multiplet}\n'
        f'n = {n_samples} observations, {n_boot} resamples  |  '
        f'{hypothesis_str}  |  method: {ci_method}',
        fontsize=7, pad=4,
    )
    fig.legend(
        handles=ax.get_legend_handles_labels()[0],
        labels=ax.get_legend_handles_labels()[1],
        loc='lower center',
        ncol=2,
        fontsize=6.5,
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)

    fname = f"fig3_boot_dist_fam{family}_mult{'_'.join(map(str, multiplet))}"
    _save(fig, os.path.join(output_dir, fname))


# ===========================================================================
# Figure 4 – Summary statistics table (text figure)
# ===========================================================================

def plot_summary_table(
        BCa_boot_all: Dict[int, Dict[tuple, tuple]],
        n_samples: int,
        n_boot: int,
        alpha: float = 0.05,
        output_dir: str = ".",
) -> None:
    """
    Render a summary statistics table as a matplotlib figure.

    Columns: Family | n multiplets | n significant | % significant |
             median |Ω| (sig) | median CI width (sig) | CI method

    This provides the exact n values and summary statistics required by
    journal guidelines in a format suitable for supplementary material.

    Parameters
    ----------
    BCa_boot_all : full nested BCa dict
    n_samples    : number of observations in X
    n_boot       : number of bootstrap resamples used
    alpha        : significance level
    output_dir   : directory for saved figures
    """
    ci_pct = int(100 * (1 - alpha))
    families = sorted(BCa_boot_all.keys())
    rows = []

    for fam in families:
        observed, lo, hi = _unpack_family(BCa_boot_all[fam])
        labels = _classify(observed, lo, hi)
        n_tot = len(observed)
        n_sig = int(np.sum(labels != "ns"))
        n_red = int(np.sum(labels == "redundancy"))
        n_syn = int(np.sum(labels == "synergy"))
        pct_sig = 100 * n_sig / n_tot if n_tot > 0 else 0.0

        sig_mask = labels != "ns"
        med_omega = float(np.median(np.abs(observed[sig_mask]))) if n_sig > 0 else float("nan")
        med_ci_width = float(np.median(hi[sig_mask] - lo[sig_mask])) if n_sig > 0 else float("nan")

        rows.append([
            f"Family {fam}",
            str(n_tot),
            str(n_sig),
            f"{pct_sig:.1f} %",
            str(n_red),
            str(n_syn),
            f"{med_omega:.4f}" if not np.isnan(med_omega) else "—",
            f"{med_ci_width:.4f}" if not np.isnan(med_ci_width) else "—",
        ])

    col_labels = [
        "Family",
        "n multiplets",
        "n significant",
        "% significant",
        "n redundancy",
        "n synergy",
        "median |Ω|\n(sig. only)",
        f"median {ci_pct} % CI\nwidth (sig. only)",
    ]

    fig, ax = plt.subplots(figsize=(10, 1.2 + 0.5 * len(rows)))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.6)

    # Style header row.
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor(_OBS_COLOR)
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Alternate row shading.
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            tbl[(i, j)].set_facecolor("#F2F3F4" if i % 2 == 0 else "white")

    ax.set_title(
        f'Summary statistics  |  n = {n_samples} observations, '
        f'{n_boot} bootstrap resamples, {ci_pct} % BCa CI (two-sided, alpha = {alpha})\n'
        'CI method: adaptive (percentile when |skew| <= 0.5, else BCa; '
        'DiCiccio & Efron, 1996; Efron & Tibshirani, 1994)\n'
        'Hypothesis key — Redundancy: O-Info > 0, CI entirely above 0  |  '
        'Synergy: O-Info < 0, CI entirely below 0  |  '
        'Non-significant: CI contains 0',
        fontsize=8, pad=10,
    )
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, 'fig4_summary_table'))


# ===========================================================================
# Convenience wrapper – produce all figures at once
# ===========================================================================

def plot_all(
        BCa_boot_all: Dict[int, Dict[tuple, tuple]],
        n_samples: int,
        n_boot: int,
        alpha: float = 0.05,
        output_dir: str = "figures",
        representative_multiplets: Dict[str, Optional[Tuple[int, tuple, np.ndarray]]] = None,
) -> None:
    """
    Generate all publication figures and save them to *output_dir*.

    Parameters
    ----------
    BCa_boot_all             : {family: {multiplet: (obs, lo, hi, diag)}}
    n_samples                : number of rows in X
    n_boot                   : number of bootstrap resamples used
    alpha                    : significance level (default 0.05 → 95 % CI)
    output_dir               : output directory (created if absent)
    representative_multiplets : dict rep type, optional tuple (family, multiplet_key, boot_array)
                               for Figure 3.  If None, Figure 3 is skipped.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Generating Figure 1 – Ω dot-and-whisker with BCa CIs ...")
    plot_omega_ci(BCa_boot_all, n_samples=n_samples, output_dir=output_dir, alpha=alpha)

    print("Generating Figure 2 – Redundancy / synergy breakdown ...")
    plot_breakdown(BCa_boot_all, n_samples=n_samples, output_dir=output_dir)

    if representative_multiplets is not None:
        for rep_label, representative_multiplet in representative_multiplets.items():
            fam, mult_key, boot_arr = representative_multiplet
            obs, lo, hi, diag = BCa_boot_all[fam][mult_key]
            method_str = diag.get("method", "adaptive")
            print(f"Generating Figure 3 – Bootstrap distribution for multiplet {mult_key} ...")
            plot_bootstrap_distribution(
                boot_array=boot_arr,
                observed_O=obs,
                lo=lo,
                hi=hi,
                rep_label=rep_label,
                multiplet=mult_key,
                family=fam,
                n_samples=n_samples,
                ci_method=method_str,
                alpha=alpha,
                output_dir=output_dir,
            )
    else:
        print("Figure 3 skipped (no representative_multiplet provided).")

    print("Generating Figure 4 – Summary statistics table ...")
    plot_summary_table(BCa_boot_all, n_samples=n_samples, n_boot=n_boot,
                       alpha=alpha, output_dir=output_dir)

    print(f"\nAll figures saved to '{output_dir}/'.")


# ===========================================================================
# Helper – pick representative multiplets automatically
# ===========================================================================

def pick_representative_multiplets(
        BCa_boot_all: Dict[int, Dict[tuple, tuple]],
        boot_arrays_all: Dict[int, Dict[tuple, np.ndarray]],
        family: Optional[int] = None,
) -> Dict[str, Optional[Tuple[int, tuple, np.ndarray]]]:
    """
    Automatically select representative multiplets for Figure 3.

    Three candidates are returned:

    - **most_typical**    : significant multiplet whose |Ω| is closest to
                            the median |Ω| of all significant multiplets.
                            Represents the average significant result.

    - **most_borderline** : significant multiplet whose CI margin (distance
                            of the nearer bound from zero) is smallest.
                            Shows the method at its detection limit —
                            useful for convincing reviewers the threshold
                            is conservative.

    - **most_extreme**    : significant multiplet with the largest |Ω|.
                            Shows the strongest effect clearly.

    Parameters
    ----------
    BCa_boot_all    : {family: {multiplet: (obs, lo, hi, diag)}}
    boot_arrays_all : {family: {multiplet: bootstrap_array}}
                      as returned by ``bootstrap_multiplets_chunked``
    family          : if given, restrict to that family; otherwise the
                      family with the most significant multiplets is used.

    Returns
    -------
    dict with keys ``"most_typical"``, ``"most_borderline"``,
    ``"most_extreme"``.  Each value is a
    ``(family, multiplet_key, boot_array)`` tuple ready to pass to
    ``plot_bootstrap_distribution`` or ``plot_all``, or ``None`` if no
    significant multiplets exist.
    """
    # Choose which family to work on.
    if family is not None:
        fam = family
    else:
        # Automatically pick the family with the most significant multiplets.
        best_fam, best_n = None, -1
        for f, mults in BCa_boot_all.items():
            observed, lo, hi = _unpack_family(mults)
            n_sig = int(np.sum(_classify(observed, lo, hi) != "ns"))
            if n_sig > best_n:
                best_n, best_fam = n_sig, f
        fam = best_fam

    mults_dict = BCa_boot_all[fam]
    boot_dict = boot_arrays_all[fam]

    # Collect only significant multiplets (CI does not contain zero).
    keys, obs_vals, lo_vals, hi_vals = [], [], [], []
    for key, (obs, lo, hi, *_) in mults_dict.items():
        if not (lo <= 0 <= hi):
            keys.append(key)
            obs_vals.append(obs)
            lo_vals.append(lo)
            hi_vals.append(hi)

    if not keys:
        print(f"Warning: no significant multiplets found in family {fam}.")
        return {"most_typical": None, "most_borderline": None, "most_extreme": None}

    obs_arr = np.array(obs_vals)
    lo_arr = np.array(lo_vals)
    hi_arr = np.array(hi_vals)
    abs_obs = np.abs(obs_arr)

    # most_typical: |Ω| closest to the median across significant multiplets.
    idx_typ = int(np.argmin(np.abs(abs_obs - np.median(abs_obs))))

    # most_borderline: smallest margin between the nearer CI bound and zero.
    margins = np.minimum(np.abs(lo_arr), np.abs(hi_arr))
    idx_bord = int(np.argmin(margins))

    # most_extreme: largest |Ω|.
    idx_ext = int(np.argmax(abs_obs))

    def _pack(idx):
        key = keys[idx]
        return fam, key, boot_dict[key]

    result = {
        "most_typical": _pack(idx_typ),
        "most_borderline": _pack(idx_bord),
        "most_extreme": _pack(idx_ext),
    }

    # Print a summary of what was selected.
    print(f"\nRepresentative multiplets selected from family {fam}:")
    for label, (f, k, _) in result.items():
        obs, lo, hi, diag = mults_dict[k]
        margin = min(abs(lo), abs(hi))
        print(
            f"  {label:18s}  multiplet={k}  "
            f"Ω={obs:+.4f}  CI=[{lo:.4f}, {hi:.4f}]  "
            f"margin from 0={margin:.4f}  method={diag.get('method', '?')}"
        )

    return result
