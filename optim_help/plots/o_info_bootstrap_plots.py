"""
o_info_bootstrap_plots.py
==============
Publication-ready figures for O-information permutation bootstrap results.

Expected input
--------------
The main input is `out`, the direct output of `perm_bootstrap_parallel`:

    out = {
        family_key (int): {
            multiplet_key (tuple of ints + dict): (
                observed_O   : float,
                p_value      : float,
                bootstrap_O  : np.ndarray, shape (n_boot,)
                diagnostics  : dict  {mean_null, median_null, std_null,
                                      skew_null, q025_null, q975_null}
            )
        }
    }

Usage example
-------------
    from plots_oinfo import save_all_figures

    # One call saves all figures + supplementary table to a folder
    save_all_figures(out, output_dir="figures")
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — never opens a display window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

from typing import Dict, Tuple, Any

# ---------------------------------------------------------------------------
# Shared style — applied once at module level so every figure is consistent.
# Minimalist palette suited for print and journal submission (no colours that
# rely on hue alone for meaning, safe for colour-blind readers).
# ---------------------------------------------------------------------------
PALETTE = {
    "synergy":     "#C0392B",   # deep red   — Ω > 0, significant
    "redundancy":  "#2471A3",   # steel blue — Ω < 0, significant
    "nonsig":      "#BDC3C7",   # light grey — not significant
    "null":        "#5D6D7E",   # slate      — permutation null histogram
    "null_ci":     "#D5D8DC",   # pale grey  — 95 % CI band
    "observed":    "#E74C3C",   # vivid red  — observed Ω line
    "zero":        "#2C3E50",   # near-black — Ω = 0 reference line
}

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Liberation Sans"],
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
})


# ============================================================
# UTILITY: flatten results dict → tidy DataFrame
# ============================================================

def results_to_dataframe(results: Dict[int, Dict[tuple, tuple]]) -> pd.DataFrame:
    """
    Convert the nested dict returned by `perm_bootstrap_parallel` into a
    flat, tidy DataFrame — one row per multiplet.

    Parameters
    ----------
    results : dict
        Direct output of perm_bootstrap_parallel.

    Returns
    -------
    pd.DataFrame with columns:
        order        — multiplet size (= family_key)
        multiplet    — tuple of variable indices (= multiplet_key)
        omega        — observed O-information
        p_value      — two-sided permutation p-value
        t_obs        — test statistic |Ω_obs − mean_null|
        mean_null    — mean of the permutation null distribution
        std_null     — SD of the permutation null distribution
        q025_null    — 2.5th percentile of the null
        q975_null    — 97.5th percentile of the null
        significant  — bool, p_value < 0.05
        direction    — "synergy" | "redundancy" | "n.s."
    """
    rows = []
    for order, family in results.items():
        for multiplet_key, result in family.items():
            observed_O, p_value, bootstrap_O, diag = result

            # Test statistic: distance from the null mean (same as in the bootstrap)
            t_obs = abs(observed_O - diag["mean_null"])

            sig = p_value < 0.05
            if sig and observed_O < 0:
                direction = "synergy"
            elif sig and observed_O > 0:
                direction = "redundancy"
            else:
                direction = "n.s."

            rows.append({
                "order":       order,
                "multiplet":   multiplet_key,
                "omega":       observed_O,
                "p_value":     p_value,
                "t_obs":       t_obs,
                "mean_null":   diag["mean_null"],
                "std_null":    diag["std_null"],
                "q025_null":   diag["q025_null"],
                "q975_null":   diag["q975_null"],
                "significant": sig,
                "direction":   direction,
            })

    df = pd.DataFrame(rows).sort_values(["order", "omega"]).reset_index(drop=True)
    return df


# ============================================================
# FIGURE 1 — Null distribution histograms (representative examples)
# ============================================================

def plot_null_distributions(
        results: Dict[int, Dict[tuple, tuple]],
        n_examples: int = 3,
        alpha: float = 0.05,
        figsize: tuple = (5.5, 2.2),
) -> plt.Figure:
    """
    Show the permutation null distribution for a few representative multiplets.

    Selection strategy (automatic):
        • 1 clearly synergistic  (largest positive Ω that is significant)
        • 1 clearly redundant    (largest negative Ω that is significant)
        • 1 non-significant      (p closest to 0.5 — typical null behaviour)

    This is Figure 1 in the paper: it validates the permutation procedure
    without requiring manual selection of examples.

    Parameters
    ----------
    results    : direct output of perm_bootstrap_parallel
    n_examples : how many panels to draw (max 3 recommended for a panel figure)
    alpha      : significance threshold (default 0.05)
    figsize    : (width, height) per panel in inches; total width = n * width

    Returns
    -------
    matplotlib Figure
    """
    df = results_to_dataframe(results)

    # --- Automatic selection of representative multiplets ---
    sig   = df[df["significant"]]
    nonsig = df[~df["significant"]]

    candidates = []

    # Most synergistic significant multiplet: Ω < 0 (most negative)
    if not sig[sig["omega"] < 0].empty:
        row = sig[sig["omega"] < 0].sort_values("omega").iloc[0]
        candidates.append(("Synergy (sig.)", row))

    # Most redundant significant multiplet: Ω > 0 (most positive)
    if not sig[sig["omega"] > 0].empty:
        row = sig[sig["omega"] > 0].sort_values("omega", ascending=False).iloc[0]
        candidates.append(("Redundancy (sig.)", row))

    # Most "typical null" non-significant multiplet (p closest to 0.5)
    if not nonsig.empty:
        row = nonsig.iloc[(nonsig["p_value"] - 0.5).abs().argmin()]
        candidates.append(("Non-significant", row))

    candidates = candidates[:n_examples]
    n_panels = len(candidates)

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(figsize[0] * n_panels, figsize[1]),
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, (title, row) in zip(axes, candidates):
        # Retrieve the full bootstrap array for this multiplet
        order      = row["order"]
        multiplet  = row["multiplet"]
        obs_O, p_val, boot_O, diag = results[order][multiplet]

        mean_null = diag["mean_null"]
        q025      = diag["q025_null"]
        q975      = diag["q975_null"]
        std_null  = diag["std_null"]

        # Histogram of permutation null
        ax.hist(boot_O, bins=35, color=PALETTE["null"], alpha=0.75,
                edgecolor="white", linewidth=0.3, zorder=2)

        # 95 % CI shading (spans the full y range for readability)
        ax.axvspan(q025, q975, alpha=0.25, color=PALETTE["null_ci"],
                   label="95% CI null", zorder=1)

        # Null mean (dashed)
        ax.axvline(mean_null, color=PALETTE["null"], linestyle="--",
                   linewidth=1.2, label=f"Null mean\n({mean_null:.3f})", zorder=3)

        # Observed Ω (solid, coloured by direction)
        # synergy = Ω < 0 (red), redundancy = Ω > 0 (blue)
        line_color = (PALETTE["synergy"] if obs_O < 0 else PALETTE["redundancy"])
        ax.axvline(obs_O, color=line_color, linewidth=2.0,
                   label=f"Observed Ω\n({obs_O:.3f})", zorder=4)

        '''
        SHIFT from ZERO
        "The null distribution is shifted toward negative values (mean ≈ −x) due to 
        marginal imbalance in Likert-scale responses. The dotted line at Ω = 0 shows 
        the expectation for balanced synthetic data. Significance is assessed relative 
        to the empirical null mean, not zero."
        '''
        # Add zero reference line to show the shift from theoretical expectation
        ax.axvline(0, color="black", linewidth=0.8, linestyle=":",
                   label="Zero (synthetic expectation)", zorder=1)

        # p-value annotation in the corner
        p_str = f"p = {p_val:.3f}" if p_val >= 0.001 else f"p < 0.001\n(exact: {p_val:.2e})"
        ax.text(0.97, 0.97, p_str,
                transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))

        ax.set_title(f"{title}\n{multiplet}", fontsize=8, pad=4)
        ax.set_xlabel("Ω (O-information)")
        ax.set_ylabel("Count" if ax is axes[0] else "")
        ax.legend(fontsize=6.5, loc="upper left",
                  frameon=False, handlelength=1.2)

    fig.suptitle(
        f"Figure 1 — Permutation null distributions (B = {len(boot_O)} replicates)",
        fontsize=9, y=1.02,
    )
    return fig


# ============================================================
# FIGURE 2 — Ω scatter plot by multiplet order
# ============================================================

def plot_omega_scatter(
        df: pd.DataFrame,
        alpha: float = 0.05,
        jitter: float = 0.15,
        figsize: tuple = (7, 4),
) -> plt.Figure:
    """
    One dot per multiplet, x = multiplet order, y = observed Ω.
    Colour encodes significance and direction.

    A small random horizontal jitter is added so overlapping dots are visible
    (the jitter is purely cosmetic and does not change order values).

    Parameters
    ----------
    df      : output of results_to_dataframe()
    alpha   : significance threshold (default 0.05)
    jitter  : horizontal spread in data units (default 0.15)
    figsize : figure size in inches

    Returns
    -------
    matplotlib Figure
    """
    rng = np.random.default_rng(0)   # fixed seed → reproducible jitter

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Split into three groups for layered plotting (non-sig drawn first/bottom)
    groups = {
        "n.s.":       (PALETTE["nonsig"],    12, 0.5, "n.s."),
        "redundancy": (PALETTE["redundancy"], 22, 0.85, "Redundancy (sig.)"),
        "synergy":    (PALETTE["synergy"],    22, 0.85, "Synergy (sig.)"),
    }

    for direction, (color, size, alpha_val, label) in groups.items():
        sub = df[df["direction"] == direction]
        if sub.empty:
            continue
        x_jittered = sub["order"] + rng.uniform(-jitter, jitter, len(sub))
        ax.scatter(x_jittered, sub["omega"],
                   c=color, s=size, alpha=alpha_val,
                   edgecolors="none", zorder=3 if direction != "n.s." else 2,
                   label=f"{label} (n={len(sub)})")

    # Reference line at Ω = 0
    ax.axhline(0, color=PALETTE["zero"], linewidth=0.8,
               linestyle="--", label="Zero (synthetic expectation)", zorder=1)

    ax.axhline(df["mean_null"].mean(), color="gray", linewidth=1.0,
               linestyle="--", label=f"Avg null mean ({df['mean_null'].mean():.2f})")

    ax.set_xlabel("Multiplet order (k)")
    ax.set_ylabel("Ω (O-information)")
    ax.set_title("Figure 2 — O-information landscape across multiplet orders")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    legend = ax.legend(
        title=f"α = {alpha}",
        title_fontsize=7,
        frameon=False,
        markerscale=1.3,
    )
    return fig


# ============================================================
# FIGURE 3 — Violin plot of Ω per order
# ============================================================

def plot_omega_violin(
        df: pd.DataFrame,
        figsize: tuple = (7, 4),
) -> plt.Figure:
    """
    Distribution of observed Ω values within each multiplet family (order).

    Each violin = one order. The embedded box shows median ± IQR; whiskers
    extend to 1.5 × IQR. Individual points are overlaid (strip plot style)
    for transparency when n per order is small.

    This plot satisfies the journal requirement for error bars: the caption
    should state "violin = full distribution; inner box: median ± IQR;
    dots = individual multiplets."

    Parameters
    ----------
    df      : output of results_to_dataframe()
    figsize : figure size in inches

    Returns
    -------
    matplotlib Figure
    """
    orders = sorted(df["order"].unique())
    n_orders = len(orders)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    rng = np.random.default_rng(1)

    for i, order in enumerate(orders):
        sub    = df[df["order"] == order]["omega"].values
        n_sub  = len(sub)
        color  = plt.cm.coolwarm(i / max(n_orders - 1, 1))  # one colour per order

        if n_sub >= 4:
            # Violin (kernel density estimate of the full distribution)
            parts = ax.violinplot(sub, positions=[i], widths=0.6,
                                  showmedians=False, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.55)
                pc.set_edgecolor("none")

        # Box overlay: Q1, median, Q3
        q1, med, q3 = np.percentile(sub, [25, 50, 75])
        iqr = q3 - q1
        ax.plot([i, i], [q1, q3], color="black", linewidth=2.5, zorder=4,
                solid_capstyle="round")
        ax.scatter([i], [med], color="white", s=28, zorder=5,
                   edgecolors="black", linewidths=0.8)

        # Individual dots (jittered) — especially important for small families
        x_jitter = rng.uniform(-0.12, 0.12, n_sub)
        ax.scatter(i + x_jitter, sub,
                   s=10, alpha=0.4, color=color, edgecolors="none", zorder=3)

        # Annotate n per violin
        ax.text(i, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else min(sub) - 0.01,
                f"n={n_sub}", ha="center", va="top", fontsize=6.5, color="grey")

    # Reference line
    ax.axhline(0, color=PALETTE["zero"], linewidth=0.8, linestyle="--", zorder=1)

    ax.set_xticks(range(n_orders))
    ax.set_xticklabels([f"order {o}" for o in orders])
    ax.set_ylabel("Ω (O-information)")
    ax.set_title(
        "Figure 3 — Distribution of Ω per multiplet order\n"
        "Box: median ± IQR; dots: individual multiplets"
    )
    return fig


# ============================================================
# FIGURE 4 — Volcano plot (effect size vs. significance)
# ============================================================

def plot_volcano(
        df: pd.DataFrame,
        alpha: float = 0.05,
        figsize: tuple = (6, 5),
        label_top_n: int = 5,
) -> plt.Figure:
    """
    x-axis : observed Ω (effect size)
    y-axis : −log10(p-value)

    This is the standard high-dimensional significance plot. Every multiplet
    is one dot; the horizontal dashed line marks α = 0.05.

    The top `label_top_n` most significant multiplets per direction are
    annotated with their variable indices.

    Parameters
    ----------
    df          : output of results_to_dataframe()
    alpha       : significance threshold for the dashed line (default 0.05)
    figsize     : figure size in inches
    label_top_n : how many extreme multiplets to annotate (default 5)

    Returns
    -------
    matplotlib Figure
    """
    df = df.copy()

    # Clip p-values to avoid log(0); 1e-10 is below any realistic permutation p
    df["neg_log_p"] = -np.log10(df["p_value"].clip(lower=1e-10))

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Map direction → colour
    color_map = {
        "synergy":    PALETTE["synergy"],
        "redundancy": PALETTE["redundancy"],
        "n.s.":       PALETTE["nonsig"],
    }
    colors = df["direction"].map(color_map)

    # Draw non-significant first (bottom layer), then significant on top
    for direction in ["n.s.", "redundancy", "synergy"]:
        sub = df[df["direction"] == direction]
        if sub.empty:
            continue
        ax.scatter(
            sub["omega"], sub["neg_log_p"],
            c=color_map[direction],
            s=18 if direction == "n.s." else 28,
            alpha=0.5 if direction == "n.s." else 0.85,
            edgecolors="none",
            zorder=2 if direction == "n.s." else 3,
            label=f"{direction} (n={len(sub)})",
        )

    # Significance threshold line
    thresh_y = -np.log10(alpha)
    ax.axhline(thresh_y, color="black", linestyle="--",
               linewidth=0.9, label=f"p = {alpha}")

    # Ω = 0 reference
    ax.axvline(0, color=PALETTE["zero"], linewidth=0.6, linestyle=":",  zorder=1)

    # Annotate top significant multiplets (most extreme Ω in each direction)
    sig = df[df["significant"]]
    to_label = pd.concat([
        sig[sig["omega"] < 0].nsmallest(label_top_n, "omega"),   # synergy: most negative
        sig[sig["omega"] > 0].nlargest(label_top_n, "omega"),    # redundancy: most positive
    ])

    for _, row in to_label.iterrows():
        label_text = str(row["multiplet"])
        ax.annotate(
            label_text,
            xy=(row["omega"], row["neg_log_p"]),
            xytext=(4, 3), textcoords="offset points",
            fontsize=5.5, color="black",
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
        )

    ax.set_xlabel("Ω (O-information)")
    ax.set_ylabel("−log₁₀(p-value)")
    ax.set_title(f"Figure 4 — Volcano plot  (α = {alpha})")
    ax.legend(frameon=False, markerscale=1.3)

    return fig


# ============================================================
# SUPPLEMENTARY TABLE — exact values for all multiplets
# ============================================================

def save_supplementary_table(
        df: pd.DataFrame,
        path: str = "supplementary_table.csv",
) -> pd.DataFrame:
    """
    Write the full per-multiplet statistics to a CSV file.

    Columns in the output
    ---------------------
    order, multiplet, omega, p_value, t_obs,
    mean_null, std_null, q025_null, q975_null,
    significant, direction

    This table satisfies the journal requirement for exact p-values and
    test statistics for every multiplet (significant AND non-significant).

    Parameters
    ----------
    df   : output of results_to_dataframe()
    path : output file path

    Returns
    -------
    The same DataFrame (for inspection in a notebook).
    """
    out = df[[
        "order", "multiplet",
        "omega", "p_value", "t_obs",
        "mean_null", "std_null", "q025_null", "q975_null",
        "significant", "direction",
    ]].copy()

    # Round for readability; full precision kept in the returned DataFrame
    for col in ["omega", "t_obs", "mean_null", "std_null", "q025_null", "q975_null"]:
        out[col] = out[col].round(6)
    out["p_value"] = out["p_value"].round(6)

    out.to_csv(path, index=False)
    print(f"Supplementary table saved → {path}  ({len(out)} multiplets)")
    return df

# ============================================================
# MAIN ENTRY POINT — save everything in one call
# ============================================================

def save_all_figures(
        out: Dict[int, Dict[tuple, tuple]],
        output_dir: str = "figures",
        fmt: str = "pdf",
        dpi: int = 300,
        n_examples: int = 3,
        alpha: float = 0.05,
        label_top_n: int = 5,
) -> pd.DataFrame:
    """
    Generate and save all four publication figures plus the supplementary
    CSV table. Figures are never displayed on screen.

    Parameters
    ----------
    out        : direct return value of perm_bootstrap_parallel()
    output_dir : folder where all files are written (created if missing)
    fmt        : image format — "pdf" (vector, best for journals),
                 "svg", or "png"
    dpi        : resolution for raster formats (ignored for pdf/svg)
    n_examples : number of null-distribution panels in Figure 1
    alpha      : significance threshold used in Figures 2 and 4
    label_top_n: number of extreme multiplets annotated in Figure 4

    Returns
    -------
    df : tidy DataFrame (one row per multiplet) — for further inspection
         in a notebook without needing to reload the CSV.

    Files written
    -------------
    <output_dir>/fig1_null_distributions.<fmt>
    <output_dir>/fig2_omega_scatter.<fmt>
    <output_dir>/fig3_omega_violin.<fmt>
    <output_dir>/fig4_volcano.<fmt>
    <output_dir>/supplementary_table.csv
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build the tidy DataFrame once; reused by all plot functions
    df = results_to_dataframe(out)

    # Helper: save and immediately close a figure (frees memory, no display)
    def _save(fig: plt.Figure, name: str) -> None:
        path = os.path.join(output_dir, f"{name}.{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)   # critical: prevents the figure from being shown
        print(f"Saved -> {path}")

    # Figure 1 — null distribution histograms (representative examples)
    fig1 = plot_null_distributions(out, n_examples=n_examples, alpha=alpha)
    _save(fig1, "fig1_null_distributions")

    # Figure 2 — Omega scatter by multiplet order
    fig2 = plot_omega_scatter(df, alpha=alpha)
    _save(fig2, "fig2_omega_scatter")

    # Figure 3 — violin plot per order
    fig3 = plot_omega_violin(df)
    _save(fig3, "fig3_omega_violin")

    # Figure 4 — volcano plot
    fig4 = plot_volcano(df, alpha=alpha, label_top_n=label_top_n)
    _save(fig4, "fig4_volcano")

    # Supplementary table (CSV only, no figure to close)
    save_supplementary_table(df, os.path.join(output_dir, "supplementary_table.csv"))

    return df
