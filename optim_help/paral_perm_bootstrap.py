from __future__ import annotations
import pandas as pd

from typing import List, Dict
from joblib import Parallel, delayed
import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:
    # Fallback: if tqdm is not installed, tqdm() is a no-op passthrough
    def tqdm(x, total=None, desc=None):
        return x

try:
    from tqdm_joblib import tqdm_joblib
except Exception:
    from contextlib import contextmanager


    # Fallback: if tqdm_joblib is not installed, the context manager does nothing
    @contextmanager
    def tqdm_joblib(*args, **kwargs):
        yield

# Swap this import depending on your data type:
#   - discrete variables     → use O_Information
#   - continuous variables   → use O_Info_Gaussian (assumes Gaussian distribution)
# from . import O_Information as o_info
from . import O_Info_Gaussian as o_info


# =====================================
# SECTION 1 — Core permutation helper
# =====================================

def _permute_once_and_compute_O_small(X: np.ndarray, rng: np.random.Generator) -> float:
    """
    Fast single-permutation path for small multiplets (m ≤ 9, n ~ 1 000).

    Strategy
    --------
    Instead of permuting each column with a Python loop, we build a full
    (n × m) index matrix in one stack call and gather all columns at once
    with `np.take_along_axis`. This keeps the hot path in NumPy C code and
    avoids repeated Python overhead.

    Parameters
    ----------
    X   : C-contiguous 2-D array of shape (n_samples, n_variables).
    rng : NumPy Generator (passed in so the caller controls the seed).

    Returns
    -------
    float
        O-information of the column-wise permuted matrix.
    """
    n, m = X.shape

    # Build an (n × m) permutation index: each column is an independent shuffle
    # of [0, n-1], so every variable is permuted independently (breaks all
    # statistical dependencies while keeping each marginal intact).
    idx = np.stack([rng.permutation(n) for _ in range(m)], axis=1)  # dtype intp

    # Gather shuffled values: X_perm[i, j] = X[idx[i, j], j]
    X_perm = np.take_along_axis(X, idx, axis=0)

    return o_info.o_information(X_perm)


# ============================================================
# SECTION 2 — Serial bootstrap for a single multiplet
# ============================================================

def bootstrap_significance_serial(
        X,
        n_boot: int = 1000,
        random_state: int | None = None,
        finite_correction: bool = True,
):
    """
    Column-wise permutation bootstrap to assess the significance of O-information.
    The null hypothesis is that all variables are mutually independent
    (no higher-order interaction). Under H0, permuting columns independently
    destroys all dependencies while preserving marginals.

    Test statistic
    --------------
    Two-sided, bias-corrected pivot:
        t = |Ω_obs  - mean(Ω_null)|
    p-value = P(|Ω_perm - mean(Ω_null)| ≥ t)

    Centering on the null mean rather than zero corrects for the small but
    systematic bias that permutation introduces in finite samples.

    Finite-sample correction (Phipson & Smyth, 2010)
    -------------------------------------------------
    p = (k + 1) / (B + 1)   instead of   k / B
    This avoids p = 0 exactly and keeps the estimator unbiased.

    Parameters
    ----------
    X               : array-like, shape (n_samples, n_variables)
    n_boot          : number of permutation replicates (default 1 000)
    random_state    : integer seed or None for non-reproducible runs
    finite_correction : apply (k+1)/(B+1) correction (recommended)

    Returns
    -------
    observed_O  : float  — Ω computed on the original data
    p_value     : float  — empirical two-sided p-value
    bootstrap_O : ndarray of shape (n_boot,) — null distribution
    diagnostics : dict   — summary statistics of the null distribution
    """

    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n_samples, n_variables).")
    if not X.flags["C_CONTIGUOUS"]:
        X = np.ascontiguousarray(X)

    # --- Step 1: compute O-information on the real (observed) data ---
    observed_O = o_info.o_information(X)

    # --- Step 2: set up the random number generator ---
    # Using the modern Generator API (not legacy np.random) for better
    # statistical quality and reproducibility across NumPy versions.
    rng = np.random.default_rng(random_state)

    # --- Step 3: build the null distribution via column-wise permutation ---
    bootstrap_O = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        bootstrap_O[b] = _permute_once_and_compute_O_small(X, rng)

    # --- Step 4: compute the p-value ---

    # Helper: apply finite-sample correction if requested
    def _finite(k: int, B: int) -> float:
        add = 1 if finite_correction else 0
        den = (B + 1) if finite_correction else B
        return (k + add) / den

    x = bootstrap_O
    B = x.size

    # Summary statistics of the null distribution (useful for diagnostics /
    # plotting the permutation histogram later)
    mean_null = float(np.mean(x))
    median_null = float(np.median(x))
    std_null = float(np.std(x, ddof=0))
    # Skewness (manual, avoids scipy dependency); 0 if std is degenerate
    skew_null = float(np.mean(((x - mean_null) / std_null) ** 3)) if std_null > 0 else 0.0

    # Two-sided pivot test statistic (distance from the NULL MEAN)
    # not testing on zero - the expected center of null distro due to shift
    t_obs = abs(observed_O - mean_null)
    t_perm = np.abs(x - mean_null)

    # Count permutation replicates at least as extreme as the observed value
    k_abs = int(np.sum(t_perm >= t_obs))
    p_value = _finite(k_abs, B)

    diagnostics = {
        "mean_null": mean_null,
        "median_null": median_null,
        "std_null": std_null,
        "skew_null": skew_null,
        # 95 % confidence interval of the null distribution
        "q025_null": float(np.quantile(x, 0.025)),
        "q975_null": float(np.quantile(x, 0.975)),
        "t_obs": float(t_obs),
    }

    return float(observed_O), float(p_value), bootstrap_O, diagnostics


# ============================================================
# SECTION 3 — Worker function (one multiplet, runs in a subprocess)
# ============================================================

def _process_multiplet(
        X: np.ndarray,
        col_idx: List[int],
        n_boot: int,
        alternative: str,
        random_state: int,
):
    """
    Worker task dispatched by joblib for a single multiplet.
    This function runs SERIALLY inside each worker process — no nested
    parallelism — to avoid oversubscription when many workers are alive.

    Parameters
    ----------
    X         : full dataset (n_samples × n_variables), C-contiguous.
    col_idx   : 1-based column indices for this multiplet.
    n_boot    : number of bootstrap replicates.
    alternative : kept for API consistency; the serial function is two-sided.
    random_state : integer seed forwarded to the serial bootstrap.

    Returns
    -------
    (key, result)
        key    : tuple of original 1-based indices (used as dict key upstream)
        result : output of bootstrap_significance_serial
    """
    # Preserve the original 1-based indices as the result key
    key = tuple(int(x) for x in col_idx)

    # Convert to 0-based for NumPy indexing
    col_idx_0 = [int(x) - 1 for x in col_idx]

    X = np.asarray(X)
    if not X.flags["C_CONTIGUOUS"]:
        X = np.ascontiguousarray(X)

    # Slice only the relevant columns (creates a contiguous copy, not a view,
    # which is safer to pickle and send across processes)
    sub = X[:, col_idx_0]

    result = bootstrap_significance_serial(
        sub,
        n_boot=n_boot,
        random_state=random_state,
    )
    return key, result


# ============================================================
# SECTION 4 — Chunking utility
# ============================================================

def _chunks(lst: list, k: int):
    """
    Yield successive non-overlapping slices of `lst` of size `k`.
    The last chunk may be smaller than `k` if len(lst) is not divisible by k.

    Example
    -------
    list(_chunks([1,2,3,4,5], 2))  →  [[1,2], [3,4], [5]]
    """
    for i in range(0, len(lst), k):
        yield lst[i: i + k]


# ============================================================
# SECTION 5 — Omega computation without bootstrapping
# ============================================================

def compute_omega_for_multiplets(multiplets_dict, X):
    """
    Compute O-information (Ω) for every multiplet in `multiplets_dict`,
    without any significance testing or bootstrapping.

    Use this as a lightweight first pass to rank multiplets before running
    the full permutation test on a subset of interest.

    Parameters
    ----------
    multiplets_dict : dict
        Nested structure:
            {
               order (int): {
                   (node1, node2, ..., nodeK): label (str),
                   ...
               },
               ...
            }
        Node indices are 1-based (matching your variable numbering).
    X : array-like, shape (n_samples, n_variables)
        Dataset with columns indexed starting from 1.

    Returns
    -------
    pd.DataFrame
        One row per multiplet with columns:
            order, node1, node2, ..., nodeK, label, omega
        Sorted by order then node indices for easy inspection.
    """
    X = np.asarray(X)
    rows = []

    for order, mp_dict in multiplets_dict.items():
        for nodes, label in mp_dict.items():
            # Convert 1-based node indices → 0-based column positions
            cols = [n - 1 for n in nodes]

            # Extract the sub-matrix for this multiplet
            subX = X[:, cols]

            # Compute Ω (positive = synergy, negative = redundancy)
            omega_val = o_info.o_information(subX)

            # Flatten node indices into separate columns for readability
            row = {"order": order, "label": label, "omega": omega_val}
            for i, n in enumerate(nodes):
                row[f"node{i + 1}"] = n

            rows.append(row)

    df = pd.DataFrame(rows)
    try:
        node_cols = [c for c in df.columns if c.startswith("node")]
        # Convert node columns to nullable int (keeps NaN but as Int64)
        for col in node_cols:
            df[col] = df[col].astype("Int64")

        df = df.sort_values(["order"] + node_cols).reset_index(drop=True)
        return df
    except Exception:
        print("Warning: could not sort DataFrame — returning unsorted.")
        return df


def _process_filter_chunk(nodes_chunk, X, threshold):
    """Return a list of booleans indicating which multiplets to keep."""
    keep = []

    for nodes in nodes_chunk:
        omega = o_info.o_information(X[:, nodes - 1])  # 1-based -> 0-based
        keep.append(abs(omega) >= threshold)

    return keep


def chunked_filter_multiplets_by_omega(
    multiplets_dict,
    X,
    threshold,
    n_jobs=-1,
):
    """
    Parallel filtering of multiplets by |O-information|.
    Each worker processes approximately N/n_jobs multiplets.
    """

    X = np.asarray(X)

    if n_jobs == -1:
        import os
        n_jobs = os.cpu_count()

    filtered = {}

    for order, df in multiplets_dict.items():

        node_cols = [f"node{i+1}" for i in range(order)]
        nodes = df[node_cols].to_numpy(dtype=int)

        # Split rows into approximately equal chunks
        chunks = np.array_split(nodes, n_jobs)

        keep_chunks = Parallel(n_jobs=n_jobs)(
            delayed(_process_filter_chunk)(chunk, X, threshold)
            for chunk in chunks
        )

        # Flatten the boolean masks
        keep = np.concatenate(keep_chunks)

        filtered[order] = df.loc[keep].reset_index(drop=True)

    return filtered


import numpy as np


def filter_multiplets_by_omega(
    multiplets_dict,
    X,
    threshold,
):
    """
    Filter multiplets based on the absolute value of O-information and
    return the retained multiplets ordered by increasing O-information.

    Parameters
    ----------
    multiplets_dict : dict
        Dictionary of the form
            {order: DataFrame}
        where each DataFrame contains columns node1, node2, ..., nodeK
        with 1-based node indices.

    X : ndarray
        Data matrix of shape (n_samples, n_variables).

    threshold : float
        Retain only multiplets with abs(omega) >= threshold.

    Returns
    -------
    dict
        Same structure as the input, containing only the retained
        multiplets, ordered by ascending omega.
    """

    X = np.asarray(X)
    filtered = {}
    filtered_with_omega = {}

    for order, df in multiplets_dict.items():
        node_cols = [f"node{i + 1}" for i in range(order)]
        omegas = []

        for _, row in df.iterrows():
            nodes = row[node_cols].to_numpy(dtype=int)

            omega = o_info.o_information(
                X[:, nodes - 1]  # 1-based -> 0-based
            )
            omegas.append(omega)

        tmp = df.copy()
        tmp["omega"] = omegas

        tmp = (
            tmp[tmp["omega"].abs() >= threshold]
            .sort_values("omega", key=np.abs, ascending=False)
            .reset_index(drop=True)
        )

        filtered_with_omega[order] = tmp
        filtered[order] = tmp.drop(columns="omega")

    return filtered, filtered_with_omega


# ============================================================
# SECTION 6 — Main entry point: parallel permutation bootstrap
# ============================================================

def perm_bootstrap_parallel(
        multiplets_dict: Dict[int, pd.DataFrame],
        X: pd.DataFrame,
        n_boot: int = 500,
        alternative: str = "auto",
        random_state: int = 42,
        n_jobs: int = -1,
        backend: str = "loky",
        batch_size: str | int = "auto",
        verbose: int = 0,
        progress_pct: float = 0.25,  # update the progress bar every 25% of multiplets
):
    """
    Run the permutation bootstrap in parallel across all multiplets.

    Architecture
    ------------
    • Outer loop  : iterates over "families" (groups of multiplets sharing
                    the same order / size), each with its own joblib pool.
    • Inner loop  : dispatches multiplets in chunks; one chunk = one progress
                    bar update. Chunk size is derived from `progress_pct` so
                    the bar advances at 0%, 25%, 50%, 75%, 100% (or 20% steps
                    if you pass progress_pct=0.20).
    • Each worker : runs `_process_multiplet`, which calls
                    `bootstrap_significance_serial` — fully serial inside,
                    so there is no nested parallelism.

    Parameters
    ----------
    multiplets_dict : dict structured as {order (int): DataFrame}
        Each row of the DF is a multiplet; each value is a 1-based column index into X.
    X : array-like, shape (n_samples, n_variables)
        The dataset. Converted to a C-contiguous float64 array internally.
    n_boot : int - num of permutation replicates per multiplet (default 500).
    alternative : str
        Passed through to the worker; currently unused (test is always
        two-sided), but kept for forward compatibility.
    random_state : int - Master seed. Every worker receives the same seed, so replicates are
        reproducible but independent across variables within each permutation.
    n_jobs : int - num of parallel workers. -1 = use all available CPUs.
    backend : str - joblib backend: "loky" (default, safe), "threading", or "multiprocessing".
    batch_size : int or "auto"
        num of tasks sent to each worker per dispatch. "auto" lets joblib decide (recommended).
    verbose : int - joblib verbosity level (0 = silent).
    progress_pct : float
        Fraction of a family's multiplets between consecutive progress-bar
        updates. E.g., 0.25 → updates at 25%, 50%, 75%, 100%.
        Accepted values: any float in (0, 1], but values like 0.20 or 0.25
        are most meaningful. The actual step is rounded up so at least one
        multiplet is always processed per chunk.

    Returns
    -------
    dict
        {family_key: {multiplet_key (tuple): bootstrap_result}}
        where bootstrap_result is the tuple returned by
        bootstrap_significance_serial:
            (observed_O, p_value, bootstrap_O_array, diagnostics_dict)
    """
    # Ensure X is a C-contiguous NumPy array for safe pickling across processes
    X = np.asarray(X)
    if not X.flags["C_CONTIGUOUS"]:
        X = np.ascontiguousarray(X)

    out = {}

    # ------------------------------------------------------------------
    # Outer progress bar: one tick per family (lightweight, always shown)
    # ------------------------------------------------------------------
    for family_key, df in tqdm(
            multiplets_dict.items(),
            total=len(multiplets_dict),
            desc="Families",
            leave=False,
    ):
        # Build a plain list of multiplets: each entry is a list of 1-based
        # column indices extracted from a row of the DataFrame.
        col_index_lists = [[int(x) for x in df.iloc[m]] for m in range(len(df))]
        N = len(col_index_lists)

        # ------------------------------------------------------------------
        # Compute the chunk size from the requested progress percentage.
        # ------------------------------------------------------------------
        step = max(1, int(np.ceil(progress_pct * N)))

        # ------------------------------------------------------------------
        # Inner progress bar: updates only at chunk boundaries.
        # Using a custom bar_format to show count + elapsed without ETA
        # (ETA is unreliable when updates are coarse-grained).
        # ------------------------------------------------------------------
        pbar = tqdm(
            total=N,
            desc=f"  Multiplets | order={family_key}",
            dynamic_ncols=False,
            bar_format="{l_bar}{n_fmt}/{total_fmt} |{bar}| {elapsed}",
            leave=False,
        )

        family_results: dict = {}

        # ------------------------------------------------------------------
        # Reuse a single joblib pool for all chunks in this family.
        # This avoids the overhead of spawning/destroying workers between
        # chunks (process startup cost can dominate for small chunks).
        # ------------------------------------------------------------------
        with Parallel(
                n_jobs=n_jobs,
                backend=backend,
                batch_size=batch_size,
                verbose=verbose,
        ) as parallel:
            for chunk in _chunks(col_index_lists, step):
                # Dispatch one chunk of multiplets to the worker pool.
                # Each call to `delayed(...)` is one independent task.
                results = parallel(
                    delayed(_process_multiplet)(
                        X,
                        col_idx,
                        n_boot=n_boot,
                        alternative=alternative,
                        random_state=random_state,
                    )
                    for col_idx in chunk
                )

                # Merge chunk results into the family dict
                family_results.update(dict(results))

                # Advance the bar by exactly the number of multiplets
                # processed in this chunk (last chunk may be smaller than step)
                pbar.update(len(chunk))

        pbar.close()
        out[family_key] = family_results

    return out


if __name__ == '__main__':
    l = {5: {(1, 3, 4, 6, 31): 'synergy',
             (1, 2, 3, 7, 16): 'synergy',
             (1, 2, 7, 9, 11): 'synergy',
             (1, 2, 3, 9, 16): 'synergy',
             (1, 2, 3, 11, 16): 'synergy',
             (1, 3, 8, 11, 29): 'synergy',
             (1, 2, 3, 11, 49): 'synergy',
             (1, 2, 7, 11, 49): 'synergy',
             (1, 2, 3, 16, 49): 'synergy',
             (1, 2, 3, 19, 20): 'synergy',
             (1, 2, 7, 19, 20): 'synergy',
             (1, 2, 7, 9, 10): 'synergy',
             (1, 2, 7, 10, 16): 'synergy',
             (1, 2, 7, 12, 16): 'synergy',
             (1, 2, 6, 7, 27): 'synergy',
             (1, 2, 9, 10, 16): 'synergy',
             (1, 2, 6, 9, 27): 'synergy',
             (1, 2, 8, 9, 32): 'synergy',
             (1, 2, 6, 12, 27): 'synergy',
             (1, 2, 7, 12, 27): 'synergy',
             (1, 2, 6, 16, 27): 'synergy',
             (1, 2, 7, 16, 27): 'synergy',
             (1, 2, 7, 16, 32): 'synergy',
             (1, 2, 9, 16, 32): 'synergy',
             (1, 2, 10, 16, 49): 'synergy',
             (2, 7, 9, 32, 45): 'synergy',
             (1, 2, 9, 32, 49): 'synergy',
             (1, 3, 6, 10, 39): 'synergy',
             (1, 3, 6, 14, 22): 'synergy',
             (1, 3, 6, 14, 39): 'synergy',
             (1, 3, 4, 22, 39): 'synergy',
             (1, 3, 6, 22, 39): 'synergy'},
         4: {(1, 2, 7, 16): 'synergy',
             (1, 2, 9, 49): 'synergy',
             (1, 2, 19, 20): 'synergy',
             (2, 7, 10, 16): 'synergy',
             (1, 2, 7, 32): 'synergy',
             (2, 7, 8, 32): 'synergy',
             (2, 7, 8, 49): 'synergy',
             (1, 2, 9, 27): 'synergy',
             (2, 6, 9, 27): 'synergy',
             (1, 2, 9, 32): 'synergy',
             (1, 2, 9, 45): 'synergy',
             (2, 7, 9, 45): 'synergy',
             (2, 9, 10, 49): 'synergy',
             (1, 2, 12, 27): 'synergy',
             (2, 6, 12, 27): 'synergy',
             (1, 2, 16, 27): 'synergy',
             (2, 6, 16, 27): 'synergy',
             (1, 2, 16, 32): 'synergy',
             (2, 10, 16, 32): 'synergy',
             (2, 7, 16, 45): 'synergy',
             (2, 10, 16, 49): 'synergy',
             (2, 11, 16, 49): 'synergy',
             (2, 7, 32, 45): 'synergy',
             (2, 10, 32, 45): 'synergy',
             (1, 2, 32, 49): 'synergy',
             (2, 10, 32, 49): 'synergy',
             (1, 3, 6, 14): 'synergy',
             (3, 4, 6, 14): 'synergy',
             (1, 3, 6, 22): 'synergy',
             (3, 4, 6, 22): 'synergy',
             (1, 3, 6, 39): 'synergy',
             (3, 4, 6, 39): 'synergy',
             (1, 3, 14, 22): 'synergy',
             (3, 4, 14, 22): 'synergy',
             (1, 3, 14, 39): 'synergy',
             (3, 4, 14, 39): 'synergy',
             (1, 3, 22, 39): 'synergy',
             (3, 4, 22, 39): 'synergy'},
         3: {(4, 5, 38): 'redundancy'}}
    path = 'C:/Users/utente/Documents/DataScience/TESI_MAGISTRALE/DATA/EDI_DIAG/BN.csv'
    import utilities as ut

    data = ut.preproc_df(path)
    print(compute_omega_for_multiplets(multiplets_dict=l, X=data))
