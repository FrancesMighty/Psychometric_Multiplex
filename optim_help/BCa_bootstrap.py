"""
BCa_bootstrap.py
================
Bootstrap confidence-interval utilities for O-Information multiplets.

Public API
----------
get_boot_stats_parallel_   – parallel bootstrap sampling with chunked progress
get_boot_stats_parallel    – simpler parallel bootstrap (tqdm_joblib style)
bca_bootstrap_parallel     – BCa / adaptive CI computation for all multiplets
BCa_CI_mults_selection     – filter multiplets whose CI is statistically significant

Internal helpers (prefixed with _) are not part of the public API.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard-library / third-party imports
# ---------------------------------------------------------------------------
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.special import erf, erfinv

# Optional progress-bar libraries – fall back gracefully if not installed.
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, *, total=None, desc=None, **_):  # type: ignore[misc]
        """No-op tqdm replacement when the package is unavailable."""
        return iterable

try:
    from tqdm_joblib import tqdm_joblib
except ImportError:
    @contextmanager
    def tqdm_joblib(*args, **kwargs):  # type: ignore[misc]
        """No-op context manager when tqdm_joblib is unavailable."""
        yield

# Swap this import depending on your data type:
#   - discrete variables     → use O_Information
#   - continuous variables   → use O_Info_Gaussian (assumes Gaussian distribution)
from . import O_Information as o_info

# from . import O_Info_Gaussian as o_info

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_EPS = 1e-12  # Small value used to avoid log(0) / division by zero


# ===========================================================================
# Section 1 – Numeric helpers
# ===========================================================================

def _phi(z: np.ndarray | float) -> np.ndarray | float:
    """Standard normal CDF evaluated at *z*."""
    return 0.5 * (1.0 + erf(np.asarray(z) / np.sqrt(2.0)))


def _phi_inv(p: np.ndarray | float) -> np.ndarray | float:
    """Inverse of the standard normal CDF (quantile function) at *p*."""
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return np.sqrt(2.0) * erfinv(2.0 * p - 1.0)


def _ensure_2d_ndarray(X) -> np.ndarray:
    """
    Validate and return *X* as a C-contiguous (n_samples, n_variables) float64
    array.  Raises ValueError if the input is not 2-D.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n_samples, n_variables).")
    return np.ascontiguousarray(X) if not X.flags["C_CONTIGUOUS"] else X


def _slice_cols(X_np: np.ndarray, col_idx: Iterable[int]) -> np.ndarray:
    """
    Select columns from *X_np* using a **1-based** index list.

    Parameters
    ----------
    X_np    : (n, d) array
    col_idx : 1-based column indices

    Returns
    -------
    (n, k) sub-array
    """
    idx = np.asarray(col_idx, dtype=int) - 1  # convert 1-based → 0-based
    if (idx < 0).any():
        raise ValueError("Column index must be ≥ 1 (1-based indexing is expected).")
    return X_np[:, idx]


# ===========================================================================
# Section 2 – Jackknife (used to estimate the BCa acceleration constant)
# ===========================================================================

def _jackknife_values(
        X_np: np.ndarray,
        n_jobs_jk: int = 1,
        backend: str = "loky",
        show_progress: bool = False,
) -> np.ndarray:
    """
    Compute leave-one-out jackknife replicates of Ω (O-Information).

    Each replicate θ_i is Ω estimated on all rows *except* row i.

    Parameters
    ----------
    X_np          : (n, d) data array
    n_jobs_jk     : number of parallel workers (1 = serial)
    backend       : joblib backend string
    show_progress : show a tqdm bar when running serially

    Returns
    -------
    thetas : (n,) array of jackknife estimates
    """
    n = X_np.shape[0]

    if n_jobs_jk == 1:
        # Serial path – straightforward loop, optionally wrapped with tqdm.
        thetas = np.empty(n, dtype=float)
        iterator = tqdm(range(n), total=n, desc="Jackknife") if show_progress else range(n)
        for i in iterator:
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            thetas[i] = o_info.o_information(X_np[mask])
        return thetas

    # Parallel path – joblib distributes each leave-one-out computation.
    pbar = tqdm(total=n, desc="Jackknife") if show_progress else None
    with tqdm_joblib(pbar):
        thetas = Parallel(n_jobs=n_jobs_jk, backend=backend)(
            delayed(lambda i: o_info.o_information(X_np[np.arange(n) != i]))(i)
            for i in range(n)
        )
    return np.asarray(thetas, dtype=float)


def _jackknife_values_subsampled(
        X_np: np.ndarray,
        m: int,
        rng: np.random.Generator,
) -> np.ndarray:
    """
    Approximate jackknife using a random subsample of *m* leave-one-out
    replicates.  Falls back to the full jackknife when m ≥ n.

    Parameters
    ----------
    X_np : (n, d) data array
    m    : number of jackknife replicates to compute
    rng  : NumPy random Generator (for reproducibility)

    Returns
    -------
    (m,) array of jackknife estimates
    """
    n = X_np.shape[0]
    if m >= n:
        # No point sub-sampling – use the full jackknife instead.
        return _jackknife_values(X_np, n_jobs_jk=1, backend="loky", show_progress=False)

    idxs = rng.choice(n, size=m, replace=False)
    thetas = np.empty(m, dtype=float)
    for k, i in enumerate(idxs):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        thetas[k] = o_info.o_information(X_np[mask])
    return thetas


# ===========================================================================
# Section 3 – BCa confidence interval
# ===========================================================================

def _bca_interval(
        stat_obs: float,
        boot_stats: np.ndarray,
        jack_stats: np.ndarray,
        alpha: float = 0.05,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Compute a BCa (bias-corrected and accelerated) bootstrap confidence
    interval.

    Parameters
    ----------
    stat_obs   : observed statistic θ̂
    boot_stats : (B,) bootstrap replicates of θ̂
    jack_stats : (n,) jackknife replicates of θ̂ (used for acceleration)
    alpha      : significance level (two-sided; default 0.05 → 95 % CI)

    Returns
    -------
    lo   : lower CI bound
    hi   : upper CI bound
    diag : dict with z0, a, prop, ql, qu for diagnostics
    """
    boot_stats = np.asarray(boot_stats, dtype=float).ravel()
    jack_stats = np.asarray(jack_stats, dtype=float).ravel()

    # --- Bias-correction z0 ---
    # Proportion of bootstrap replicates below the observed statistic.
    prop = np.clip(np.mean(boot_stats < stat_obs), _EPS, 1.0 - _EPS)
    z0 = float(_phi_inv(prop))

    # --- Acceleration constant a (jackknife-based skewness estimate) ---
    theta_bar = float(jack_stats.mean())
    diff = theta_bar - jack_stats
    num = float(np.sum(diff ** 3))
    den = float(6.0 * (np.sum(diff ** 2) ** 1.5))
    a = 0.0 if den == 0.0 else num / den

    def _adjusted_quantile(q: float) -> float:
        """Map nominal quantile *q* to a BCa-adjusted quantile."""
        z = _phi_inv(q)
        denom = 1.0 - a * (z0 + z)
        # Guard against zero denominator.
        za = z0 + (z0 + z) / denom if denom != 0.0 else z0 + (z0 + z)
        return float(np.clip(_phi(za), _EPS, 1.0 - _EPS))

    ql = _adjusted_quantile(alpha / 2.0)
    qu = _adjusted_quantile(1.0 - alpha / 2.0)

    lo = float(np.quantile(boot_stats, ql, method="linear"))
    hi = float(np.quantile(boot_stats, qu, method="linear"))

    diag = {"z0": z0, "a": a, "prop": prop, "ql": ql, "qu": qu}
    return lo, hi, diag


# ===========================================================================
# Section 4 – Alternative CI methods (percentile, basic, BC)
# ===========================================================================

def _percentile_interval(boot_stats: np.ndarray, alpha: float) -> Tuple[float, float]:
    """Standard percentile bootstrap interval [α/2, 1-α/2]."""
    boot = np.asarray(boot_stats, float).ravel()
    lo = float(np.quantile(boot, alpha / 2, method="linear"))
    hi = float(np.quantile(boot, 1.0 - alpha / 2, method="linear"))
    return lo, hi


def _basic_interval(
        stat_obs: float,
        boot_stats: np.ndarray,
        alpha: float,
) -> Tuple[float, float]:
    """
    Basic (reflection) bootstrap interval.
    CI = [2θ̂ − Q(1-α/2), 2θ̂ − Q(α/2)]
    """
    boot = np.asarray(boot_stats, float).ravel()
    lo_p = float(np.quantile(boot, 1.0 - alpha / 2, method="linear"))
    hi_p = float(np.quantile(boot, alpha / 2, method="linear"))
    return float(2 * stat_obs - lo_p), float(2 * stat_obs - hi_p)


def _skew(boot_stats: np.ndarray) -> float:
    """Sample skewness of *boot_stats* (Fisher's moment coefficient)."""
    x = np.asarray(boot_stats, float).ravel()
    mu = x.mean()
    s = x.std(ddof=1)
    if s == 0:
        return 0.0
    return float(((x - mu) ** 3).mean() / (s ** 3))


# ===========================================================================
# Section 5 – Per-multiplet CI selector (BCa / adaptive)
# ===========================================================================

def bca_or_adaptive_ci_for_multiplet(
        X_np: np.ndarray,
        col_idx: Iterable[int],
        omega_obs: float,
        boot_stats: np.ndarray,
        alpha: float = 0.05,
        mode: Literal["adaptive", "bca", "bc", "percentile", "basic"] = "adaptive",
        n_jobs_jk: int = 1,
        backend: str = "loky",
        show_progress: bool = False,
        jk_max: int = 200,
        random_state: int | None = 42,
) -> Tuple[float, float, float, Dict[str, Any]]:
    """
    Select and compute a bootstrap CI for a single multiplet.

    Parameters
    ----------
    X_np        : full (n, d) data array
    col_idx     : 1-based column indices that define the multiplet
    omega_obs   : observed Ω value for the multiplet
    boot_stats  : (B,) bootstrap distribution of Ω
    alpha       : significance level
    mode        : CI method.  ``"adaptive"`` uses percentile when
                  |skewness| ≤ 0.5, otherwise falls back to BCa with a
                  subsampled jackknife.
    n_jobs_jk   : parallel workers for the jackknife (BCa only)
    backend     : joblib backend
    show_progress : show tqdm bar for jackknife
    jk_max      : maximum jackknife sub-sample size (BCa only)
    random_state: seed for the jackknife sub-sampler

    Returns
    -------
    (omega_obs, lo, hi, diagnostics)
    """
    boot_stats = np.asarray(boot_stats, float).ravel()
    diag: Dict[str, Any] = {"method": mode}

    # --- Closed-form CI methods that do not require jackknife ---
    if mode == "percentile":
        lo, hi = _percentile_interval(boot_stats, alpha)
        return float(omega_obs), lo, hi, diag

    if mode == "basic":
        lo, hi = _basic_interval(omega_obs, boot_stats, alpha)
        return float(omega_obs), lo, hi, diag

    if mode == "bc":
        # Bias-corrected without acceleration (a = 0).
        prop = np.clip(np.mean(boot_stats < omega_obs), _EPS, 1.0 - _EPS)
        z0 = _phi_inv(prop)
        ql = _phi(2.0 * z0 + _phi_inv(alpha / 2))
        qu = _phi(2.0 * z0 + _phi_inv(1.0 - alpha / 2))
        lo = float(np.quantile(boot_stats, np.clip(ql, _EPS, 1.0 - _EPS), method="linear"))
        hi = float(np.quantile(boot_stats, np.clip(qu, _EPS, 1.0 - _EPS), method="linear"))
        diag.update({"z0": float(z0), "ql": float(ql), "qu": float(qu)})
        return float(omega_obs), lo, hi, diag

    # --- Adaptive: decide based on bootstrap skewness ---
    if mode == "adaptive":
        sk = _skew(boot_stats)
        diag["skew"] = sk
        if abs(sk) <= 0.5:
            # Distribution is close to symmetric – percentile CI is adequate.
            lo, hi = _percentile_interval(boot_stats, alpha)
            diag["method"] = "percentile"
            return float(omega_obs), lo, hi, diag
        # Distribution is skewed – fall through to full BCa.
        mode = "bca"

    # --- BCa path ---
    X_sub = _slice_cols(X_np, col_idx)
    rng = np.random.default_rng(random_state)

    if n_jobs_jk == 1 and jk_max is not None:
        # Use a sub-sampled jackknife to keep runtime manageable.
        jk_vals = _jackknife_values_subsampled(X_sub, jk_max, rng)
    else:
        jk_vals = _jackknife_values(
            X_sub, n_jobs_jk=n_jobs_jk, backend=backend, show_progress=show_progress
        )

    lo, hi, bca_diag = _bca_interval(omega_obs, boot_stats, jk_vals, alpha=alpha)
    diag.update(bca_diag)
    diag["method"] = "bca"
    return float(omega_obs), lo, hi, diag


# ===========================================================================
# Section 6 – Bootstrap sampling helpers (parallel)
# ===========================================================================

def _bootstrap_stats_for_cols(
        X_np: np.ndarray,
        col_idx: Iterable[int],
        n_boot: int,
        seed: Optional[int],
) -> np.ndarray:
    """
    Draw *n_boot* bootstrap replicates of Ω for the sub-matrix defined by
    *col_idx*.

    Parameters
    ----------
    X_np    : full (n, d) data array
    col_idx : 1-based column indices
    n_boot  : number of bootstrap resamples
    seed    : integer seed for the RNG

    Returns
    -------
    (n_boot,) array of bootstrap Ω values
    """
    rng = np.random.default_rng(seed)
    X_sub = _slice_cols(X_np, col_idx)
    n = X_sub.shape[0]
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n, dtype=np.int64)
        boot[b] = o_info.o_information(X_sub[idx])
    return boot


def _boot_worker(
        X_np: np.ndarray,
        cols: List[int],
        n_boot: int,
        seed: int,
) -> Tuple[Tuple[int, ...], np.ndarray]:
    """
    Top-level picklable worker for joblib.
    Returns (multiplet_key, bootstrap_values).

    Using a named function instead of a lambda avoids pickling issues with
    multiprocessing backends.
    """
    key = tuple(map(int, cols))
    vals = _bootstrap_stats_for_cols(X_np, cols, n_boot=n_boot, seed=seed)
    return key, vals


def _chunks(seq: list, size: int):
    """Yield (start_index, chunk) pairs from *seq* in steps of *size*."""
    for i in range(0, len(seq), size):
        yield i, seq[i: i + size]


# ===========================================================================
# Section 7 – get_boot_stats_parallel_  (chunked, milestone progress)
# ===========================================================================

def get_boot_stats_parallel_chunked(
        multiplets_dict: Dict[int, list],
        X,
        n_boot: int = 500,
        random_state: int = 42,
        n_jobs: int = -1,
        backend: str = "loky",
        batch_size: str | int = "auto",
        verbose: int = 0,
        *,
        progress_step: int = 100,
) -> Dict[int, Dict[tuple, np.ndarray]]:
    """
    Compute bootstrap distributions for every multiplet in every family,
    running jobs in parallel.

    Progress is reported at 0 %, 25 %, 50 %, 75 %, and 100 % of each
    family's multiplets, plus a lightweight ``tqdm`` bar over families.

    Parameters
    ----------
    multiplets_dict : {family_key: [multiplet, ...]}
    X               : (n, d) data array
    n_boot          : bootstrap resamples per multiplet
    random_state    : master seed (spawns independent streams per multiplet)
    n_jobs          : joblib workers (-1 = all CPUs)
    backend         : joblib backend
    batch_size      : joblib batch size
    verbose         : joblib verbosity
    progress_step   : chunk size for inner progress updates

    Returns
    -------
    {family_key: {multiplet_tuple: bootstrap_array}}
    """
    X_np = _ensure_2d_ndarray(X)
    out: Dict[int, Dict[tuple, np.ndarray]] = {}
    ss = np.random.SeedSequence(random_state)

    # Outer bar – one tick per family.
    family_iter = tqdm(
        multiplets_dict.items(),
        total=len(multiplets_dict),
        desc="Families",
        leave=False,
    )

    for family_key, mult_dict in family_iter:
        col_index_lists: List[List[int]] = [list(map(int, m)) for m in mult_dict]
        N = len(col_index_lists)

        # Spawn one independent RNG stream per multiplet for reproducibility.
        child_seeds = ss.spawn(N)
        seeds: List[int] = [int(sseq.generate_state(1)[0]) for sseq in child_seeds]

        # Milestones at 0 %, 25 %, 50 %, 75 %, 100 %.
        milestones = {int(N * pct) for pct in (0.0, 0.25, 0.50, 0.75, 1.0)}

        # Inner tqdm bar – updated once per chunk, not per item.
        pbar = tqdm(
            total=N,
            desc=f"Multiplets | family {family_key}",
            dynamic_ncols=False,
            bar_format="{l_bar}{n_fmt}/{total_fmt} |{bar}| {elapsed}",
            leave=False,
        )

        family_results: Dict[tuple, np.ndarray] = {}

        # Reuse a single joblib pool for the entire family.
        with Parallel(
                n_jobs=n_jobs, backend=backend, batch_size=batch_size, verbose=verbose
        ) as parallel:
            for start, rows in _chunks(col_index_lists, progress_step):
                these_seeds = seeds[start: start + len(rows)]

                results = parallel(
                    delayed(_boot_worker)(
                        X_np=X_np,
                        cols=cols_row,
                        n_boot=n_boot,
                        seed=s,
                    )
                    for cols_row, s in zip(rows, these_seeds)
                )

                for k, v in results:
                    family_results[k] = v

                done = start + len(rows)
                pbar.update(len(rows))

                # Print a milestone line at 25 % increments.
                if done in milestones or done == N:
                    pct = 100 * done // N
                    print(
                        f"  [family {family_key}] bootstrap progress: "
                        f"{done}/{N} multiplets ({pct}%)"
                    )

        pbar.close()
        out[family_key] = family_results

    return out


# ===========================================================================
# Section 8 – get_boot_stats_parallel  (simpler tqdm_joblib variant)
# ===========================================================================

def get_boot_stats_parallel(
        multiplets_dict: Dict[int, list],
        X,
        n_boot: int = 500,
        random_state: int = 42,
        n_jobs: int = -1,
        backend: str = "loky",
        batch_size: str | int = "auto",
        verbose: int = 0,
) -> Dict[int, Dict[tuple, np.ndarray]]:
    """
    Compute bootstrap distributions for every multiplet in every family.

    This variant wraps ``joblib.Parallel`` with ``tqdm_joblib`` for a single
    nested progress bar per family.  It is simpler than
    :func:`get_boot_stats_parallel_` but provides less granular progress.

    Parameters
    ----------
    multiplets_dict : {family_key: [multiplet, ...]}
    X               : (n, d) data array
    n_boot          : bootstrap resamples per multiplet
    random_state    : master seed
    n_jobs          : joblib workers (-1 = all CPUs)
    backend         : joblib backend
    batch_size      : joblib batch size
    verbose         : joblib verbosity

    Returns
    -------
    {family_key: {multiplet_tuple: bootstrap_array}}
    """
    X_np = _ensure_2d_ndarray(X)
    out: Dict[int, Dict[tuple, np.ndarray]] = {}
    ss = np.random.SeedSequence(random_state)

    for family_key, mult_dict in tqdm(
            multiplets_dict.items(), total=len(multiplets_dict), desc="Families"
    ):
        col_index_lists = [list(map(int, m)) for m in mult_dict]
        N = len(col_index_lists)

        # Spawn one independent SeedSequence child per multiplet, then
        # immediately extract a plain integer seed from each child.
        # _boot_worker calls _bootstrap_stats_for_cols internally, so the
        # actual resampling logic lives in one place.
        child_seeds = ss.spawn(N)
        seeds: List[int] = [int(child_seeds[i].generate_state(1)[0]) for i in range(N)]

        with tqdm_joblib(tqdm(total=N, desc=f"Multiplets | family {family_key}")):
            results = Parallel(
                n_jobs=n_jobs, backend=backend, batch_size=batch_size, verbose=verbose
            )(
                delayed(_boot_worker)(
                    X_np=X_np,
                    cols=list(map(int, cols)),
                    n_boot=n_boot,
                    seed=seeds[i],
                )
                for i, cols in enumerate(col_index_lists)
            )

        # _boot_worker returns (tuple_key, array) pairs.
        out[family_key] = {k: v for k, v in results}

    return out


# ===========================================================================
# Section 9 – bca_bootstrap_parallel  (CI computation with milestone progress)
# ===========================================================================

def _bca_for_multiplet(
        X_np: np.ndarray,
        col_idx: Iterable[int],
        omega_obs: float,
        boot_stats: np.ndarray,
        *,
        alpha: float = 0.05,
        n_jobs_jk: int = 1,
        backend: str = "loky",
        show_progress: bool = False,
) -> Tuple[float, float, float, Dict[str, Any]]:
    """
    Compute a full BCa CI for a single multiplet (no adaptive fallback).

    This is a thin wrapper around :func:`_bca_interval` that handles the
    jackknife computation internally.
    """
    X_sub = _slice_cols(X_np, col_idx)
    jk = _jackknife_values(
        X_sub, n_jobs_jk=n_jobs_jk, backend=backend, show_progress=show_progress
    )
    lo, hi, diag = _bca_interval(omega_obs, boot_stats, jk, alpha=alpha)
    return float(omega_obs), lo, hi, diag


def bca_bootstrap_parallel(
        multiplets_dict: Dict[int, Dict[tuple, Dict[str, Any]]],
        X,
        alpha: float = 0.05,
        n_jobs: int = -1,
        backend: str = "loky",
        batch_size: str | int = "auto",
        verbose: int = 0,
        chunk_size: int = 200,
        show_families_bar: bool = True,
) -> Dict[int, Dict[tuple, Tuple[float, float, float, Dict[str, float]]]]:
    """
    Compute adaptive / BCa confidence intervals for all multiplets.

    Multiplets are processed in chunks.  A milestone progress line is printed
    at **0 %, 25 %, 50 %, 75 %, and 100 %** of each family.

    Parameters
    ----------
    multiplets_dict   : nested dict ``{family: {multiplet: {"observed_O": …,
                        "samp_distro": …}}}``
    X                 : (n, d) data array
    alpha             : significance level
    n_jobs            : joblib workers
    backend           : joblib backend
    batch_size        : joblib batch size
    verbose           : joblib verbosity
    chunk_size        : multiplets per parallel batch (controls print frequency)
    show_families_bar : wrap the outer family loop with tqdm

    Returns
    -------
    {family_key: {multiplet_tuple: (omega_obs, lo, hi, diagnostics)}}
    """
    X_np = _ensure_2d_ndarray(X)
    out: Dict[int, Dict[tuple, Tuple[float, float, float, Dict[str, float]]]] = {}

    family_items = list(multiplets_dict.items())
    fam_iter = (
        tqdm(family_items, total=len(family_items), desc="Families", dynamic_ncols=True)
        if show_families_bar
        else family_items
    )

    for family_key, mult_dict in fam_iter:
        col_index_lists = [list(map(int, m)) for m in mult_dict.keys()]
        observed_list = [float(v["observed_O"]) for v in mult_dict.values()]
        distro_list = [
            np.asarray(v["samp_distro"], dtype=float).ravel()
            for v in mult_dict.values()
        ]
        total_M = len(col_index_lists)

        # Pre-allocate result storage for this family.
        results_accum: List[Tuple[float, float, float, Dict]] = [None] * total_M  # type: ignore

        # Milestones at 0 %, 25 %, 50 %, 75 %, 100 % of total_M.
        milestones = {int(total_M * pct) for pct in (0.0, 0.25, 0.50, 0.75, 1.0)}

        print(f"\n[BCa | family {family_key}] starting – {total_M} multiplets")

        for start_idx, idxs in _chunks(list(range(total_M)), chunk_size):
            # Slice the current chunk.
            cols_chunk = [col_index_lists[i] for i in idxs]
            obs_chunk = [observed_list[i] for i in idxs]
            dist_chunk = [distro_list[i] for i in idxs]

            # Run BCa / adaptive CI for each multiplet in the chunk.
            chunk_results = Parallel(
                n_jobs=n_jobs,
                backend=backend,
                batch_size=batch_size,
                verbose=verbose,
            )(
                delayed(bca_or_adaptive_ci_for_multiplet)(
                    X_np,
                    cols,
                    omega_obs,
                    boot_stats,
                    alpha=alpha,
                    mode="adaptive",  # percentile when symmetric, else BCa
                    n_jobs_jk=1,  # serial jackknife per multiplet
                    jk_max=200,  # sub-sample cap for BCa jackknife
                    backend=backend,
                    show_progress=False,
                    random_state=12345,
                )
                for cols, omega_obs, boot_stats in zip(cols_chunk, obs_chunk, dist_chunk)
            )

            # Place results back in their original positions.
            for local_pos, res in enumerate(chunk_results):
                results_accum[start_idx + local_pos] = res

            done = min(start_idx + len(idxs), total_M)

            # Print a progress line at every milestone.
            if done in milestones or done == total_M:
                pct = 100 * done // total_M
                print(
                    f"  [BCa | family {family_key}] {done}/{total_M} multiplets "
                    f"({pct}%)"
                )

        # Assemble the output dict for this family.
        out[family_key] = {
            tuple(col_index_lists[i]): results_accum[i] for i in range(total_M)
        }

    return out


# ===========================================================================
# Section 10 – Hypothesis testing and multiplet selection
# ===========================================================================

def get_observed_hypothesis(observed_O: float, tol: float) -> str:
    """
    Classify the observed O-Information value into one of three hypotheses.

    Parameters
    ----------
    observed_O : observed Ω
    tol        : absolute tolerance around zero

    Returns
    -------
    ``"zero_interaction"`` | ``"redundancy"`` | ``"synergy"``
    """
    if abs(observed_O) < tol:
        return "zero_interaction"
    return "redundancy" if observed_O > 0 else "synergy"


def BCa_CI_check_(
        observed_O: float,
        L: float,
        U: float,
        hypothesis: str,
        tol: float,
) -> bool:
    """
    Check whether the BCa CI supports the given hypothesis.

    A multiplet is *selected* (returns ``True``) when:

    - ``"zero_interaction"`` : zero lies within [L-tol, U+tol]
    - ``"redundancy"``       : the entire CI is strictly positive (L > 0)
    - ``"synergy"``          : the entire CI is strictly negative (L < 0)

    A basic sanity check ensures the observed value lies within [L, U];
    a warning is printed if it does not (and ``False`` is returned).

    Parameters
    ----------
    observed_O : observed Ω
    L, U       : lower and upper CI bounds
    hypothesis : one of the strings returned by :func:`get_observed_hypothesis`
    tol        : tolerance used for the zero-interaction check

    Returns
    -------
    bool
    """
    if not (L <= observed_O <= U):
        print(
            f"Warning: observed value {observed_O:.6g} is outside CI "
            f"[{L:.6g}, {U:.6g}]."
        )
        return False

    if hypothesis == "zero_interaction":
        return L - tol <= 0.0 <= U + tol
    if hypothesis == "redundancy":
        return L > 0.0
    # hypothesis == "synergy"
    return L < 0.0


def BCa_CI_mults_selection(
        BCa_boot: Dict[int, Dict[tuple, tuple]],
        tol: float = 0.05,
) -> Dict[int, Dict[tuple, str]]:
    """
    Filter multiplets whose BCa CI is statistically significant.

    Parameters
    ----------
    BCa_boot : nested dict ``{order: {multiplet: (observed_O, L, U, …)}}``
    tol      : tolerance for the zero-interaction hypothesis

    Returns
    -------
    ``{order: {multiplet: hypothesis}}`` containing only significant multiplets.

    Example input format
    --------------------
    ::

        {3: {(1, 2, 3): (-0.012, -0.086, -0.0025, {"method": "percentile", …})}}
    """
    BCa_selected: Dict[int, Dict[tuple, str]] = {}

    for order, mults in BCa_boot.items():
        for mult, info in mults.items():
            observed_O, L, U = info[0], info[1], info[2]
            hypothesis = get_observed_hypothesis(observed_O, tol)

            if BCa_CI_check_(
                    observed_O=observed_O,
                    L=L,
                    U=U,
                    hypothesis=hypothesis,
                    tol=tol,
            ):
                BCa_selected.setdefault(order, {})[mult] = hypothesis

    return BCa_selected


# ===========================================================================
# Section 11 – Retrieve full info for selected multiplets
# ===========================================================================

def retrieve_selected_multiplets_info(
        BCa_selected: Dict[int, Dict[tuple, str]],
        BCa_boot_full: Dict[int, Dict[tuple, tuple]],
        boot_arrays_full: Dict[int, Dict[tuple, np.ndarray]],
) -> Tuple[
    Dict[int, Dict[tuple, tuple]],
    Dict[int, Dict[tuple, np.ndarray]],
]:
    """
    Retrieve full BCa results and bootstrap arrays for the multiplets that
    survived ``BCa_CI_mults_selection``.

    ``BCa_CI_mults_selection`` keeps only ``{family: {multiplet: hypothesis}}``,
    discarding the CI bounds, diagnostics, and bootstrap arrays needed for
    plotting.  This function re-joins the selection with the full outputs of
    ``bca_bootstrap_parallel`` and ``bootstrap_multiplets_chunked``.

    Parameters
    ----------
    BCa_selected     : output of ``BCa_CI_mults_selection``
                       ``{family: {multiplet: hypothesis_string}}``
    BCa_boot_full    : output of ``bca_bootstrap_parallel``
                       ``{family: {multiplet: (obs, lo, hi, diag)}}``
    boot_arrays_full : output of ``bootstrap_multiplets_chunked``
                       ``{family: {multiplet: bootstrap_array}}``

    Returns
    -------
    BCa_boot_selected    : ``{family: {multiplet: (obs, lo, hi, diag)}}``
                           containing only the selected multiplets
    boot_arrays_selected : ``{family: {multiplet: bootstrap_array}}``
                           containing only the selected multiplets
    """
    BCa_boot_selected: Dict[int, Dict[tuple, tuple]] = {}
    boot_arrays_selected: Dict[int, Dict[tuple, np.ndarray]] = {}

    for family, selected_mults in BCa_selected.items():
        if family not in BCa_boot_full:
            print(f"Warning: family {family} not found in BCa_boot_full — skipped.")
            continue

        BCa_boot_selected[family] = {}
        boot_arrays_selected[family] = {}

        for multiplet in selected_mults:
            # Retrieve full CI info.
            if multiplet not in BCa_boot_full[family]:
                print(f"Warning: multiplet {multiplet} (family {family}) "
                      f"not found in BCa_boot_full — skipped.")
                continue
            BCa_boot_selected[family][multiplet] = BCa_boot_full[family][multiplet]

            # Retrieve bootstrap array.
            if multiplet not in boot_arrays_full[family]:
                print(f"Warning: multiplet {multiplet} (family {family}) "
                      f"not found in boot_arrays_full — skipped.")
                continue
            boot_arrays_selected[family][multiplet] = boot_arrays_full[family][multiplet]

        n_sel = len(BCa_boot_selected[family])
        n_tot = len(BCa_boot_full[family])
        print(f"  Family {family}: {n_sel}/{n_tot} multiplets retrieved.")

    return BCa_boot_selected, boot_arrays_selected
