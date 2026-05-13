from __future__ import annotations
from dataclasses import dataclass

import math
import numpy as np
import pandas as pd
import igraph as ig
from typing import Dict, Hashable, Optional, Iterable, List, Sequence, Tuple, Set
from heapq import nlargest


# ---------- Communities & priors ----------

def spinglass_communities_ig(
        g: ig.Graph,
        spins: int = 12,
        gamma: float = 1.0,
        name_attr: str = "name",
) -> Dict[Hashable, int]:
    """
    Spinglass on an undirected, connected igraph.Graph.
    ----------
    Parameters:
        g : ig.Graph
            Must be undirected and connected. If weighted, use g.es['weight'].
            Signed edges are auto-detected (neg implementation if any weight < 0).
        spins : int - Upper bound on number of communities.
        gamma : float - Resolution parameter.
        name_attr : str - Vertex attribute to use as node id in the output (fallback: vertex index).
    -------
    Returns: dict[Hashable: node_id, int:community_id]
    """
    # assume undirected + connected
    # (asserts are helpful; remove if you prefer silent behavior)
    assert not g.is_directed(), "Graph must be undirected."
    assert g.is_connected(), "Graph must be connected."

    weights = g.es["weight"] if "weight" in g.es.attributes() else None
    has_negative = any(w < 0 for w in weights) if weights is not None else False
    implementation = "negative" if has_negative else "original"

    node_ids = g.vs[name_attr] if name_attr in g.vs.attributes() else list(range(g.vcount()))

    cl = g.community_spinglass(
        weights=weights,
        spins=spins,
        gamma=gamma,
        implementation=implementation,
    )
    membership = cl.membership  # list[int], length = g.vcount()
    # return node2comm
    return {node_ids[i]: int(membership[i]) for i in range(g.vcount())}


def _make_u_hard(node2comm):
    """
    Parameters:
        node2comm:
            dict[Hashable: node_id, int:community_id]
            result of Spinglass community detection
    -------
    Returns: (U, node2row, K)
    -------
    Build hard one-hot membership matrix U (shape N x K) aligned to node ids [0..N-1] if possible.
    If node ids are arbitrary ints, we still build a dense mapping  node_id -> row index.
    """
    nodes = sorted(node2comm.keys())
    node2row = {n: i for i, n in enumerate(nodes)}
    K = max(node2comm.values()) + 1
    U = np.zeros((len(nodes), K), dtype=float)
    for n, c in node2comm.items():
        U[node2row[n], c] = 1.0
    return U, node2row, K  # Tuple[np.ndarray, Dict[int, int], int]


def _estimate_w_from_R_or_G(
        R: Optional[np.ndarray],
        G: ig.Graph,
        node2comm,  # dict communities mapping
        node2row  # Dict[int, int]
) -> np.ndarray:
    """
    Parameters:
        R: optional, the correlation matrix
        G: the ig.Graph graph object
        node2comm:
            dict[Hashable: node_id, int:community_id]
            result of Spinglass community detection - dict communities mapping
        node2row:
            nodes enumeration: Dict[int, int]
    -------
    Returns:
        community affinity matrix W
    -------
    Estimate community affinity matrix w_{kl}.
    If R is provided, use |R_ij| aggregated across (comm k, comm l).
    Else use |edge weight| (or 1.0 if unweighted) from G.
    """
    K = max(node2comm.values()) + 1
    num = np.zeros((K, K), dtype=float)
    den = np.zeros((K, K), dtype=float)

    if R is not None:
        # R is assumed square over node indices 0..N-1; if your node ids differ,
        # we require them to match sorted nodes used to build node2row
        # Aggregate only existing pairs present in G nodes.
        nodes = sorted(node2row.keys())
        for i_idx, i in enumerate(nodes):
            ci = node2comm[i]
            ri = node2row[i]
            for j_idx in range(i_idx + 1, len(nodes)):
                j = nodes[j_idx]
                cj = node2comm[j]
                rj = node2row[j]
                val = abs(float(R[ri, rj]))
                num[ci, cj] += val
                num[cj, ci] += val
                den[ci, cj] += 1.0
                den[cj, ci] += 1.0
    else:
        print('R is None')
        # use the indeces 1 based, not the names from node labels
        names = G.vs["name"] if "name" in G.vs.attributes() else None
        weights = G.es["weight"] if "weight" in G.es.attributes() else None

        for idx, e in enumerate(G.es):
            i, j = e.source, e.target

            key_i = names[i] if names else i
            key_j = names[j] if names else j
            ci, cj = node2comm[key_i], node2comm[key_j]

            w = abs(float(weights[idx])) if weights is not None else 1.0

            num[ci, cj] += w
            num[cj, ci] += w

            den[ci, cj] += 1.0
            den[cj, ci] += 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        W = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    # Small ridge to avoid zero-rows (keeps scoring stable)
    eps = 1e-8
    W = W + eps
    return W


# ---------- Clique discovery & scoring ----------

def _initial_cliques(
        G: ig.Graph,
        k_min: int = 3,
        k_max: int = 6,
        max_count=None):
    """
    Get maximal cliques and keep only those whose size is within [k_min, k_max].
    Then decompose larger maximal cliques into all k-subsets in range (to not miss candidates).
    """
    max_cliques_idx = G.maximal_cliques()  # lists of vertex indices

    '''
    # if you want names instead of indices:
    names = G.vs["name"] if "name" in G.vs.attributes() else None
    max_cliques = [
        [names[i] for i in C] if names is not None else list(C)
        for C in max_cliques_idx
    ]
    '''

    out = set()

    for C in max_cliques_idx:
        C = tuple(sorted(int(x) for x in C))
        if len(C) < k_min:
            continue
        if len(C) <= k_max:
            out.add(C)
        else:
            # break big maximal cliques into all k-subsets within range
            for k in range(k_min, k_max + 1):
                for subset in _k_subsets(C, k):
                    out.add(subset)

    cliques = sorted(out, key=lambda t: (len(t), t))
    if max_count is not None and len(cliques) > max_count:
        cliques = cliques[:max_count]
    return cliques


def _k_subsets(items: Sequence[int], k: int) -> Iterable[Tuple[int, ...]]:
    from itertools import combinations
    return combinations(items, k)


def _kappa(size: int, mode: str = "factorial") -> float:
    if mode == "factorial":
        return 1.0 / math.factorial(size)
    if mode == "none":
        return 1.0
    # mild penalty (default)
    return 1.0 / (size ** 1.5)


def score_hyperedge(
        e: Sequence[int],
        U: np.ndarray,
        W: np.ndarray,
        kappa_mode: str = "factorial",
) -> float:
    """
    Contisciani-inspired score: S(e) = kappa(|e|) * sum_{i<j in e} u_i^T W u_j
    with hard memberships (u_i are one-hot rows of U).
    """
    nodes = list(e)
    acc = 0.0
    for a in range(len(nodes)):
        # ia = node2row[nodes[a]]
        # u_i = U[ia]
        u_i = U[a]
        for b in range(a + 1, len(nodes)):
            u_j = U[b]
            acc += float(u_i @ W @ u_j)
    return _kappa(len(nodes), kappa_mode) * acc


def rank_candidates(
        candidates: List[Tuple[int, ...]],
        U: np.ndarray,
        W: np.ndarray
) -> List[Tuple[Tuple[int, ...], float]]:
    # print('Candidates are', len(candidates))
    scored = [(c, score_hyperedge(c, U, W)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ---------- Expansion (hierarchical growth) ----------
def expand_candidates(
        G: ig.Graph,
        ranked: List[Tuple[Tuple[int, ...], float]],
        U: np.ndarray,
        W: np.ndarray,
        kappa_mode: str,
        max_size: int = 7,
        top_per_seed: int = 3,
        min_gain_ratio: float = 0.05,
) -> List[Tuple[Tuple[int, ...], float]]:
    """
    Try to add one neighbor at a time if score improves by at least `min_gain_ratio`.
    Limit to `top_per_seed` best expansions per seed to control branching.
    """
    seen: Set[Tuple[int, ...]] = {c for c, _ in ranked}
    out = list(ranked)

    for seed, s_seed in ranked:
        if len(seed) >= max_size:
            continue
        # neighbors that connect to at least one node in seed
        neighs = set()
        for v in seed:
            neighs.update(G.neighbors(v))
        neighs.difference_update(seed)

        improvers: List[Tuple[Tuple[int, ...], float]] = []
        for u in neighs:
            cand = tuple(sorted((*seed, int(u))))
            if cand in seen:
                continue
            s_cand = score_hyperedge(cand, U, W, kappa_mode)
            if s_cand >= s_seed * (1.0 + min_gain_ratio):
                improvers.append((cand, s_cand))

        if improvers:
            best = nlargest(min(top_per_seed, len(improvers)), improvers, key=lambda t: t[1])
            for c in best:
                if c[0] not in seen:
                    seen.add(c[0])
                    out.append(c)

    # de-duplicate and re-rank globally
    out = list({c: s for c, s in out}.items())
    out.sort(key=lambda x: x[1], reverse=True)
    return out


# ---------- Packaging for perm_bootstrap_parallel ----------

def to_multiplets_dict(
        ranked: List[Tuple[Tuple[int, ...], float]],
        size_range: Tuple[int, int] = (3, 7),
        top_per_size: Optional[int] = None,
) -> Dict[int, pd.DataFrame]:
    """
    Group candidates by size k and return {k: DataFrame[node1..nodeK]}.
    Optionally keep only top_per_size per k by score.
    """
    by_k: Dict[int, List[Tuple[Tuple[int, ...], float]]] = {}
    for c, s in ranked:
        k = len(c)
        if k < size_range[0] or k > size_range[1]:
            continue
        by_k.setdefault(k, []).append((c, s))

    out: Dict[int, pd.DataFrame] = {}
    for k, lst in by_k.items():
        lst.sort(key=lambda x: x[1], reverse=True)
        if top_per_size is not None:
            lst = lst[:top_per_size]
        rows = [c for c, _ in lst]
        cols = [f"node{i + 1}" for i in range(k)]
        out[k] = pd.DataFrame(rows, columns=cols, dtype=int)
    return out


# ---------- Entrypoint ----------

@dataclass
class CandidateConfig:
    k_min: int = 3
    k_max: int = 6
    max_initial: Optional[int] = None
    kappa_mode: str = "factorial"
    expand_max_size: int = 7
    expand_top_per_seed: int = 3
    expand_min_gain_ratio: float = 0.05
    top_per_size: Optional[int] = None  # e.g., keep best 100 per size


def shift_multiplets_one_based(multiplets_dict):
    out = {}
    for k, df in multiplets_dict.items():
        df2 = df.copy()
        for col in df2.columns:
            df2[col] = df2[col].astype(int) + 1
        out[k] = df2
    return out


def build_candidate_multiplets(
        G: ig.Graph,
        R: Optional[np.ndarray] = None,
        cfg: CandidateConfig = CandidateConfig(),
) -> Dict[int, pd.DataFrame]:
    """
    Full pipeline:
      1) detect communities -> U
      2) estimate W from R or G
      3) get initial cliques in [k_min,k_max]
      4) rank by Contisciani-style score
      5) expand promising seeds
      6) package as {k: DataFrame[node1..nodeK]}
    """
    # node2comm = _detect_communities(G, method=community_method)
    node2comm = spinglass_communities_ig(G, spins=12, gamma=1.0)  # maps each vertex  to community id

    U, node2row, _K = _make_u_hard(node2comm)
    W = _estimate_w_from_R_or_G(R, G, node2comm, node2row)

    seeds = _initial_cliques(G, cfg.k_min, cfg.k_max, cfg.max_initial)
    ranked = rank_candidates(seeds, U, W)  # no need for node2row - use directly indeces instead

    expanded = ranked

    if cfg.expand_max_size > cfg.k_max:
        kappa_mode = 'mild_penalty'
        # re-score uses kappa inside score fn; nothing else needed
        # here for scoring we did not used factorial penalty, but a linear one
        expanded = expand_candidates(
            G, ranked, U, W, kappa_mode,
            max_size=cfg.expand_max_size,
            top_per_seed=cfg.expand_top_per_seed,
            min_gain_ratio=cfg.expand_min_gain_ratio,
        )

    multiplets_dict = to_multiplets_dict(expanded, size_range=(cfg.k_min, cfg.expand_max_size),
                                         top_per_size=cfg.top_per_size)
    return shift_multiplets_one_based(multiplets_dict)


def load_correlation_matrix_R(R_path: str) -> np.ndarray:
    """
    Load a correlation matrix saved as CSV with names in the first column.
    Ensures symmetry and unit diagonal.
    """
    df = pd.read_csv(R_path, index_col=0)
    corr_R = df.to_numpy(dtype=float, copy=True)
    corr_R = 0.5 * (corr_R + corr_R.T)
    np.fill_diagonal(corr_R, 1.0)
    return corr_R
