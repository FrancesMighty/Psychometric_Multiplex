import re
import pandas as pd
from statsmodels.stats.multitest import multipletests
import numpy as np
from itertools import combinations, product
import networkx as nx
import random

from collections import defaultdict


# ============================================================
# SECTION  1 — preproc utilities
# ============================================================

def preproc_df(path):
    df = pd.read_csv(path).iloc[:, 1:92]  # extract only items' columns - no patients' info
    cols = (pd.Index(df.columns)
            .str.replace(r"^EDI_0*", "", regex=True)
            .astype(str))
    # convert purely numeric labels to int (leave others as-is)
    df.columns = [int(c) if c.isdigit() else c for c in cols]
    return df


# ============================================================
# SECTION 2 — SUMMARIES - descriptive funcs
# ============================================================

def summary(dict_):
    perm_summary = {}
    for order, ms in dict_.items():
        perm_summary[order] = {}
        for m, info in ms.items():
            perm_summary[order][m] = (info[0], info[1])
    return perm_summary


def summary_omega(dict_):
    perm_summary = {}
    for order, ms in dict_.items():
        perm_summary[order] = {}
        for m, info in ms.items():
            perm_summary[order][m] = info['Omega']
    return perm_summary


def describe_omega_partition(o_p_diags):
    for diag, nature_dict in o_p_diags.items():
        print('Diagnosis:', diag)
        for nature, mults in nature_dict.items():
            print('\t Interation type:', nature, 'there are:')
            for order in mults:
                print('\t \t Order:', order, '# multiplets:', len(mults[order]))


def describe_hyperedges(hyper_diags):
    for diag, hyper_dict in hyper_diags.items():
        print('Diagnosis:', diag)
        for order, mults in hyper_dict.items():
            print('\t Order', str(order) + ':')
            syn_count = len([m for m, nature in mults.items() if nature == 'synergy'])
            red_count = len([m for m, nature in mults.items() if nature == 'redundancy'])
            print('\t \t Synergy:', syn_count, 'Redundancy:', red_count)


# ============================================================
# SECTION 3 — Filtering utilities (FDR + thresholding)
# ============================================================

def fdr_neuter_filter(perm_dict, alpha=0.05, omega_threshold=0.10, method='fdr_bh'):
    """
    Apply FDR correction and keep only significant entries that also show
    a meaningful effect size.

    A multiplet is retained only if BOTH conditions hold:
        1. FDR-corrected p-value is significant (reject = True)
        2. |Ω_observed| >= omega_threshold  (non-negligible interaction)

    Parameters
    ----------
    perm_dict       : direct output of perm_bootstrap_parallel
    alpha           : FDR significance level (default 0.05)
    method          : multiple testing correction method (default 'fdr_bh')
    omega_threshold : minimum |Ω| to consider a non-zero interaction
                      (default 0.15 — domain-informed for Likert data)

    Returns
    -------
    fdr_families : dict
        {order: {multiplet_key: {observed_O, pval, qval}}}
        Only multiplets passing both filters are included.
    """
    fdr_families = {}
    for order, pval_dict in perm_dict.items():
        keys = list(pval_dict.keys())
        observed = [v[0] for v in pval_dict.values()]
        pvals = np.array([float(v[1]) for v in pval_dict.values()])

        # FDR correction
        reject, qvals, _, _ = multipletests(pvals, alpha=alpha, method=method)

        # Keep only significant and synergistic/redundant ones - discard neutral ones
        significant_dict = {
            k: {
                'observed_O': o,
                'pval': p,
                'qval': q
                # 'samp_distro': boot_stats[k] to be added after filtering
            }
            for k, o, p, q, r in zip(keys, observed, pvals, qvals, reject)
            if r and abs(o) >= omega_threshold
        }
        fdr_families[order] = significant_dict

    return fdr_families


# ============================================================
# SECTION 4 — PARTITIONING MULTIPLETS - descriptive funcs
# ============================================================

def get_interaction_nature(info, alpha, omega_threshold):
    omega, pval = info['observed_O'], info['pval']
    if pval > alpha:
        return 'non_significant'
    if abs(omega) < omega_threshold:
        return 'neuter'
    if omega > 0:
        return 'redundant'
    else:
        return 'synergistic'


def omega_partition(orders_dict, alpha, omega_threshold):
    '''
    partition by multiplets nature
        synergistic: Ω < 0
        redudant Ω > 0
        neuter |Ω| < omega_threshold or statistically
                     non significant (pval > alpha)
    '''
    synergy_dict = {k: {} for k in orders_dict}
    redundancy_dict = {k: {} for k in orders_dict}
    neutrality_dict = {k: {} for k in orders_dict}  # discarded ones - should be none after filtering, but check
    non_sig_dict = {k: {} for k in orders_dict}
    for order, mults in orders_dict.items():
        for mult, info in mults.items():
            interaction = get_interaction_nature(info, alpha, omega_threshold)
            if interaction == 'non_significant':
                non_sig_dict[order][mult] = info['observed_O']
            elif interaction == 'neuter':
                neutrality_dict[order][mult] = info['observed_O']  # save omega value to use later as hypergraph weight
            elif interaction == 'synergistic':
                synergy_dict[order][mult] = info['observed_O']
            elif interaction == 'redundant':
                redundancy_dict[order][mult] = info['observed_O']
    return {'syn': synergy_dict,
            'red': redundancy_dict,
            'neut': neutrality_dict,
            'non_sig': non_sig_dict}


# ============================================================
# SECTION 5 — BCa utilities
# ============================================================

def get_BCa_input_dict_from_fdr(samp_D, fdr_dict):
    '''
    adding to the dict of the fdr survived multiplets
    info about the BCa sample_distro
    '''
    for order, mult_dict in fdr_dict.items():
        for mult, info in mult_dict.items():
            fdr_dict[order][mult]['samp_distro'] = samp_D[order][mult]
    return fdr_dict


def get_BCa_input_dict(samp_D, perm_dict):
    BCa_dict = {}
    for order, mult_dict in perm_dict.items():
        BCa_dict[order] = {}
        for mult, info in mult_dict.items():
            BCa_dict[order][mult] = {
                'observed_O': info[0],
                'samp_distro': samp_D[order][mult]
            }
    return BCa_dict


# ============================================================
# SECTION 6 — MULTIPLETS utilities
# ============================================================

def all_mults_combinations(D: dict, orders=(3, 4, 5)):
    """
    D: {scale: list}
    Generate all combinations (multiplets) of given orders from D
    """
    all_combo = {}
    for scale, L in D.items():
        all_combo[scale] = {}
        for k in orders:
            all_combo[scale][k] = list(combinations(L, k))
    return all_combo


def interscale_mults_combinations(D: dict, orders=(3, 4, 5)):
    """
    Generate all inter-scale multiplets of specified orders from pairs of scales.
    Inter-scale multiplets: combinations of items drawn from TWO DIFFERENT scales,
    capturing higher-order interactions ACROSS scales rather than within a single scale.
    Parameters:
    D : dict mapping scale names to lists of item indices.
        Format: {scale_name: [item1, item2, ...], ...}
    Returns:
        { (scale1, scale2): {order: list_of_multiplets} }
    """
    all_combo = {}

    # Get all unordered pairs of distinct scales
    for scale1, scale2 in combinations(D.keys(), 2):
        items1, items2 = D[scale1], D[scale2]
        pair_key = (scale1, scale2)
        all_combo[pair_key] = {}

        for k in orders:
            combined_items = items1 + items2
            # Only include combinations that draw from both scales (cross-scale)
            # A valid inter-scale multiplet MUST include:
            # at least one item from scale1 and one item from scale2
            all_k_combos = [
                combo for combo in combinations(combined_items, k)
                if any(i in items1 for i in combo) and any(i in items2 for i in combo)
            ]
            all_combo[pair_key][k] = all_k_combos

    return all_combo


def interscale_mults_combinations_G(D: dict, G: nx.Graph, orders=(3, 4, 5)):
    def is_pairwise_connected(nodes):
        edge_count = sum(1 for u, v in combinations(nodes, 2) if G.has_edge(str(u), str(v)))
        return edge_count >= len(nodes)

    all_combo = {}

    # Iterate over all unordered pairs of scales
    for scale1, scale2 in combinations(D.keys(), 2):
        items1, items2 = D[scale1], D[scale2]
        pair_key = (scale1, scale2)
        all_combo[pair_key] = {}

        for k in orders:
            combined_items = items1 + items2

            # Generate combinations that include at least one item from each scale
            valid_combos = [
                combo for combo in combinations(combined_items, k)
                if any(i in items1 for i in combo)
                   and any(i in items2 for i in combo)
                   and is_pairwise_connected(combo)
            ]

            all_combo[pair_key][k] = valid_combos

    return all_combo


def filter_and_sample_multiplets(multiplets_dict, sample_size=400):
    """
    From the output of interscale_mults_combinations(), keep all order-3 multiplets
    and sample up to `sample_size` multiplets for higher orders.

    Returns:
        Same structure, with filtered/sampled multiplets.
    """
    filtered = {}

    for scale_pair, order_dict in multiplets_dict.items():
        filtered[scale_pair] = {}
        for k, combo_list in order_dict.items():
            if k == 3:
                filtered[scale_pair][k] = combo_list  # keep all
            else:
                n = min(sample_size, len(combo_list))
                filtered[scale_pair][k] = random.sample(combo_list, n)

    return filtered


def flatten_multiplets_to_df(inter_candidate):
    """
    Convert nested {scale_pair: {order: list_of_tuples}} structure
    into flat {order: DataFrame} where each row is a multiplet,
    and adds a column for the scale pair label.
    """
    flat_dict = defaultdict(list)

    for scale_pair, order_dict in inter_candidate.items():
        for k, multiplets in order_dict.items():
            for combo in multiplets:
                flat_dict[k].append({
                    **{f"{i + 1}": item for i, item in enumerate(combo)},
                    "scale_pair": f"{scale_pair[0]}–{scale_pair[1]}"
                })

    # Convert each list to a DataFrame
    return {
        k: pd.DataFrame(rows) for k, rows in flat_dict.items()
    }


def get_multiplets_as_dict_list(op_diags):
    dl = {}
    for diag, op_part in op_diags.items():
        syn = {order: [mult for mult in op_part['syn'][order]] for order in op_part['syn']}
        red = {order: [mult for mult in op_part['red'][order]] for order in op_part['red']}
        dl[diag] = {order: syn.get(order, []) + red.get(order, []) for order in set(syn) | set(red)}
    return dl


def get_multiplets_as_dict_list_F(f_diags):
    '''
    Prepare multiplets for BCa data format
    from: dict(multiplet as tuple : dict(observed_O, pval, qval))
    to  : dict(diag, dict(order: list of multiplets))
    '''
    dl = {}
    for diag, f_mults in f_diags.items():
        dl[diag] = {order: [mult for mult in f_mults[order]] for order in f_mults}
    return dl


def all_multiplets_to_dataframes(multiplets_dict, include_label=False):
    """
    Convert a dict of the form: {order: {(tuple): label, ...}, ...}
    into: {order: DataFrame}
    """
    out = {}

    for order, tuples_dict in multiplets_dict.items():
        tuples = list(tuples_dict.keys())
        df = pd.DataFrame(tuples, columns=[f"node{i + 1}" for i in range(order)])

        if include_label:
            df["label"] = list(tuples_dict.values())

        df = df.sort_values(df.columns.tolist()).reset_index(drop=True)
        out[order] = df

    return out


# ============================================================
# SECTION 7 —  SCORING - Nodes' perspective
# ============================================================

def process_row_wn(row, desc_dict):
    sel = row[row > 0].sort_values(ascending=False)
    return {
        "row": row.name,
        "nodes": sel.index,
        "values": sel.values
    }


def sum_values_per_scale(names, values, scale_of):
    # nodes: of specific layer, scores: weighted degree, items_scales_dict: 1: DT
    score = defaultdict(float)
    for name, val in zip(names, values):
        scale = scale_of[name]
        score[scale] += val
    return dict(score)


def compute_overlapping_degree(df_degrees: pd.DataFrame, norm=False) -> pd.Series:
    """
    df_degrees: rows = layers, columns = nodes, values = k_i_alpha
    Returns: Series indexed by node, with overlapping degree O_i.
    """
    O_i = df_degrees.sum(axis=0)  # sum over layers
    if norm:
        denom = O_i.abs().sum()  # L1 norm
        if denom == 0:
            return O_i * 0
        O_norm = O_i / denom
        return O_norm
    return O_i


def compute_participation_coefficient(df_degrees: pd.DataFrame) -> pd.Series:
    """
    df_degrees: rows = layers, columns = nodes, values = k_i_alpha
    Returns: Series indexed by node, with participation coefficient P_i.
    """
    M = df_degrees.shape[0]  # number of layers
    O_i = compute_overlapping_degree(df_degrees)  # overlapping degree per node

    # Avoid division by zero: nodes with O_i = 0 have undefined P_i
    # Here we set them to 0 (you could also set to np.nan if you prefer).
    valid = O_i > 0

    # Fractions k_i_alpha / O_i for valid nodes
    frac = df_degrees.loc[:, valid].div(O_i[valid], axis=1)

    # Sum over layers of (k_i_alpha / O_i)^2
    sum_squares = (frac ** 2).sum(axis=0)

    # Apply formula: P_i = (M/(M-1)) * (1 - sum_squares)
    P_i = pd.Series(0.0, index=df_degrees.columns)  # default 0 for O_i = 0
    P_i[valid] = (M / (M - 1)) * (1.0 - sum_squares)

    return P_i


def compute_participation_coefficient_v2(df_degrees: pd.DataFrame) -> pd.Series:
    """
    df_degrees: rows = layers, columns = nodes, values = k_{i,alpha}^{(w)} >= 0
    Returns: Series indexed by node, with participation coefficient P_i.
             P_i is NaN for nodes with o_i = 0 (never active).
    """
    M = df_degrees.shape[0]  # number of layers
    O_i = compute_overlapping_degree(df_degrees)

    # Initialize with NaN: inactive nodes are explicitly undefined
    P_i = pd.Series(np.nan, index=df_degrees.columns, dtype=float)

    # Active nodes
    valid = O_i > 0

    # p_{i,alpha} = k_{i,alpha} / o_i
    frac = df_degrees.loc[:, valid].div(O_i[valid], axis=1)

    # sum_alpha p_{i,alpha}^2
    sum_squares = (frac ** 2).sum(axis=0)

    # Participation coefficient
    P_i[valid] = (M / (M - 1)) * (1.0 - sum_squares)

    return P_i


### INTRA _ INTER
# STRUCTURE: BCa_selected_diags_scales_INTRA_loaded['ANR']['A'][3] = {66, 68, 75): 'synergy'...} - get all keys
def get_selected_INTRA(multiplets_data):
    selected_ms = {}
    for diagnose, scales in multiplets_data.items():
        selected_ms[diagnose] = {}
        for scale, orders in scales.items():
            selected_ms[diagnose][scale] = {}
            for order, ms in orders.items():
                selected_ms[diagnose][scale][order] = list(ms.keys())
    return selected_ms


# STRUCTURE: BCa_selected_diags_INTER_loaded['ANR'][3] = {66, 68, 75): 'synergy'...} - get all keys
def get_selected_INTER(multiplets_data):
    selected_ms = {}
    for diagnose, orders in multiplets_data.items():
        selected_ms[diagnose] = {}
        for order, ms in orders.items():
            selected_ms[diagnose][order] = list(ms.keys())
    return selected_ms


# further filtering
def filtering_he(prop_d, d, threshold=0.10):
    merged_filtered = {}
    merged_prop_filtered = {}
    for layer, info in prop_d.items():
        below, singletons = 0, 0
        merged_filtered[layer] = {}
        merged_prop_filtered[layer] = {}
        for m, mult in info.items():
            if mult['is_singleton'] is True:
                singletons += 1
                print(m, 'is singleton')
                continue
            elif abs(mult['omega']) >= threshold:
                below += 1
                merged_filtered[layer][m] = d[layer][m]
                merged_prop_filtered[layer][m] = prop_d[layer][m]
        print(layer, len(info.keys()), singletons, below, len(info.keys()) - below)
    return merged_filtered, merged_prop_filtered
