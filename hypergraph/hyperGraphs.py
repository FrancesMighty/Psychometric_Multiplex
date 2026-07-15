import hypernetx as hnx
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import List, Hashable, Dict, Set, Any, Optional, Tuple


def preproc_hyperedges(df_omega_path, origin):
    df_omega = pd.read_csv(df_omega_path)

    hyperedges = {}
    edge_properties = {}

    for idx, row in df_omega.iterrows():
        # extract node columns dynamically
        node_cols = [c for c in df_omega.columns if c.startswith("node")]
        nodes_numeric = sorted([int(row[c]) for c in node_cols if not pd.isna(row[c])])

        edge_name = "-".join(str(x) for x in nodes_numeric)

        # real hyperedge
        hyperedges[edge_name] = nodes_numeric

        if origin == 'scales':
            scale = row["scale"]
        else:
            scale = "-"

        # full edge properties (as before)
        edge_properties[edge_name] = {
            "omega": float(row["omega"]),
            "weight": float(row["omega"]),
            "label": row["label"],
            "order": int(row["order"]),
            "nodes_numeric": nodes_numeric,
            "scale": scale,
            "is_singleton": False
        }

    return hyperedges, edge_properties


def build_hypergraph_from_multiplets(hyperedges, edge_properties, node_label_map):
    """
    Build a HyperNetX hypergraph where:

    - Node IDs are the **numeric IDs** (e.g., 1..91)
    - Node descriptions (labels) stored in node properties
    - ALL nodes appear even if unused in any multiplet

    df_omega_path where file must contain: order, node1..nodeK, label, omega
    node_label_map must be: {1: "desc1", 2: "desc2", ..., 91: "desc91"}
    """
    # ---------------------- 1. REAL hyperedges ----------------------
    # hyperedges, edge_properties : from input

    # ---------------------- 2. Identify isolated nodes ----------------------
    all_nodes = set(node_label_map.keys())

    nodes_in_edges = set().union(*hyperedges.values()) if hyperedges else set()

    missing_nodes = all_nodes - nodes_in_edges
    print('Missing nodes are:', len(missing_nodes), missing_nodes)

    # ---------------------- 3. Add singleton edges for missing nodes ----------------------
    for n in missing_nodes:
        eid = str(n)  # edge ID = string
        hyperedges[eid] = [n]

        edge_properties[eid] = {
            "omega": None,
            "weight": 0,
            "label": None,
            "order": 1,
            "nodes_numeric": [n],
            "scale": "-",
            "is_singleton": True
        }

    # ---------------------- 4. Node properties ----------------------
    node_properties = {
        node_id: {"desc": node_label_map[node_id]}
        for node_id in all_nodes
    }

    # ---------------------- 5. Build Hypergraph ----------------------
    H = hnx.Hypergraph(
        hyperedges,
        edge_properties=edge_properties,
        node_properties=node_properties
    )

    return H


# Weighted Degree _ NODES
def weighted_degrees_normalized(H: hnx.Hypergraph,
                                weight_key="weight",
                                norm="sum"):
    """
    Compute incidence matrix, edge weights, weighted degree, and normalized weighted degree.

    Parameters
    ----------
    H : Hypergraph
    weight_key : str
        Name of edge property containing weights.
    norm : {"sum", "max"}
        "sum": k_i / sum_j k_j
        "max": k_i / max_j k_j

    Returns
    -------
    H_inc : (N,E) incidence matrix
    w     : (E,) edge weights
    k_w   : (N,) weighted degree
    k_norm: (N,) normalized weighted degree
    nodes : list of nodes
    edges : list of edges
    """
    # ---- ordered lists ---- #
    nodes = list(H.nodes)
    edges = list(H.edges)
    N, E = len(nodes), len(edges)

    node_index = {n: i for i, n in enumerate(nodes)}
    edge_index = {e: j for j, e in enumerate(edges)}

    # ---- incidence matrix ---- #
    H_inc = np.zeros((N, E))
    for e in edges:
        j = edge_index[e]
        for n in H.edges[e]:
            i = node_index[n]
            H_inc[i, j] = 1.0

    # ---- weights ---- #
    w = np.zeros(E)
    for e in edges:
        j = edge_index[e]
        props = H.edges[e].properties
        if weight_key in props:
            w[j] = float(props[weight_key])
        elif hasattr(H.edges[e], "weight"):
            w[j] = float(H.edges[e].weight)
        else:
            w[j] = 1.0

    # ---- weighted degree ---- #
    k_w = H_inc @ w

    # ---- normalization ---- #
    k_norm = np.zeros_like(k_w)
    if norm == "sum":
        total = k_w.sum()
        if abs(total) > 0:
            k_norm = k_w / total
    elif norm == "max":
        mx = k_w.max()
        if abs(mx) > 0:
            k_norm = k_w / mx
    else:
        raise ValueError("norm must be 'sum' or 'max'")

    return H_inc, w, k_w, k_norm, nodes, edges


def combine_layer_weighted_degrees(layer_dfs: dict, value="weighted_degree"):
    """
    layer_dfs : dict
        { layer_id : df_from_weighted_degree_df }
    value : str
        Which column to extract (e.g. "weighted_degree" or "normalized_weighted_degree")

    Returns
    -------
    M : DataFrame
        Rows = layers, Columns = nodes
    """
    rows = []

    for layer_id, df in layer_dfs.items():
        # pivot: node -> value
        row = df.set_index("node")[value]
        row.name = layer_id
        rows.append(row)

    # Combine all rows, aligning by node id
    M = pd.DataFrame(rows).fillna(0)

    return M[sorted(M.columns)]


# hyperedges overlap
def hyperedge_overlap(
        layers: List[hnx.Hypergraph],
        layer_names: Optional[List[Hashable]] = None
):
    """
    Compute hyperedge overlap for a multiplex hypergraph.

    Parameters
    ----------
    layers : list of hnx.Hypergraph
        Each element is a layer H_alpha with the *same node set*.
    layer_names : list, optional
        Names/labels for each layer. If None, uses range(len(layers)).

    Returns
    -------
    edge_to_layers : dict
        {frozenset(nodes): set(layer_name, ...)}
    overlap_df : pandas.DataFrame
        One row per distinct interaction (node set) with columns:
        - 'nodes'   : frozenset of nodes in the hyperedge
        - 'order'   : size of the hyperedge
        - 'overlap' : number of layers where it appears
        - 'layers'  : set of layer names where it appears
    """
    # Map each node-set (interaction) -> set of layers where it appears
    edge_to_layers: Dict[frozenset, Set[Any]] = defaultdict(set)

    for H_alpha, l_name in zip(layers, layer_names):
        # H_alpha.edges is a dict-like {edge_id: edge_obj}
        for e_id in H_alpha.edges:
            # HyperNetX edge is iterable over its nodes
            nodes = frozenset(H_alpha.edges[e_id])
            if len(nodes) > 2:
                edge_to_layers[nodes].add(l_name)

    # Build a DataFrame summarizing the distribution
    records = []
    for nodes, layer_set in edge_to_layers.items():
        records.append(
            {
                "nodes": nodes,
                "order": len(nodes),
                "overlap": len(layer_set),  # number of layers in which it repeats
                "layers": layer_set,
            }
        )

    overlap_df = pd.DataFrame.from_records(records)

    return edge_to_layers, overlap_df


def he_overlap_description(overlap_data):
    counts = {}
    print("All hes are ", len(overlap_data))
    print("Hes compairing in more layers are",
          len(overlap_data[(overlap_data["overlap"] > 1) & (overlap_data["order"] > 1)]))
    layers = overlap_data[(overlap_data["overlap"] > 1)]['layers']
    layers = [tuple(e) for e in list(layers)]
    counts['ALL'] = Counter(layers)
    for order in (3, 4, 5):
        print('Order', order)
        layers_order = overlap_data[(overlap_data["overlap"] > 1) & (overlap_data["order"] == order)]['layers']
        print("Hes compairing in more layers are", len(layers_order))
        layers_order = [tuple(e) for e in list(layers_order)]
        counts[order] = Counter(layers_order)
        print('-----')
    return counts


# hyperedges statistics - order distro
def hyperedge_order_discrete_stats(
        layers: List[hnx.Hypergraph],
        layer_names: Optional[List[Hashable]] = None,
        min_order: int = 2,
) -> pd.DataFrame:
    """
    Compute discrete hyperedge order distribution per layer (no binning).

    Parameters
    ----------
    layers : list of hnx.Hypergraph
        One Hypergraph per layer.
    layer_names : list, optional
        Names of layers (for legend). If None, uses range(len(layers)).
    min_order : int, default=2
        Ignore hyperedges smaller than this (e.g. size-1).

    Returns
    -------
    stats_df : pandas.DataFrame
        Columns:
        - 'layer'  : layer name
        - 'order'  : hyperedge order (|e|)
        - 'count'  : #hyperedges of that order in that layer
        - 'fraction' : count / total hyperedges in that layer
    """
    if layer_names is None:
        layer_names = list(range(len(layers)))
    if len(layer_names) != len(layers):
        raise ValueError("layer_names must have same length as layers")

    records = []

    for H, name in zip(layers, layer_names):
        # collect all orders in this layer
        orders = [
            len(H.edges[e_id])
            for e_id in H.edges
            if len(H.edges[e_id]) >= min_order
        ]
        if not orders:
            continue

        orders = np.asarray(orders, dtype=int)
        unique_orders, counts = np.unique(orders, return_counts=True)
        total = counts.sum()

        for ord_val, cnt in zip(unique_orders, counts):
            records.append(
                {
                    "layer": name,
                    "order": int(ord_val),
                    "count": int(cnt),
                    "fraction": float(cnt / total),
                }
            )

    stats_df = pd.DataFrame.from_records(records)
    return stats_df


# hyperedges weights distros (OMEGA)
import pandas as pd
import numpy as np
import hypernetx as hnx
from typing import List, Optional, Hashable, Tuple


def hyperedge_weight_distributions(
        layers: List[hnx.Hypergraph],
        layer_names,
        weight_key: str = "weight",
        min_order: int = 2) -> pd.DataFrame:
    """
    Collect hyperedge weights per order and per layer.

    Parameters
    ----------
    layers : list of hnx.Hypergraph. One Hypergraph per layer.
    layer_names : list, Names of layers. If None, uses range(len(layers)).
    weight_key : str, default="weight" Name of the edge property containing the weight.
    min_order : int, default=2 Ignore hyperedges smaller than this.

    Returns
    -------
    df : pandas.DataFrame
        One row per hyperedge with columns:
        - 'layer'   : layer name
        - 'edge_id' : edge identifier within that layer
        - 'order'   : hyperedge order |e|
        - 'weight'  : hyperedge weight
    """
    records = []

    for H, layer_name in zip(layers, layer_names):
        for e_id in H.edges:
            nodes = list(H.edges[e_id])
            order = len(nodes)
            if order < min_order:
                continue

            edge_obj = H.edges[e_id]
            props = getattr(edge_obj, "properties", {})

            if weight_key in props:
                w = float(props[weight_key])
            elif hasattr(edge_obj, weight_key):
                w = float(getattr(edge_obj, weight_key))
            elif hasattr(edge_obj, "weight"):
                w = float(edge_obj.weight)
            else:
                w = 1.0  # fallback if no weight provided

            records.append(
                {
                    "layer": layer_name,
                    "edge_id": e_id,
                    "order": order,
                    "weight": w,
                }
            )

    df = pd.DataFrame.from_records(records)
    return df


# SUB_SCALES PATTERNS
def pattern_stats_for_order(
        H,
        node_to_subscale: Dict[Hashable, Hashable],
        target_order: int,
        weight_key: str = "weight",
        sorted_by=("avg_weight", "n_hyperedges", "pattern_arity"),
        ascending=(False, False, False),
        arity=True
) -> pd.DataFrame:
    """
    For a given hyperedge order, compute pattern statistics:

    - pattern = set of subscale names present in a hyperedge
    - omega   = hyperedge weight

    For each pattern, compute:
    - number of hyperedges with that pattern
    - average omega
    - pattern arity (number of distinct subscales)
    - pattern (as sorted tuple)

    Returns
    -------
    df : pandas.DataFrame
        Columns:
        - 'order'          : hyperedge order (== target_order)
        - 'pattern'        : tuple of subscale names (sorted)
        - 'pattern_arity'  : number of distinct subscales in the pattern
        - 'n_hyperedges'   : number of hyperedges with this pattern
        - 'avg_weight'     : average weight (omega) over those hyperedges

        Sorted by:
        - n_hyperedges (desc), avg_weight (desc), pattern_arity (desc)
    """
    agg = defaultdict(lambda: {"count": 0, "sum_w": 0.0})

    for e_id in H.edges:
        edge = H.edges[e_id]
        nodes = list(edge)
        order = len(nodes)
        # skip singletons and diff. orders
        if order != target_order or len(nodes) == 1:
            continue

        # pattern = set of all subscales appearing in this hyperedge
        pattern_set = set()
        for n in nodes:
            if n in node_to_subscale:
                pattern_set.add(node_to_subscale[n])
            # if node not in mapping, you can skip or raise; here we skip

        if not pattern_set:
            continue

        # get weight (omega)
        props = getattr(edge, "properties", {})
        if weight_key in props:
            w = float(props[weight_key])
        elif hasattr(edge, weight_key):
            w = float(getattr(edge, weight_key))
        elif hasattr(edge, "weight"):
            w = float(edge.weight)
        else:
            w = 1.0

        key = frozenset(pattern_set)
        agg[key]["count"] += 1
        agg[key]["sum_w"] += w

    records = []
    for pattern_fset, vals in agg.items():
        count = vals["count"]
        sum_w = vals["sum_w"]
        avg_w = sum_w / count if count > 0 else np.nan
        pattern = tuple(sorted(pattern_fset))
        pattern_arity = len(pattern_fset)

        records.append(
            {
                "order": int(target_order),
                "pattern": pattern,
                "pattern_arity": int(pattern_arity),
                "n_hyperedges": int(count),
                "sum_w": sum_w,
                "avg_weight": float(avg_w)
            }
        )

    df = pd.DataFrame.from_records(records)

    if not df.empty:
        if arity:
            df = df[df["pattern_arity"] > 1]
        else:
            df = df[df["pattern_arity"] == 1]
        df = df.sort_values(
            by=list(sorted_by),
            ascending=list(ascending),
            ignore_index=True,
        )
        df["n_he_norm"] = df["n_hyperedges"] / df["n_hyperedges"].sum()

    return df


# SUB_SCALES PATTERNS - per layer (orders mixed)
def pattern_stats_for_layer(
        H,
        node_to_subscale: Dict[Hashable, Hashable],
        exclude_order=False,
        excluded_orders=tuple(),
        weight_key: str = "weight",
        sorted_by=("avg_weight", "n_hyperedges", "pattern_arity"),
        ascending=(False, False, False),
        arity=True
) -> pd.DataFrame:
    """"
    Sorted by: n_hyperedges (desc), avg_weight (desc), pattern_arity (desc)
    """
    agg = defaultdict(lambda: {"he_order": [], "count": 0, "sum_w": 0.0})

    for e_id in H.edges:
        edge = H.edges[e_id]
        nodes = list(edge)
        order = len(nodes)
        # skip singletons
        if len(nodes) == 1:
            continue
        if exclude_order and order in excluded_orders:
            continue
        he_order = len(nodes)

        # pattern = set of all subscales appearing in this hyperedge
        pattern_set = set()
        for n in nodes:
            if n in node_to_subscale:
                pattern_set.add(node_to_subscale[n])
            # if node not in mapping, you can skip or raise; here we skip

        if not pattern_set:
            continue

        # get weight (omega)
        props = getattr(edge, "properties", {})
        if weight_key in props:
            w = float(props[weight_key])
        elif hasattr(edge, weight_key):
            w = float(getattr(edge, weight_key))
        elif hasattr(edge, "weight"):
            w = float(edge.weight)
        else:
            w = 1.0

        key = frozenset(pattern_set)
        agg[key]["count"] += 1
        agg[key]["sum_w"] += w
        agg[key]['he_order'].append(he_order)

    records = []
    for pattern_fset, vals in agg.items():
        count = vals["count"]
        sum_w = vals["sum_w"]
        avg_w = sum_w / count if count > 0 else np.nan
        pattern = tuple(sorted(pattern_fset))
        pattern_arity = len(pattern_fset)

        records.append(
            {
                "he_order": set(vals['he_order']),
                "pattern": pattern,
                "pattern_arity": int(pattern_arity),
                "n_hyperedges": int(count),
                "avg_weight": float(avg_w),
                "sum_w": sum_w
            }
        )

    df = pd.DataFrame.from_records(records)

    if not df.empty:
        if arity:
            df = df[df["pattern_arity"] > 1]
        else:
            df = df[df["pattern_arity"] == 1]
        df = df.sort_values(
            by=list(sorted_by),
            ascending=list(ascending),
            ignore_index=True,
        )
        df["n_he_norm"] = df["n_hyperedges"] / df["n_hyperedges"].sum()

    return df


# return pattern to edges list dict
def pattern_to_hes_list(
        H,
        node_to_subscale: Dict[Hashable, Hashable],
        weight_key, syn):
    """"
    Return dict: pattern of sub_scales : list of edges of the same pattern
    """
    pattern_hes_list = {}

    for e_id in H.edges:
        edge = H.edges[e_id]
        nodes = list(edge)
        # skip singletons
        if len(nodes) == 1:
            continue
        he_order = len(nodes)

        # pattern = set of all subscales appearing in this hyperedge
        pattern_set = set()
        for n in nodes:
            if n in node_to_subscale:
                pattern_set.add(node_to_subscale[n])
            # if node not in mapping, you can skip or raise; here we skip

        if not pattern_set:
            continue

        # get weight (omega)
        props = getattr(edge, "properties", {})
        if weight_key in props:
            w = float(props[weight_key])
        elif hasattr(edge, weight_key):
            w = float(getattr(edge, weight_key))
        elif hasattr(edge, "weight"):
            w = float(edge.weight)
        else:
            w = 1.0

        key = frozenset(pattern_set)
        if key in pattern_hes_list:
            pattern_hes_list[key].append((nodes, w))
        else:
            pattern_hes_list[key] = [(nodes, w)]

    for k, v in pattern_hes_list.items():
        if syn:
            pattern_hes_list[k] = sorted(v, key=lambda t: t[1])
        else:
            pattern_hes_list[k] = sorted(v, key=lambda t: -t[1])

    return pattern_hes_list
