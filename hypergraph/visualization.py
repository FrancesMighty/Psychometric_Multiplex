import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_degree_ecdf(df_deg, show=True, file_name=None, drop_zeros=True):
    plt.figure(figsize=(8, 5))

    for graph_id, row in df_deg.iterrows():
        degrees = row.values.astype(float)
        if drop_zeros:
            degrees = degrees[degrees != 0]
        if degrees.size == 0:
            continue

        x = np.sort(degrees)
        y = np.arange(1, len(x) + 1) / len(x)

        plt.step(x, y, where="post", label=str(graph_id))

    plt.xlabel("Weighted degree")
    plt.ylabel("ECDF")
    plt.title("Degree distributions (ECDF) across graphs")
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        plt.close()


def plot_per_layer_degree_distributions(df_overlap, show=True, file_name=None, drop_zeros=True, bins=20):
    """
    df_overlap: rows = layers, columns = nodes, values = overlapping degree
    """
    plt.figure(figsize=(9, 5))

    for layer, row in df_overlap.iterrows():
        degrees = row.values.astype(float)

        if drop_zeros:
            degrees = degrees[degrees > 0]

        if len(degrees) == 0:
            continue

        plt.hist(
            degrees,
            bins=bins,
            histtype='step',
            density=True,
            alpha=0.9,
            linewidth=1.5,
            label=str(layer)
        )

    plt.xlabel("Overlapping degree")
    plt.ylabel("Density")
    plt.title("Overlapping Degree Distribution Across Layers")
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        plt.close()


def plot_overlapping_degree_distribution(O_i: pd.Series,
                                         show: bool = True,
                                         file_name: str | None = None,
                                         drop_zeros: bool = True,
                                         bins: int = 20):
    """
    O_i: Series indexed by node, values = overlapping degree for each node.
    """
    degrees = O_i.values.astype(float)

    if drop_zeros:
        degrees = degrees[degrees > 0]

    if len(degrees) == 0:
        return

    plt.figure(figsize=(7, 4))
    plt.hist(
        degrees,
        bins=bins,
        histtype='step',
        density=True,
        linewidth=1.5,
    )

    plt.xlabel("Overlapping degree $O_i$")
    plt.ylabel("Density")
    plt.title("Distribution of overlapping degree across nodes")
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        plt.close()


def plot_participation_distribution(P_i, show=True, file_name=None, bins=20):
    plt.figure(figsize=(7, 5))

    values = P_i.values.astype(float)

    plt.hist(values, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel("Participation coefficient $P_i$")
    plt.ylabel("Count")
    plt.title("Distribution of Participation Coefficient")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        plt.close()


def plot_participation_per_node(P_i, show=True, file_name=None):
    plt.figure(figsize=(12, 5))

    P_i.sort_index().plot(kind="bar", color="steelblue", edgecolor="black")

    plt.ylabel("Participation coefficient $P_i$")
    plt.xlabel("Node")
    plt.title("Participation Coefficient per Node")
    plt.ylim(0, 1)  # coefficients always between 0 and 1
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        plt.close()


# hyperedge overlap
def hyperedges_order_overlap(overlap_df, show=True, file_name=None):
    # count hyperedges per (order, overlap)
    counts = (
        overlap_df
        .groupby(["order", "overlap"])
        .size()
        .reset_index(name="n_hyperedges")
    )

    # 2. Plot: #hyperedges vs overlap, for each order
    plt.figure(figsize=(8, 5))

    for ord_val in sorted(counts["order"].unique()):
        df_o = counts[counts["order"] == ord_val].sort_values("overlap")
        plt.plot(
            df_o["overlap"],
            df_o["n_hyperedges"],
            marker="o",
            linestyle="-",
            label=f"order = {ord_val}",
        )

    plt.xlabel("Overlap (number of layers)")
    plt.ylabel("Number of hyperedges")
    plt.title("Hyperedge count vs overlap, by order")
    plt.legend(title="Hyperedge order")
    plt.grid(True)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        plt.close()


# order
def plot_hyperedge_order_discrete(
        stats_df: pd.DataFrame,
        normalize: bool = False,
        title: str = "Hyperedge order distribution by layer (discrete)",
        show=True,
        file_name=None
):
    """
    Plot discrete hyperedge order distribution per layer.

    Parameters
    ----------
    stats_df : pandas.DataFrame
        Output of hyperedge_order_discrete_stats.
    normalize : bool, default=False
        If True, use 'fraction' on y-axis; else 'count'.
    title : str
        Plot title.
    """
    y_col = "fraction" if normalize else "count"
    y_label = "Fraction of hyperedges" if normalize else "Number of hyperedges"

    plt.figure(figsize=(8, 5))

    for layer in sorted(stats_df["layer"].unique()):
        df_l = stats_df[stats_df["layer"] == layer].sort_values("order")
        plt.plot(
            df_l["order"],
            df_l[y_col],
            marker="o",
            linestyle="-",
            label=str(layer),
        )

    plt.xlabel("Hyperedge order |e|")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(title="Layer")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        plt.close()


# he weights (omega) distro
def plot_weights_by_order_for_layer(weights_df: pd.DataFrame, layer_name, show=True, file_name=None):
    df_l = weights_df[weights_df["layer"] == layer_name]
    if df_l.empty:
        print(f"No edges found for layer {layer_name}")
        return

    orders = sorted(df_l["order"].unique())
    data = [df_l[df_l["order"] == o]["weight"].values for o in orders]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, positions=orders, showfliers=False)
    plt.xlabel("Hyperedge order |e|")
    plt.ylabel("Weight")
    plt.title(f"Hyperedge weight distribution by order (layer = {layer_name})")
    plt.grid(True, alpha=0.3)

    if show:
        plt.show()
    else:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        plt.close()


# all layers
def plot_weights_by_order_all_layers(
        weights_df: pd.DataFrame,
        out_dir: str,
        file_prefix: str = "weights_by_order",
        file_ext: str = "png",
        show: bool = False,
):
    """
    For each layer, plot hyperedge weight distribution per order and save to file.
    """
    os.makedirs(out_dir, exist_ok=True)

    for layer in sorted(weights_df["layer"].unique()):
        df_l = weights_df[weights_df["layer"] == layer]
        if df_l.empty:
            continue

        orders = sorted(df_l["order"].unique())
        data = [df_l[df_l["order"] == o]["weight"].values for o in orders]

        plt.figure(figsize=(8, 5))
        plt.boxplot(data, positions=orders, showfliers=False)
        plt.xlabel("Hyperedge order |e|")
        plt.ylabel("Weight")
        plt.title(f"Hyperedge weight distribution by order\nLayer = {layer}")
        plt.grid(True, alpha=0.3)

        # build safe filename
        safe_layer = str(layer).replace(" ", "_")
        filename = f"{file_prefix}_{safe_layer}.{file_ext}"
        filepath = os.path.join(out_dir, filename)

        plt.tight_layout()

        if show:
            plt.show()
        else:
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()


#
def plot_weights_by_layer_all_orders(
        weights_df: pd.DataFrame,
        out_dir: str,
        file_prefix: str = "weights_by_layer",
        file_ext: str = "png",
        show: bool = False,
):
    """
    For each hyperedge order, plot weight distribution across layers and save to file.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Loop over orders instead of layers
    for order in sorted(weights_df["order"].unique()):
        df_o = weights_df[weights_df["order"] == order]
        if df_o.empty:
            continue

        layers = sorted(df_o["layer"].unique())
        data = [df_o[df_o["layer"] == L]["weight"].values for L in layers]

        plt.figure(figsize=(8, 5))
        plt.boxplot(data, positions=range(len(layers)), showfliers=False)
        plt.xticks(range(len(layers)), layers, rotation=45)
        plt.xlabel("Layer")
        plt.ylabel("Weight")
        plt.title(f"Hyperedge weight distribution across layers\nOrder = {order}")
        plt.grid(True, alpha=0.3)

        # safe filename
        filename = f"{file_prefix}_order_{order}.{file_ext}"
        filepath = os.path.join(out_dir, filename)

        plt.tight_layout()

        if show:
            plt.show()
        else:
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()


# tables to txt
def tables_to_txt(file_name, all_tables, top=False, n=10):
    with open(file_name, "w") as f:
        for layer, table in all_tables.items():
            if top:
                top_n = int(len(table) * n / 100)
                table = table.head(top_n)
            table = table[['he_order', 'pattern', 'n_hyperedges', 'n_he_norm', 'avg_weight', 'sum_w']]
            table = table.rename(columns={
                "n_hyperedges": "n_HE",
                "avg_weight": "avg_W"
            })
            pd.set_option("display.max_rows", None)
            print(layer + ':', len(table), 'different patterns.', file=f)
            print(table, file=f)
            print('\n', file=f)
            pd.reset_option("display.max_rows")


# tables to txt
def tables_to_txt_order(file_name, all_tables, top=False, n=10):
    with open(file_name, "w") as f:
        for layer, table in all_tables.items():
            if top:
                top_n = min(len(table), int(len(table) * n / 100))
                table = table.head(top_n)
            table = table[['pattern', 'n_hyperedges', 'n_he_norm', 'avg_weight', 'sum_w']]
            table = table.rename(columns={
                "n_hyperedges": "n_HE",
                "avg_weight": "avg_w"
            })
            pd.set_option("display.max_rows", None)
            print(layer + ':', len(table), 'different patterns.', file=f)
            print(table, file=f)
            print('\n', file=f)
            pd.reset_option("display.max_rows")
