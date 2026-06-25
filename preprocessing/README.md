# Hyperedge Pre-selection for Psychometric Item Networks

A pipeline for estimating higher-order interactions (hyperedges) among psychometric items from ordinal Likert-scale data, combining dual graph estimation, community detection, and community-coherence scoring.

---

## Motivation

Standard psychometric network analysis operates on **dyadic graphs** — edges between pairs of items. But psychological constructs often involve **higher-order dependencies**: groups of three or more items that co-vary in ways not reducible to their pairwise relationships. Hypergraphs can represent this structure, but estimating hyperedges from observed data is non-trivial.

This pipeline addresses that problem by:
1. Building two complementary sparse graphs from the item correlation structure
2. Detecting community structure within those graphs
3. Extracting and scoring candidate hyperedges based on community coherence
4. Expanding candidates beyond pure clique constraints

The result is a ranked list of candidate hyperedges, suitable for downstream confirmatory testing.

---

## Pipeline Overview

```
Two correlation estimators  ──►  Two sparse graphs (EBICglasso)
                                          │
                                 Community detection (Spinglass)
                                          │
                              U (membership) + W (affinity)
                                          │
                         Maximal cliques from both graphs
                                          │
                          Score by community coherence
                                          │
                          Expand beyond clique constraints
                                          │
                      Final ranked candidate hyperedges  ──►  confirmatory testing
```

---

## Stage-by-Stage Description

### Stage 1 — Dual Graph Estimation

Two sparse partial correlation graphs are estimated via **EBICglasso** using two different correlation estimators:

| Estimator | What it captures |
|---|---|
| **Polychoric** | Latent linear structure — assumes observed ordinal responses are discretized normal latent variables. Theoretically grounded for reflective psychometric constructs. Sensitive to latent factor structure. |
| **Nonparanormal** | Rank-based empirical dependence — semiparametric Gaussian copula with free marginals. Robust to skewed distributions, floor/ceiling effects, and non-normal item response patterns. |

Using both estimators is deliberate: they make different assumptions and therefore capture different aspects of the dependence structure. Candidates supported by both graphs are doubly validated; candidates from only one are still included, as they represent structure one lens sees and the other misses.

---

### Stage 2 — Community Detection

**Spinglass** community detection is run on each estimated graph, producing a partition of items into communities. Communities represent groups of items with dense mutual partial correlations — the mesoscale structure of the item space.

---

### Stage 3 — Hard Membership Matrix `U`

```
_make_u_hard(node2comm) → U, node2row, K
```

The community partition is encoded as a binary **N × K one-hot matrix** `U`, where `U[i, k] = 1` iff item `i` belongs to community `k`. This algebraic representation is the input to downstream scoring. The "hard" encoding reflects the crisp partition from Spinglass — each item belongs to exactly one community.

---

### Stage 4 — Community Affinity Matrix `W`

```
_estimate_w_from_R_or_G(R, G, node2comm, node2row) → W
```

A **K × K** matrix where `W[k, l]` is the mean absolute (partial) correlation between all item pairs with one item in community `k` and one in community `l`. Collapses the full N × N correlation structure into a community-level summary:

- **Diagonal** `W[k, k]` — within-community cohesion
- **Off-diagonal** `W[k, l]` — between-community coupling

This matrix is the substrate for scoring: it encodes which community combinations are meaningfully associated.

---

### Stage 5 — Initial Candidate Extraction

```
_initial_cliques(G, k_min, k_max) → candidates
```

**Maximal cliques** are extracted from each graph via the Bron–Kerbosch algorithm (igraph). A maximal clique is a fully connected subgraph that cannot be extended by adding any further node.

Cliques are filtered and decomposed as follows:

| Clique size | Action |
|---|---|
| `< k_min` | Discarded |
| `k_min ≤ size ≤ k_max` | Kept as-is |
| `> k_max` | Decomposed into all k-subsets within `[k_min, k_max]` |

Decomposing oversized cliques ensures no valid sub-hyperedge candidate is discarded. Candidates are pooled from both the polychoric and nonparanormal graphs.

---

### Stage 6 — Scoring and Ranking

```
score_hyperedge(e, U, W) → float
rank_candidates(candidates, U, W) → ranked list
```

Each candidate hyperedge `e` is scored as:

$$S(e) = \kappa(|e|) \cdot \sum_{i < j \in e} \mathbf{u}_i^\top W \mathbf{u}_j$$

Since `U` is one-hot, each term `uᵢᵀ W uⱼ` reduces to `W[comm(i), comm(j)]` — a direct lookup of the affinity between the communities of items `i` and `j`. The score is therefore the **sum of pairwise community affinities** across all pairs in the hyperedge, normalized by a size factor κ.

A hyperedge scores high when all its items belong to communities that are strongly coupled in `W`. κ prevents larger hyperedges from trivially dominating due to having more pairs.

This scoring approach is inspired by **Contisciani et al.**'s generative model for hypergraphs with community structure.

---

### Stage 7 — Candidate Expansion

```
expand_candidates(G, ranked, U, W, ...) → expanded ranked list
```

Pure clique extraction is topologically rigid — every item must be directly connected to every other. Real higher-order structures in psychometric data may not form perfect cliques in the dyadic graph. Expansion relaxes this constraint.

For each top-ranked seed hyperedge:
1. Collect all graph neighbors of any node in the seed
2. For each neighbor, score the enlarged hyperedge
3. Accept the expansion if the score improves by at least `min_gain_ratio` (default 5%)
4. Keep only the top `top_per_seed` (default 3) expansions per seed

Key parameters:

| Parameter | Default | Role |
|---|---------|---|
| `max_size` | 5       | Maximum hyperedge size |
| `min_gain_ratio` | 0.05    | Minimum score improvement to accept expansion |
| `top_per_seed` | 3       | Maximum expansions per seed — controls branching |

This stage recovers **near-clique hyperedges** that are community-coherent but not fully connected in the dyadic graph.

---

## Design Principles

**Multi-evidence aggregation.** Using two correlation estimators with complementary assumptions maximizes candidate coverage without committing to a single statistical model of the data-generating process.

**Separation of topology and semantics.** Cliques provide topologically grounded candidates; community-coherence scoring filters them by semantic/statistical meaning. Neither criterion alone is sufficient.

**Exploratory, not confirmatory.** The pipeline generates a ranked candidate list. Hyperedges are not accepted as ground truth — they are hypotheses to be tested in a confirmatory step (e.g. likelihood ratio tests in a generative hypergraph model, or cross-validation).

**Graceful handling of ordinal data.** Both correlation estimators are appropriate for ordinal Likert data and avoid the distortions introduced by treating ordinal responses as continuous.

---

## Dependencies

- `numpy`
- `igraph`
- `scikit-learn` (or `scipy`)
- A polychoric correlation estimator (e.g. `pingouin`, `semopy`, or R via `polycor`)
- A nonparanormal estimator (e.g. `sklearn` + rank-based preprocessing, or R via `huge`)
- EBICglasso implementation (e.g. via `rpy2` + R `qgraph`, or a Python port)

---

## References

- Contisciani, M., Battiston, F., & De Bacco, C. (2022). *Inference of hyperedges and overlapping communities in hypergraphs*. Nature Coomunications.
- Epskamp, S., Borsboom, D., & Fried, E. I. (2018). Estimating psychological networks and their accuracy. *Behavior Research Methods*.
- Liu, H., Lafferty, J., & Wasserman, L. (2009). The nonparanormal: Semiparametric estimation of high dimensional undirected graphs. *JMLR*.
- Olsson, U. (1979). Maximum likelihood estimation of the polychoric correlation coefficient. *Psychometrika*.
- Bron, C., & Kerbosch, J. (1973). Algorithm 457: finding all cliques of an undirected graph. *Communications of the ACM*.
