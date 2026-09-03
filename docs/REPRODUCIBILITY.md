# Reproducibility notes

## Environment

- Recommended Python: 3.10+
- Manuscript OS: Windows 11
- Manuscript hardware: Intel Core i5-10300H 2.50 GHz, 16 GB RAM
- Dependencies: `requirements.txt`

Install and validate:

```bash
python -m pip install -r requirements.txt
python -m tools.validate_repository
pytest -q
```

## Preprocessing

`network_loader.py` applies the same empirical preprocessing pipeline:

1. load the raw graph;
2. convert directed graphs to undirected graphs;
3. aggregate repeated/temporal interactions through the simple unweighted graph representation;
4. remove self-loops;
5. retain the largest connected component;
6. relabel nodes to consecutive integers starting from 0.

`tools/evaluate_preprocessing.py` generates the preprocessing comparison used for Supplementary Table S1.

## Exact maximal-clique enumeration

The code uses exact maximal-clique enumeration, with concrete backends documented explicitly:

- `precompute_rankings.py`: python-igraph `Graph.maximal_cliques()`;
- `msh_methods.enumerate_maximal_cliques()`: python-igraph when available, NetworkX `find_cliques()` as an exact fallback;
- average-degree/clustering stress scripts: NetworkX exact enumeration by default.

The manuscript should therefore use backend-neutral wording such as **exact maximal-clique enumeration** unless it is referring to a specific script/backend.

## Ranking cache and score direction

Run:

```bash
python precompute_rankings.py --force
```

This writes ranking caches under `results/node_rankings/` and maximal-clique caches under `results/clique_cache/`.

The manuscript constraint `C_i` is smaller-is-better, while the released priority score is `1 - C_i`, so larger values mean higher ranking priority. SH and ISH are also returned in the same larger-is-higher priority direction. This common direction is used by the all-node Kendall analysis.

Some caches/scripts retain the legacy key `HOSH`; it is an internal alias for the manuscript method MSH.

## Tie handling

### Static score rankings

For ordinary top-k selection, nodes are sorted by descending score and then ascending node ID after graph relabeling. The common helper rounds scores to 8 decimal places before this deterministic secondary sort to suppress floating-point noise.

### VoteRank internal iterative ties

`msh_methods.calculate_voterank()` explicitly chooses the smallest node ID whenever multiple remaining candidates have the same voting score at an iterative selection step. This implements the Supplementary Table S3 rule.

The migration audit in `docs/VOTERANK_TIE_AUDIT.md` shows that making this rule explicit did not alter the complete VoteRank ordering on any of the nine empirical networks used in the manuscript.

### Analyses where ties must remain ties

- Monotonicity: score groups are retained; no ID-based tie-breaking is used.
- Kendall `tau_b`: raw method scores are aligned by node ID, but ID is not used to break score ties.
- Table S2: deterministic ID, repeated random boundary, and SIR-based boundary strategies are compared as a sensitivity analysis.

## SIR protocol

The discrete-time SIR implementation uses synchronous updates: infected nodes first attempt to infect susceptible neighbors, then recover; newly infected nodes become active spreaders at the next time step. Main experiments use `gamma = 1` and `beta = 2.5 * beta_th`, where `beta_th = <k>/(<k^2>-<k>)`.

For every fixed network/setting/method, topology and ranked seed set are fixed. Stochastic SIR realizations vary across 50 blocks and 20 repetitions per block. The 20 repetitions are averaged before inference, so the statistical sample size is 50 block means.

## Recovery-rate robustness

The current Supplementary Table S5 is reproduced by `experiments/exp16_recovery_rate_tableS5.py`, retained from the first revision. It adjusts `beta` as `gamma` varies so the edge transmissibility remains `2.5 * beta_th`.

## Matched controls

See `experiments/exp10_matched_controls_fig10_tablesS6S7.py` and Supplementary Section S6. The code records candidate generation, matching diagnostics, accepted-control counts, `L_s` approximation details for large selected sets, and one strategy-level value per Monte Carlo block.

## Runtime experiments

The scale, average-degree, and clustering stress tests use independently generated synthetic graphs. Five independent graph instances are used per setting. Graph generation is excluded from ranking-method runtime. Runtime claims concern measured wall-clock behavior; no peak-memory scalability claim is made.

## Full execution plan

List the ordered reproduction steps:

```bash
python reproduce_all.py --list
```

Preview without executing:

```bash
python reproduce_all.py --dry-run
```

Run everything sequentially:

```bash
python reproduce_all.py
```

Individual manuscript-item commands are listed in `docs/figure_table_manifest.csv`.

## Immutable release

After the final numerical regression check, create the GitHub release and immutable archive. Fill `REVISION_VERSION.txt` and cite the same release in the manuscript Availability statement.
