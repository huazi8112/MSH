# Mesoscopic Structural Holes (MSH)

Code, data, and reproducibility material for the revised manuscript:

**Mesoscopic Structural Holes for Redundancy-Aware Node Ranking in Complex Networks**

MSH is a redundancy-aware topology-based **node-ranking method** for simple, unweighted, undirected pairwise graphs. Maximal cliques inferred from pairwise adjacency are treated as mesoscopic cohesive units. MSH combines effective dependence on affiliated cliques with inter-clique redundancy and assigns one structural priority score to each node.

The manuscript defines an MSH constraint coefficient `C_i`, where a smaller value indicates a stronger mesoscopic structural-hole position. The implementation exports the equivalent priority score `1 - C_i`, so throughout the released code **larger score = higher ranking priority**.

The revised evaluation focuses on ranking-derived node sets rather than isolated-node SIR superiority. It includes collective multi-seed SIR spreading, ranking resolution, node-set redundancy, matched controls, all-node structural rank correlations, ablation studies, and computational stress tests.

## Repository layout

```text
MSH_second_revision/
├── README.md
├── CITATION.cff
├── CHANGELOG.md
├── REVISION_VERSION.txt
├── requirements.txt
├── reproduce_all.py
├── msh_methods.py
├── hosh_methods.py                  # compatibility wrapper for legacy cache keys
├── network_loader.py
├── precompute_rankings.py
├── configs/
│   └── revision_parameters.json
├── docs/
│   ├── DATASETS.md
│   ├── REPRODUCIBILITY.md
│   ├── STATISTICAL_PROTOCOL.md
│   ├── REVIEWER_RESPONSE_MAPPING.md
│   ├── VOTERANK_TIE_AUDIT.md
│   ├── FINAL_AUDIT.md
│   ├── figure_table_manifest.csv
│   ├── RELEASE_CHECKLIST.md
│   └── *_sha256.txt
├── experiments/
├── tools/
├── tests/
├── reference_results/
│   ├── Table_S6.xlsx
│   ├── Table_S7.xlsx
│   ├── Table_S10.xlsx
│   └── Table_S11.xlsx
├── networks_data/
└── results/
```

Some revision scripts and caches retain the historical internal identifier `HOSH`. Public manuscript labels use **MSH**. `msh_methods.py` accepts both identifiers so the released code remains compatible with the exact revision workflow.

## Installation

Python 3.10 or later is recommended.

```bash
python -m pip install -r requirements.txt
```

Main dependencies are NetworkX, NumPy, pandas, SciPy, Matplotlib, tqdm, openpyxl, python-igraph, and python-louvain.

## Quick validation

Before running the full experiments:

```bash
python -m tools.validate_repository
python -m tools.validate_reference_results
pytest -q
```

The lightweight validation checks repository structure, dataset files, reference tables, source syntax, and the deterministic VoteRank tie rule. It does **not** rerun the expensive SIR experiments.

## Data preprocessing

All nine empirical datasets are converted to simple, unweighted, undirected graphs. Directed edges are symmetrized, repeated/temporal interactions are aggregated through the simple-graph representation, self-loops are removed, the largest connected component is retained, and nodes are relabeled to consecutive integer IDs starting from 0.

Dataset paths and source notes are documented in `docs/DATASETS.md`.

## Maximal-clique enumeration

The released workflow uses **exact maximal-clique enumeration**. The concrete backend is documented rather than hidden behind a more specific algorithm claim:

- empirical ranking/cache generation: `python-igraph` `Graph.maximal_cliques()`;
- method-library fallback when igraph is unavailable: NetworkX `find_cliques()`;
- average-degree and clustering stress scripts: NetworkX exact enumeration by default.

These backends enumerate maximal cliques exactly; the repository does not claim that every experiment uses the same internal enumeration implementation.

## Baseline settings and deterministic VoteRank ties

Exact fixed settings are recorded in `configs/revision_parameters.json` and Supplementary Table S3. In particular:

- CI shell radius: `2`;
- VoteRank initial voting ability: `1`;
- VoteRank neighbor decrement: `1/<k>`;
- VoteRank selected-node voting ability: `0`;
- VoteRank **internal voting-score ties: ascending node ID**;
- SNIM minimum maximal-clique size: `3`;
- CHBC: Louvain, resolution `1`, random seed `42`.

The VoteRank tie rule is now explicit in `msh_methods.calculate_voterank()`. An audit confirmed that this explicit rule reproduces the complete VoteRank ordering produced by the uploaded revision implementation on all nine empirical networks; see `docs/VOTERANK_TIE_AUDIT.md`.

For ordinary static score rankings, top-k ties are resolved by ascending node ID after relabeling. Monotonicity and Kendall `tau_b` analyses intentionally retain score ties and do not apply ID tie-breaking.

## Statistical unit

SIR experiments use 50 Monte Carlo blocks with 20 stochastic realizations per block. The **mean of the 20 realizations within one block is one statistical observation**, so inferential tests use `n = 50` paired block means rather than 1000 individual simulations.

For a fixed network and experimental setting, the network topology and ranked node set are fixed. Only stochastic SIR realizations vary across blocks and repetitions. Paired methods use a method-independent pseudo-random seed schedule (common random-number initialization).

See `docs/STATISTICAL_PROTOCOL.md` for confidence intervals, Wilcoxon tests, rank-biserial effect sizes, Friedman testing, Kendall `tau_b`, and BH correction families.

## Reproduce rankings and manuscript analyses

Precompute ranking/clique caches:

```bash
python precompute_rankings.py --force
```

Run individual analyses using the commands listed in `docs/figure_table_manifest.csv`, or inspect the complete ordered plan with:

```bash
python reproduce_all.py --list
python reproduce_all.py --dry-run
```

A full sequential reproduction can be started with:

```bash
python reproduce_all.py
```

The full run is computationally expensive. `reproduce_all.py` also supports `--only`, `--from-step`, `--through-step`, and `--continue-on-error`; see `python reproduce_all.py --help`.

## Matched-control reproducibility

`experiments/exp10_matched_controls_fig10_tablesS6S7.py` implements the reviewer-requested protocol:

- 10 degree-quantile bins;
- up to 1000 unique initial degree-matched candidates;
- DMR: 10 randomly selected candidates;
- DMD: one maximum-`L_s` candidate from the initial pool;
- DDMR: relative `L_s` error `<= 5%`;
- adaptive DDMR search up to 10000 unique candidates;
- if only 1-9 valid DDMR controls are found, all are retained;
- zero valid controls are reported as unmatched with no out-of-caliper fallback;
- multiple DMR/DDMR controls are averaged within each Monte Carlo block, preserving the block as the inferential unit;
- `L_s` is exact for up to 6000 unordered node pairs and uses up to 6000 sampled pairs for larger selected sets.

The supplied regression references support the manuscript summaries: valid DDMR controls in `79/90` settings, and larger MSH final spread in `85/90` DMR, `75/90` DMD, and `74/79` valid DDMR comparisons.

## Recovery-rate robustness

Supplementary Table S5 uses the recovery-rate robustness workflow retained from the first revision. `experiments/exp16_recovery_rate_tableS5.py` varies `gamma = 0.5, 0.75, 1.0` and adjusts `beta` to keep the effective edge transmissibility equal to `2.5 * beta_th`, exactly as described in the current Supplementary Material.

## Reference outputs

`reference_results/` contains the supplied final reference spreadsheets for Tables S6, S7, S10, and S11. They are included as regression references for the final audit; the experiment scripts are the source of the computational procedure.

## Reviewer-to-code map

`docs/REVIEWER_RESPONSE_MAPPING.md` maps each major second-round reviewer concern to the corresponding source files. `docs/figure_table_manifest.csv` maps each manuscript figure/table to the script that generates it.

## Immutable release

Before resubmission, freeze this repository, create a GitHub release, archive the exact release in Zenodo (or another immutable archive), and fill `REVISION_VERSION.txt` with the final Git commit, release tag, DOI, and date. The manuscript Availability statement should cite the same immutable version rather than only the moving repository homepage.
