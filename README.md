# Mesoscopic Structural Holes (MSH)

This repository contains the code and data used for the revised manuscript:

**Finding Influential Nodes via Clique-Aware Mesoscopic Structural Holes in Complex Networks**

MSH is a topology-based node-ranking method for simple, unweighted, undirected **pairwise graphs**. It uses maximal cliques inferred from pairwise adjacency as mesoscopic cohesive units. The repository follows the revised manuscript terminology and does not treat inferred cliques as directly observed hyperedges, simplices, or collective events.

## Workflow

![MSH workflow](images/workflow_msh.png)

The workflow consists of three parts:

1. **Clique-aware scoring.** Maximal cliques are enumerated from the processed pairwise graph. For each node, MSH evaluates clique-level effective dependence and inter-clique redundancy. A smaller MSH constraint coefficient indicates a stronger mesoscopic structural-hole position.
2. **Influential-node evaluation.** The resulting rankings are evaluated by SIR spreading, node-overlap analysis, ranking resolution, seed-set dispersion, and ablation analysis.
3. **Computational evaluation.** Runtime experiments examine network-size effects, average-degree stress, clustering-related stress, and clique-structure indicators.

## Repository layout

```text
MSH_revised_repository/
├── msh_methods.py                  # MSH, baselines, and ablation variants
├── network_loader.py               # Dataset loading and preprocessing
├── precompute_rankings.py          # Ranking-cache generation
├── configs/
│   └── revision_parameters.json    # Main parameters used in the revision
├── docs/
│   ├── DATASETS.md                 # Dataset sources and preprocessing notes
│   ├── REPRODUCIBILITY.md          # Execution order and reproducibility notes
│   └── figure_table_manifest.csv   # Figure/table-to-script mapping
├── experiments/                    # Scripts for manuscript figures and tables
├── images/
│   └── workflow_msh.png            # Workflow figure shown above
├── networks_data/                  # Local network files
├── results/                        # Generated outputs and caches
└── tools/                          # Preprocessing and structural-statistics utilities
```

`hosh_methods.py` is kept only as a backward-compatible wrapper for older script imports. Public-facing descriptions use **MSH**.

## Installation

Python 3.10 or later is recommended.

```bash
pip install -r requirements.txt
```

The main dependencies are `networkx`, `numpy`, `pandas`, `scipy`, `matplotlib`, `tqdm`, `openpyxl`, `python-igraph`, and `python-louvain`.

## Reproducing the revised results

Run all commands from the repository root.

### 1. Check preprocessing and generate rankings

```bash
python -m tools.evaluate_preprocessing_original_vs_processed
python precompute_rankings.py --force
```

### 2. Reproduce the main manuscript results

```bash
python -m experiments.exp01_top10_node_overlap_tables3_5
python -m experiments.exp02_sir_seed_ratio_fig3
python -m experiments.exp03_temporal_sir_fig4_table6
python -m experiments.exp04_beta_robustness_fig5
python -m experiments.exp05_monotonicity_table7
python -m experiments.exp06_ranking_frequency_fig6
python -m experiments.exp07_seed_dispersion_fig7
python -m experiments.exp08_topology_visualization_fig8
python -m experiments.exp09_matched_controls_fig9
python -m experiments.exp10_ablation_table8
python -m experiments.exp11_runtime_scale_fig10
python -m experiments.exp12_average_degree_stress_fig11_tableS4
```

### 3. Reproduce supplementary analyses

```bash
python -m experiments.exp13_tie_breaking_tableS2
python -m experiments.exp14_recovery_rate_tableS3
python -m experiments.exp15_clustering_stress_tableS5
```

The complete figure/table mapping is provided in `docs/figure_table_manifest.csv`.

## Data preprocessing

All empirical networks are converted to simple, unweighted, undirected graphs before ranking and SIR evaluation. The preprocessing pipeline is:

1. convert directed edges to undirected edges;
2. aggregate repeated or temporal interactions into static unweighted edges;
3. remove self-loops;
4. extract the largest connected component;
5. relabel nodes to consecutive integer IDs.

Dataset descriptions and source notes are provided in `docs/DATASETS.md`.

## Experimental settings

The revision parameters are recorded in `configs/revision_parameters.json`. The main settings are:

- SIR model: synchronous discrete-time network SIR;
- main recovery probability: `gamma = 1`;
- main infection probability: `beta = 2.5 * beta_th`;
- seed ratios: `1%` to `10%`;
- temporal SIR experiment: `k = 10` seeds;
- SIR repetitions: 50 random-seed blocks × 20 simulations per block;
- statistical test: paired Wilcoxon signed-rank test with Benjamini-Hochberg correction;
- main tie-breaking: ascending node ID after relabeling;
- monotonicity: identical scores are treated as tied groups.

## Computational notes

Maximal cliques are enumerated using the exact Bron-Kerbosch algorithm with pivoting and degeneracy ordering. Runtime analyses report clique-enumeration time, score-computation time, confidence intervals over independently generated synthetic instances, and clique-structure indicators. Separate peak-memory profiling is not included, so no claim is made about memory efficiency or memory scalability.

## Outputs

Generated outputs are written to `results/`, including ranking caches, figures, and tables.

