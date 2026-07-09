# Reproducibility notes

This document records the implementation choices requested during revision.

## Environment

- Python: 3.10+ recommended
- Operating system used in the manuscript experiments: Windows 11
- Hardware used in the manuscript experiments: Intel Core i5-10300H CPU 2.50 GHz, 16 GB RAM
- Implementation language: Python
- Dependencies: see `requirements.txt`

## Data preprocessing

All raw networks are transformed using the same pipeline in `network_loader.py`:

1. Load raw graph from Matrix Market, edge-list, or local file.
2. Convert directed graphs to undirected graphs.
3. Aggregate repeated, weighted, or temporal interactions to a single unweighted static edge.
4. Remove self-loops.
5. Extract the largest connected component.
6. Relabel nodes to consecutive integers starting from 0.

The script `tools/evaluate_preprocessing_original_vs_processed.py` reproduces the preprocessing comparison used for Table S1.

## Ranking cache

`precompute_rankings.py` computes and stores method scores under `results/node_rankings/` and maximal-clique caches under `results/clique_cache/`. Recompute caches after changing the method implementation:

```bash
python precompute_rankings.py --force
```

## SIR protocol

The experiments use a discrete stochastic network SIR process:

1. Initial seed nodes are infected; all other nodes are susceptible.
2. Updates are synchronous.
3. At each time step, infected nodes first attempt to infect susceptible neighbors independently with probability beta.
4. After infection attempts are evaluated, infected nodes recover with probability gamma.
5. Newly infected nodes become active spreaders only in the next time step.
6. Recovered nodes cannot infect others and cannot be reinfected.
7. The process stops when no infected nodes remain.

Main settings are recorded in `configs/revision_parameters.json`.

## Statistical settings

- SIR repetitions: 1000
- Random-seed blocks: 50
- Simulations per block: 20
- Error bars: 95% confidence intervals
- Paired tests: paired Wilcoxon signed-rank tests
- Multiple comparisons: Benjamini-Hochberg correction

## Tie-breaking

- Main seed selection: ascending node ID after relabeling.
- Tie-breaking sensitivity: deterministic node-ID, repeated random boundary tie-breaking, and SIR-based boundary tie-breaking.
- Monotonicity: no deterministic ID tie-breaking is applied; tied scores remain tied groups.

## Figure/table mapping

See `docs/figure_table_manifest.csv`. This file maps each revised manuscript figure/table to the script that generates it.

## Version and commit

This package is prepared as `MSH-revision-2026-07-08`. After pushing the repository to GitHub, create a release tag and record the final commit hash in `REVISION_VERSION.txt` before resubmission.
