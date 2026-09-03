# Mesoscopic Structural Holes (MSH)

Code and data for:

**Mesoscopic Structural Holes for Redundancy-Aware Node Ranking in Complex Networks**

MSH is a topology-based node-ranking method for simple, unweighted, undirected networks. It treats maximal cliques as mesoscopic cohesive units and combines effective dependence with inter-clique redundancy.

The method defines a constraint coefficient `C_i`, where a smaller value indicates a stronger mesoscopic structural-hole position. The implementation uses the equivalent priority score `1 - C_i`, so a larger score indicates higher ranking priority.

## Installation

Python 3.10 or later is recommended.

```bash
python -m pip install -r requirements.txt
```

## Repository structure

```text
MSH/
├── README.md
├── requirements.txt
├── reproduce_all.py
├── msh_methods.py
├── network_loader.py
├── precompute_rankings.py
├── configs/
├── experiments/
├── tools/
├── tests/
├── docs/
├── networks_data/
├── reference_results/
└── results/
```

- `msh_methods.py`: MSH and baseline ranking methods.
- `precompute_rankings.py`: ranking and maximal-clique cache generation.
- `experiments/`: scripts for the analyses reported in the paper.
- `configs/`: fixed experimental parameters.
- `networks_data/`: empirical network data.
- `reference_results/`: reference supplementary tables used for validation.
- `docs/`: dataset, implementation, and reproducibility details.

## Reproduction

Validate the repository:

```bash
python -m tools.validate_repository
python -m tools.validate_reference_results
pytest -q
```

Precompute rankings:

```bash
python precompute_rankings.py --force
```

List the available reproduction steps:

```bash
python reproduce_all.py --list
```

Run all analyses sequentially:

```bash
python reproduce_all.py
```

Individual experiment scripts can also be run directly from the `experiments/` directory.

## Implementation notes

All empirical networks are converted to simple, unweighted, undirected graphs before analysis. Exact maximal-clique enumeration is used in the workflow. Fixed method parameters and random-seed settings are provided in `configs/`, and additional implementation details are documented in `docs/REPRODUCIBILITY.md`.
