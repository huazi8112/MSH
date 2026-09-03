# Final second-round code audit

## Scope

This repository was assembled from the first-revision GitHub style and the code used for the second-round revision. The audit focused on the items explicitly raised by the reviewer and on agreement between the manuscript, Supplementary Material, released code, and supplied final reference tables.

## Resolved alignment decisions

### Recovery-rate robustness (Supplementary Table S5)

The author confirmed that the current Table S5 uses the first-revision recovery-rate robustness script/protocol. The retained file is:

`experiments/exp16_recovery_rate_tableS5.py`

It is byte-for-byte identical to the first-revision `exp14_recovery_rate_tableS3.py` supplied for this audit (SHA256 `9ec9e643fc0fbef6337b52d3e9e169e4aaa9aa03c70447759f8c9184fe649d8a`).

### Maximal-clique enumeration

The final repository documents the actual implementation by workflow:

- empirical ranking/cache generation: python-igraph exact `Graph.maximal_cliques()`;
- method library: python-igraph with NetworkX `find_cliques()` exact fallback;
- average-degree/clustering stress scripts: NetworkX exact enumeration by default.

The repository therefore uses the scientifically accurate, backend-neutral description **exact maximal-clique enumeration** rather than claiming that every experiment uses one specific backend implementation.

### VoteRank internal ties

Supplementary Table S3 specifies ascending-node-ID internal tie-breaking. The final `calculate_voterank()` now explicitly applies this rule at every iterative voting-score tie.

A nine-network migration audit shows that this explicit cleanup does not change the complete VoteRank ordering, the top-10 set, or any top-1% through top-10% set relative to the uploaded revision implementation. See `docs/VOTERANK_TIE_AUDIT.md` and `docs/voterank_tie_audit.csv`.

## Reviewer-facing analyses present

- seed-ratio collective SIR and network-wise paired statistics;
- nine-network Friedman/average-rank/post-hoc analysis;
- fixed-k temporal SIR with block-level inference and effect sizes;
- infection-parameter and recovery-parameter robustness;
- monotonicity/ranking-resolution analysis with ties retained;
- ranking-frequency visualization;
- topological dispersion `L_s`;
- neighborhood-overlap `J_s`;
- matched controls DMR/DMD/DDMR with complete matching diagnostics and block-level aggregation;
- all-node Kendall `tau_b` against DC, CP, SH, ISH, and SNIM;
- seven ablation variants with raw values for large relative improvements;
- network-size, average-degree, and clustering stress tests.

## Matched-control reference regression

The supplied final reference spreadsheets pass the following checks:

- Table S6: 90 network-ratio settings;
- DDMR: 75 settings reach 10 controls, 4 settings retain 1-9 valid controls, 11 unmatched, giving 79/90 valid settings;
- retained DDMR maximum relative `L_s` error never exceeds 5%;
- Table S7: MSH final spread is larger in 85/90 DMR, 75/90 DMD, and 74/79 valid DDMR comparisons;
- Table S10: 18 model-degree settings (ER/WS/BA × six target degrees);
- Table S11: seven rewiring/clustering settings.

These checks can be rerun with:

`python -m tools.validate_reference_results`

## Statistical reproducibility

The main SIR inferential unit is one Monte Carlo block mean (20 realizations averaged first). Inference therefore uses 50 paired observations rather than 1000 raw realizations. Analysis-specific BH correction families and effect-size definitions are documented in `docs/STATISTICAL_PROTOCOL.md`.

## Remaining release-only task

No unresolved scientific/code-alignment blocker remains in this release candidate. The only mandatory final step is creation of the immutable public release: freeze outputs, commit to GitHub, create a release tag, archive that exact version, obtain a DOI, and fill `REVISION_VERSION.txt` before resubmission.
