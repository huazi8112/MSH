# Changelog

## Second-round revision release candidate

Major changes relative to the first-revision repository:

- Reframed public repository terminology around **redundancy-aware node ranking** and collective evaluation of ranking-derived node sets.
- Added all-node Kendall `tau_b` correlations against DC, clique participation, SH, ISH, and SNIM.
- Added nine-network Friedman/post-hoc aggregate comparison.
- Expanded matched-control implementation and diagnostics (DMR, DMD, DDMR) with explicit 5% `L_s` caliper, adaptive candidate search, partial-match handling, no out-of-caliper fallback, and block-level control aggregation.
- Added raw ablation final-spread outputs for large relative improvements.
- Added average-degree and clustering stress-test tables and runtime decomposition.
- Retained and documented the first-revision recovery-rate robustness workflow as current Supplementary Table S5.
- Made VoteRank internal voting-score ties explicitly deterministic by ascending node ID, matching Supplementary Table S3.
- Documented exact maximal-clique enumeration backends rather than claiming one backend for all workflows.
- Added reviewer-to-code mapping, statistical protocol, release checklist, validation, tests, checksum tooling, and a full reproduction driver.
