# Reviewer-request-to-code mapping

This file maps the main second-round reviewer concerns to the released implementation.

| Reviewer concern | Repository support |
|---|---|
| Reframe contribution away from isolated-node SIR superiority | README/public documentation define MSH as redundancy-aware node ranking; SIR evaluates ranking-derived node sets collectively. |
| Complete all-node structural correlations | `experiments/exp11_kendall_correlation_fig11.py` + `experiments/plot_fig11_kendall.py`; MSH vs DC, CP, SH, ISH, SNIM; all nodes; Kendall `tau_b`; score ties retained. |
| Clarify Monte Carlo blocks and inferential unit | `docs/STATISTICAL_PROTOCOL.md`; 20 SIR realizations are averaged into one block mean; inference uses 50 paired block means. |
| Clarify what varies across blocks | SIR documentation/scripts: topology and ranked node sets are fixed within a setting; only stochastic SIR realizations vary. |
| Common random-number pairing | SIR scripts use a method-independent pseudo-random seed schedule for corresponding network/setting/block/repetition indices. |
| Aggregate across-network comparison | `experiments/exp02_cross_network_stats_tableS4.py`; nine-network Friedman test, average ranks, post-hoc paired Wilcoxon, BH adjustment, rank-biserial effect size. |
| Practical significance and effect sizes | Temporal and matched-control scripts export raw differences, 95% CIs, adjusted p-values, and `r_rb`. |
| Matched-control implementation details | `experiments/exp10_matched_controls_fig10_tablesS6S7.py`; 10 degree bins, 1000 initial candidates, DMR=10, DMD=max-`L_s`, DDMR 5% caliper, adaptive search to 10000, partial matches retained, no out-of-caliper fallback, block-level aggregation. |
| Raw values behind large ablation percentages | `experiments/exp12_ablation_table5_tablesS8S9.py` exports summary improvements and raw final infection scales. |
| Exact baseline settings | `configs/revision_parameters.json`, Supplementary Table S3, and deterministic VoteRank implementation in `msh_methods.py`. |
| VoteRank internal tie reproducibility | Ascending-node-ID tie rule is explicit in `calculate_voterank()`; migration audit is in `docs/VOTERANK_TIE_AUDIT.md`. |
| Tie-breaking sensitivity | `experiments/exp15_tie_breaking_tableS2.py`. |
| Recovery robustness | `experiments/exp16_recovery_rate_tableS5.py`; confirmed first-revision script/protocol used for current Table S5. |
| Scalability and clique complexity | `experiments/exp13_runtime_scale_fig12.py`, `exp14_average_degree_stress_fig13_tableS10.py`, and `exp17_clustering_stress_tableS11.py`; five independent instances and clique-structure indicators. |
| Exact maximal-clique implementation transparency | README and `docs/REPRODUCIBILITY.md` report exact enumeration and the concrete igraph/NetworkX backend per workflow. |
| Immutable code version | `REVISION_VERSION.txt` and `docs/RELEASE_CHECKLIST.md`; final commit/tag/DOI must be filled after repository freeze and archival. |
