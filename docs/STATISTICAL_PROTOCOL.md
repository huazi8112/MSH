# Statistical protocol

## Statistical unit

The fundamental SIR inferential unit is the Monte Carlo **block mean**. Each setting contains 50 blocks and 20 SIR realizations per block. The 20 realizations are averaged first; paired Wilcoxon tests, confidence intervals, and matched-pairs effect sizes operate on the resulting 50 block-level observations.

The topology and ranked node set are fixed within a network/setting. Only stochastic SIR realizations vary across blocks and repetitions.


## Deterministic ranking ties

For static score-based top-k selection, tied scores are resolved by ascending node ID after graph relabeling. VoteRank additionally applies ascending node ID **inside each iterative selection step** when multiple candidates have the same voting score. Monotonicity and Kendall `tau_b` intentionally retain score ties rather than applying this deterministic secondary rule.

## Confidence intervals

Unless otherwise noted, SIR error bars are two-sided 95% Student-t confidence intervals calculated from the 50 block means.

Synthetic runtime stress tests report 95% confidence intervals across five independent graph instances.

## Paired effect size

Matched-pairs rank-biserial correlation is defined from the signed ranks of paired non-zero differences:

`r_rb = (W_plus - W_minus) / (W_plus + W_minus)`.

Positive values favor MSH when MSH is the first member of the comparison.

## Multiple-testing families

The code implements BH correction within explicitly defined analysis families:

- Fig. 3 seed-ratio SIR: within each network, the 10 MSH-vs-best-baseline seed-ratio tests form one family.
- Table 3 / temporal final-time test: the nine network-level final-time comparisons form one family.
- Fig. 5 fixed-beta robustness: all network-beta MSH-vs-best-baseline tests produced by the fixed-beta script are corrected together.
- Table S4 cross-network post-hoc analysis: the 11 MSH-vs-baseline comparisons form one family.
- Table S7 matched controls: within each network and control strategy, the available node-selection-ratio tests form one family (normally 10; fewer for unmatched DDMR settings).
- Table S8 ablation: for each fixed seed ratio, the seven MSH-vs-variant tests form one family.
- Table S5 recovery robustness: the recovery-rate script applies BH correction to its final MSH-vs-best-baseline comparison table.

## Cross-network analysis

For Table S4, the final infection scale of each method is first averaged over the 10 seed-ratio settings within each network. The nine networks are then treated as independent blocks in the Friedman test. Post-hoc paired Wilcoxon tests use nine paired network-level aggregate values, with BH correction across the 11 MSH-vs-baseline comparisons.

## Kendall rank correlations

All-node Kendall `tau_b` is computed from raw method scores. Ties are retained and handled by `tau_b`; node ID is used only to align node vectors, not to break ties. Score definitions follow the same priority direction: larger values indicate higher ranking priority.
