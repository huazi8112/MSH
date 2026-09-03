# VoteRank internal tie-breaking audit

Supplementary Table S3 specifies **ascending node ID** for VoteRank's internal iterative tie-breaking. The final repository therefore makes this rule explicit in `msh_methods.calculate_voterank()`:

1. maximize the current voting score;
2. if several candidates have the same voting score, select the smallest node ID.

Because the empirical preprocessing pipeline relabels nodes to consecutive integers starting from 0, this rule is deterministic and unambiguous.

## Regression check against the uploaded revision implementation

The uploaded revision implementation used a Python `set` of candidate nodes and retained the first candidate encountered at the current maximum voting score. Before freezing the public code, we compared that implementation with the explicit ascending-ID implementation on all nine empirical networks.

Result: **the complete VoteRank ordering was identical on all nine networks**. Consequently, the top-10 set and every top-1% through top-10% set were also identical. The reproducibility cleanup therefore does not change the VoteRank node sets used by the manuscript experiments on these datasets.

The detailed audit is stored in `docs/voterank_tie_audit.csv` and can be regenerated with:

```bash
python -m tools.check_voterank_tie_impact
```

This audit is a code-migration/reproducibility check, not an additional manuscript hypothesis test.
