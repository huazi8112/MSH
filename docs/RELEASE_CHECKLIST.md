# Final release checklist

The scientific/code-alignment decisions requested during the final audit are resolved:

- **Recovery-rate robustness:** confirmed to use the first-revision recovery-rate script/protocol; retained as `experiments/exp16_recovery_rate_tableS5.py`.
- **Maximal-clique wording:** resolved by documenting exact maximal-clique enumeration and the concrete backend used by each workflow, rather than claiming one backend for every experiment.
- **VoteRank internal ties:** explicitly resolved by ascending node ID at every iterative voting-score tie. An audit found the full VoteRank ordering unchanged on all nine empirical networks relative to the uploaded revision implementation.

## Required before GitHub/Zenodo publication

1. Run `python -m tools.validate_repository`.
2. Run `python -m tools.validate_reference_results`.
3. Run `pytest -q`.
4. Run or verify the final manuscript reproduction set and compare regenerated outputs with the frozen manuscript/supplement.
5. Confirm exact package versions used for the final rerun and, if desired, replace version ranges in `requirements.txt` with a frozen environment file.
6. Generate final SHA256 files with `python -m tools.generate_checksums` after all source edits are complete.
7. Commit the frozen repository to GitHub.
8. Create the final GitHub release tag.
9. Archive that exact release in Zenodo (or an equivalent immutable archive) and obtain a DOI.
10. Fill `REVISION_VERSION.txt` with the exact Git commit, release tag, DOI, and release date.
11. Update the manuscript Availability statement to cite the immutable release/commit.

Do not change numerical algorithms after the final regression check unless the affected analyses are rerun and the manuscript is updated accordingly.
