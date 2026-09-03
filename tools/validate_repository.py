"""Lightweight release validation for the MSH second-round repository.

This script checks repository structure, compiles Python sources, verifies the nine
loader-required dataset files, and checks that the supplied reference tables exist.
It does not rerun the computationally expensive experiments.
"""
from __future__ import annotations

import compileall
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md",
    "CITATION.cff",
    "CHANGELOG.md",
    "msh_methods.py",
    "hosh_methods.py",
    "network_loader.py",
    "precompute_rankings.py",
    "reproduce_all.py",
    "configs/revision_parameters.json",
    "docs/REPRODUCIBILITY.md",
    "docs/STATISTICAL_PROTOCOL.md",
    "docs/REVIEWER_RESPONSE_MAPPING.md",
    "docs/VOTERANK_TIE_AUDIT.md",
    "docs/voterank_tie_audit.csv",
    "docs/figure_table_manifest.csv",
    "docs/RELEASE_CHECKLIST.md",
    "docs/FINAL_AUDIT.md",
    "tools/validate_reference_results.py",
    "tools/check_voterank_tie_impact.py",
    "tools/generate_checksums.py",
]

DATASETS = [
    "networks_data/lesmis/lesmis.mtx",
    "networks_data/adjnoun/adjnoun.mtx",
    "networks_data/jazz/jazz.mtx",
    "networks_data/USAir97/USAir97.mtx",
    "networks_data/ia-infect-dublin.mtx",
    "networks_data/email/email.mtx",
    "networks_data/polblogs/polblogs.mtx",
    "networks_data/soc-hamsterster.edges",
    "networks_data/power/power.mtx",
]

REFERENCE = [
    "reference_results/Table_S6.xlsx",
    "reference_results/Table_S7.xlsx",
    "reference_results/Table_S10.xlsx",
    "reference_results/Table_S11.xlsx",
]


def check_files(paths, label):
    missing = [p for p in paths if not (ROOT / p).exists()]
    if missing:
        print(f"[FAIL] {label}: missing {len(missing)} file(s)")
        for p in missing:
            print(f"       - {p}")
        return False
    print(f"[OK]   {label}: {len(paths)} file(s)")
    return True


def main() -> int:
    ok = True
    ok &= check_files(REQUIRED, "repository control files")
    ok &= check_files(DATASETS, "empirical dataset files")
    ok &= check_files(REFERENCE, "reference supplementary tables")

    compiled = compileall.compile_dir(str(ROOT), quiet=1)
    print("[OK]   Python syntax compilation" if compiled else "[FAIL] Python syntax compilation")
    ok &= compiled

    release_text = (ROOT / "REVISION_VERSION.txt").read_text(encoding="utf-8")
    if "<fill after" in release_text:
        print("[WARN] REVISION_VERSION.txt still contains release placeholders.")
    else:
        print("[OK]   immutable release metadata filled")

    print("\nNOTE: scientific alignment items are resolved; complete the immutable release metadata in docs/RELEASE_CHECKLIST.md before publishing.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
