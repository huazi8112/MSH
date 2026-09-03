"""Validate the supplied final reference Tables S6/S7/S10/S11.

This is a lightweight regression check used before the immutable release. It
verifies the manuscript-level matched-control counts and the expected dimensions
of the two synthetic stress-test tables.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference_results"


def main() -> int:
    s6 = pd.read_excel(REF / "Table_S6.xlsx")
    s7 = pd.read_excel(REF / "Table_S7.xlsx")
    s10 = pd.read_excel(REF / "Table_S10.xlsx")
    s11 = pd.read_excel(REF / "Table_S11.xlsx")

    assert len(s6) == 90, f"Table S6 should have 90 settings, found {len(s6)}"
    valid_ddmr = int((s6["DDMR_Selected_Count"] > 0).sum())
    assert valid_ddmr == 79, f"Expected 79 valid DDMR settings, found {valid_ddmr}"
    assert int((s6["DDMR_Selected_Count"] == 10).sum()) == 75
    assert int(((s6["DDMR_Selected_Count"] > 0) & (s6["DDMR_Selected_Count"] < 10)).sum()) == 4
    assert int((s6["DDMR_Selected_Count"] == 0).sum()) == 11
    valid_err = s6.loc[s6["DDMR_Selected_Count"] > 0, "DDMR_Selected_Max_Relative_Ls_Error(%)"].dropna()
    assert bool((valid_err <= 5.0 + 1e-12).all()), "A retained DDMR control exceeds the 5% L_s caliper"

    counts = {}
    for control in ("DMR", "DMD", "DDMR"):
        sub = s7[s7["Control"] == control]
        counts[control] = (int((sub["Delta_F(pp)"] > 0).sum()), len(sub))
    assert counts["DMR"] == (85, 90), counts
    assert counts["DMD"] == (75, 90), counts
    assert counts["DDMR"] == (74, 79), counts

    assert len(s10) == 18, f"Table S10 should have 18 model-degree settings, found {len(s10)}"
    assert set(s10["Model"].astype(str)) == {"ER", "WS", "BA"}
    assert set(s10["Target <k>"].astype(int)) == {4, 8, 12, 16, 20, 24}

    assert len(s11) == 7, f"Table S11 should have 7 rewiring settings, found {len(s11)}"

    print("[OK] Table S6: 90 settings; 79 valid DDMR = 75 target + 4 partial; 11 unmatched")
    print("[OK] Table S7: MSH higher F counts DMR 85/90, DMD 75/90, DDMR 74/79")
    print("[OK] Table S10: 18 ER/WS/BA average-degree stress settings")
    print("[OK] Table S11: 7 clustering/rewiring stress settings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
