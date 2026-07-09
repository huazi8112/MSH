"""
Appendix A preprocessing-impact evaluation.

This script generates an appendix-ready comparison between the original network
representation recorded by `network_loader.py` and the final processed graph used
in all ranking methods and SIR simulations.

Outputs:
  results/appendix_preprocessing/Appendix_A1_Preprocessing_Effect.xlsx
  results/appendix_preprocessing/Appendix_A1_Preprocessing_Effect.csv
  results/appendix_preprocessing/Appendix_A1_Preprocessing_Effect.md
  results/appendix_preprocessing/Appendix_A1_Preprocessing_Effect.tex

Expected network_loader metadata fields:
  Raw_N, LCC_N, Node_Loss_%, Raw_kmax, LCC_kmax,
  Raw_C, LCC_C, Delta_C, Raw_Beta, LCC_Beta, Delta_Beta_th

In the exported table, Raw_* is renamed as Original_* and LCC_* is renamed as
Processed_*, so the table can be described in the manuscript as a comparison
between the original network and the preprocessed network.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Any

import pandas as pd

from network_loader import download_and_load_graph, get_network_list


def parse_networks(networks_arg: str | None) -> List[str]:
    if networks_arg is None or networks_arg.strip().lower() in {"", "all"}:
        return get_network_list()
    return [x.strip().lower() for x in networks_arg.split(",") if x.strip()]


def collect_preprocessing_records(networks: List[str], verbose: bool = True) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    for idx, net in enumerate(networks, 1):
        print(f"\n[{idx}/{len(networks)}] Processing network: {net}")
        g = download_and_load_graph(net, verbose=verbose)
        if g is None:
            print(f"  [Warning] Failed to load {net}; skipped.")
            continue

        meta = g.graph.get("preprocess_meta", None)
        if meta is None:
            raise RuntimeError(
                "The loaded graph does not contain `preprocess_meta`. "
                "Please use the revised network_loader.py that records preprocessing metadata."
            )

        # Rename metadata fields into appendix-friendly terms.
        rec = {
            "Network": meta.get("Network", net),
            "Original_N": meta.get("Raw_N"),
            "Processed_N": meta.get("LCC_N"),
            "Node_Loss_%": meta.get("Node_Loss_%"),
            "Original_E": meta.get("Raw_E"),
            "Processed_E": meta.get("LCC_E"),
            "Edge_Loss_%": meta.get("Edge_Loss_%"),
            "Original_kmax": meta.get("Raw_kmax"),
            "Processed_kmax": meta.get("LCC_kmax"),
            "Original_C": meta.get("Raw_C"),
            "Processed_C": meta.get("LCC_C"),
            "Delta_C": meta.get("Delta_C"),
            "Original_Beta_th": meta.get("Raw_Beta"),
            "Processed_Beta_th": meta.get("LCC_Beta"),
            "Delta_Beta_th": meta.get("Delta_Beta_th"),
        }
        records.append(rec)
        print(f"  [OK] {net}: Original_N={rec['Original_N']}, Processed_N={rec['Processed_N']}, "
              f"Original_beta={rec['Original_Beta_th']}, Processed_beta={rec['Processed_Beta_th']}")

    if not records:
        raise RuntimeError("No valid network records were collected.")

    return pd.DataFrame(records)


def format_for_appendix(df: pd.DataFrame, include_edges: bool, include_clustering: bool) -> pd.DataFrame:
    """Create a compact appendix table with stable column order."""
    columns = [
        "Network",
        "Original_N",
        "Processed_N",
        "Node_Loss_%",
    ]

    if include_edges:
        columns += ["Original_E", "Processed_E", "Edge_Loss_%"]

    columns += ["Original_kmax", "Processed_kmax"]

    if include_clustering:
        columns += ["Original_C", "Processed_C", "Delta_C"]

    columns += ["Original_Beta_th", "Processed_Beta_th", "Delta_Beta_th"]

    out = df[columns].copy()

    # Numeric formatting / rounding for appendix readability.
    for col in out.columns:
        if col == "Network":
            continue
        if col.endswith("_N") or col.endswith("_E") or col.endswith("kmax"):
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
        elif col in {"Node_Loss_%", "Edge_Loss_%"}:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(4)

    return out


def make_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Export a LaTeX table with math-style headers and an explanatory note."""
    header_map = {
        "Network": "Network",
        "Original_N": r"Original $N$",
        "Processed_N": r"Processed $N$",
        "Node_Loss_%": r"Node loss (\\%)",
        "Original_E": r"Original $|E|$",
        "Processed_E": r"Processed $|E|$",
        "Edge_Loss_%": r"Edge loss (\\%)",
        "Original_kmax": r"Original $k_{\\max}$",
        "Processed_kmax": r"Processed $k_{\\max}$",
        "Original_C": r"Original $C$",
        "Processed_C": r"Processed $C$",
        "Delta_C": r"$\\Delta C$",
        "Original_Beta_th": r"Original $\\beta_{th}$",
        "Processed_Beta_th": r"Processed $\\beta_{th}$",
        "Delta_Beta_th": r"$\\Delta\\beta_{th}$",
    }

    df_latex = df.rename(columns=header_map)
    latex = df_latex.to_latex(index=False, escape=False, float_format="%.4f")

    note = (
        r"\\textit{Note:} Original statistics are computed for the network before the final "
        r"standardized preprocessing output, whereas processed statistics are computed for "
        r"the simple, unweighted, undirected largest connected component used in all ranking "
        r"methods and SIR simulations. $N$ denotes the number of nodes, $|E|$ denotes the "
        r"number of edges, $k_{\\max}$ denotes the maximum degree, $C$ denotes the average "
        r"clustering coefficient, and $\\beta_{th}$ denotes the degree-moment epidemic-threshold "
        r"estimate. $\\Delta\\beta_{th}=|\\beta_{th}^{\\mathrm{Original}}-\\beta_{th}^{\\mathrm{Processed}}|$."
    )

    latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        + latex
        + "\\vspace{1mm}\n"
        + "\\begin{minipage}{0.98\\linewidth}\n"
        + "\\footnotesize " + note + "\n"
        + "\\end{minipage}\n"
        + "\\end{table}\n"
    )
    return latex


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Appendix A preprocessing comparison table: Original vs Processed networks."
    )
    parser.add_argument("--networks", type=str, default="all",
                        help="Comma-separated network names, or 'all' for the default 9 networks.")
    parser.add_argument("--output-dir", type=str, default="results/appendix_preprocessing",
                        help="Directory for output files.")
    parser.add_argument("--include-edges", action="store_true",
                        help="Include Original/Processed edge counts and edge loss in the compact table.")
    parser.add_argument("--include-clustering", action="store_true",
                        help="Include Original/Processed average clustering coefficients and Delta_C.")
    parser.add_argument("--quiet", action="store_true", help="Reduce network loading messages.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    networks = parse_networks(args.networks)

    print("=" * 80)
    print("Appendix A: preprocessing effect evaluation")
    print("Comparison: Original network statistics vs processed graph statistics")
    print("=" * 80)
    print(f"Networks: {networks}")

    raw_df = collect_preprocessing_records(networks, verbose=not args.quiet)
    appendix_df = format_for_appendix(
        raw_df,
        include_edges=args.include_edges,
        include_clustering=args.include_clustering,
    )

    # Save outputs.
    xlsx_path = os.path.join(args.output_dir, "Appendix_A1_Preprocessing_Effect.xlsx")
    csv_path = os.path.join(args.output_dir, "Appendix_A1_Preprocessing_Effect.csv")
    md_path = os.path.join(args.output_dir, "Appendix_A1_Preprocessing_Effect.md")
    tex_path = os.path.join(args.output_dir, "Appendix_A1_Preprocessing_Effect.tex")

    notes = pd.DataFrame({
        "Item": [
            "Purpose",
            "Original statistics",
            "Processed statistics",
            "Delta beta definition",
            "Recommended appendix caption",
        ],
        "Description": [
            "Evaluate the effect of standardized preprocessing on network-level statistics.",
            "Statistics before the final processed graph used by the experiments, based on metadata recorded by network_loader.py.",
            "Statistics of the simple, unweighted, undirected largest connected component used in all ranking methods and SIR simulations.",
            "Delta_Beta_th = abs(Original_Beta_th - Processed_Beta_th).",
            "Table A1. Effect of preprocessing on network-level structural statistics and epidemic-threshold estimates.",
        ]
    })

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        appendix_df.to_excel(writer, sheet_name="Appendix_A1", index=False)
        raw_df.to_excel(writer, sheet_name="Full_Metadata", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)

    appendix_df.to_csv(csv_path, index=False)
    appendix_df.to_markdown(md_path, index=False)

    latex_caption = "Effect of preprocessing on network-level structural statistics and epidemic-threshold estimates."
    latex = make_latex_table(appendix_df, caption=latex_caption, label="tab:appendix_preprocessing")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex)

    print("\n" + "=" * 30 + " Appendix-ready table " + "=" * 30)
    print(appendix_df.to_markdown(index=False))
    print("\n[Output]")
    print(f"  Excel:    {xlsx_path}")
    print(f"  CSV:      {csv_path}")
    print(f"  Markdown: {md_path}")
    print(f"  LaTeX:    {tex_path}")


if __name__ == "__main__":
    main()
