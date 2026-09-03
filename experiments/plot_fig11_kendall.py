"""
Final Kendall tau-b rank-correlation lollipop matrix.

Purpose
-------
Visualize the all-node Kendall rank correlations between MSH and five related
structural measures across nine empirical networks:

    DC, CP, SH, ISH, SNIM

This is the final manuscript-facing plotting script.

Design
------
- One single 9 x 5 matrix figure
- Rows: nine networks
- Columns: DC, CP, SH, ISH, SNIM
- Each cell contains a miniature horizontal signed lollipop
- Local zero is represented by a short vertical tick
- Colored stem extends from zero to Kendall tau_b
- All observed markers are circles
- Exact tau_b values are printed above the markers
- Full four-sided black border around the matrix
- Very light internal row/column guides
- No heatmap fill
- No colorbar
- No local -1/0/1 scale hint
- No Table-2 context columns
- No top Kendall title
- Bottom labels use abbreviations only
- Style follows the manuscript SIR figures

Input
-----
results/comment2_kendall_rank_correlations/
    Table_C2_Kendall_by_network.csv

Required input columns
----------------------
Network
Degree
Clique participation
SH
ISH
SNIM

Output
------
results/comment2_kendall_rank_correlations/
    Fig_C2_MSH_Kendall_lollipop_matrix_manuscript.pdf
    Fig_C2_MSH_Kendall_lollipop_matrix_manuscript.png
    Table_C2_Kendall_manuscript_values.csv

Run
---
python plot_kendall_lollipop_matrix_manuscript.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==========================================
# 0. 路径与网络顺序
# ==========================================

DEFAULT_INPUT = Path(
    "results/comment2_kendall_rank_correlations/"
    "Table_C2_Kendall_by_network.csv"
)

DEFAULT_OUTPUT_DIR = Path(
    "results/comment2_kendall_rank_correlations"
)

NETWORK_ORDER = [
    "Lesmis",
    "Adjnoun",
    "Jazz",
    "Usair",
    "Infect",
    "Email",
    "Polblogs",
    "Hamster",
    "Power",
]

# 输入列名 -> 正文缩写
COMPARATORS = [
    ("Degree", "DC"),
    ("Clique participation", "CP"),
    ("SH", "SH"),
    ("ISH", "ISH"),
    ("SNIM", "SNIM"),
]


# ==========================================
# 1. 绘图风格：与 SIR 主实验保持一致
# ==========================================

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'axes.spines.top': True,
    'axes.spines.right': True,
})

# 与原 SIR 实验配色体系保持一致。
COLORS = {
    "DC": "#E5B25D",
    "CP": "#4FA3D1",   # 使用原 SIR 配色体系中的蓝色
    "SH": "#E2739F",
    "ISH": "#F08C3D",
    "SNIM": "#7F7F7F",
}

# 用户要求：全部统一为圆形 marker。
MARKER = "o"

TEXT_COLOR = "#000000"

# 极淡辅助线。
LOCAL_AXIS_COLOR = "#D0D0D0"
ZERO_COLOR = "#7F7F7F"
ROW_GUIDE_COLOR = "#EEEEEE"
COL_GUIDE_COLOR = "#F2F2F2"

# 每个矩阵单元内的局部 Kendall 轴。
LOCAL_HALF = 0.31

# 与 SIR 图接近的视觉线宽。
STEM_WIDTH = 1.45
MARKER_SIZE = 5.0
MARKER_EDGE_WIDTH = 0.5
ENDPOINT_SIZE = 2.6
VALUE_OFFSET_Y = 0.165

N_ROWS = len(NETWORK_ORDER)
N_COLS = len(COMPARATORS)


# ==========================================
# 2. 读取 Kendall 结果
# ==========================================

def load_kendall_results(
    csv_path: Path,
) -> pd.DataFrame:

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Kendall result file not found:\n{csv_path}\n\n"
            "Please run the Kendall correlation experiment first."
        )

    df = pd.read_csv(csv_path)

    required = (
        ["Network"]
        + [source for source, _ in COMPARATORS]
    )

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required Kendall columns: "
            + ", ".join(missing)
        )

    order_map = {
        network: idx
        for idx, network in enumerate(NETWORK_ORDER)
    }

    df = df[
        df["Network"].isin(NETWORK_ORDER)
    ].copy()

    df["_order"] = df["Network"].map(order_map)

    df = (
        df.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    return df


# ==========================================
# 3. 数值格式与局部坐标
# ==========================================

def fmt_tau(value: float) -> str:
    """Use typographic minus sign for negative Kendall coefficients."""

    if not np.isfinite(value):
        return "NA"

    if value < 0:
        return "−" + f"{abs(value):.2f}"

    return f"{value:.2f}"


def local_x(
    cell_center: float,
    tau: float,
) -> float:
    """
    Map Kendall tau_b in [-1, 1] to the local horizontal axis of one cell.
    """
    return (
        cell_center
        + LOCAL_HALF * float(tau)
    )


# ==========================================
# 4. 最终正文矩阵棒棒糖图
# ==========================================

def plot_matrix_lollipop(
    df: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
):

    # ------------------------------------------
    # A. 热力图式矩阵位置
    # ------------------------------------------
    x_centers = np.arange(
        N_COLS,
        dtype=float,
    )

    y_centers = np.arange(
        N_ROWS,
        dtype=float,
    )

    # 矩阵边界。
    matrix_left = -0.50
    matrix_right = N_COLS - 0.50
    matrix_top = -0.50
    matrix_bottom = N_ROWS - 0.50

    # 删除右侧 Table 2 后，将矩阵适当加宽，
    # 使每个局部棒棒糖更容易读取。
    fig, ax = plt.subplots(
        figsize=(7.8, 5.4)
    )

    row_lookup = {
        row["Network"]: row
        for _, row in df.iterrows()
    }

    # ------------------------------------------
    # B. 极淡内部辅助线
    # ------------------------------------------
    for i in range(1, N_ROWS):
        ax.hlines(
            i - 0.5,
            matrix_left,
            matrix_right,
            color=ROW_GUIDE_COLOR,
            linewidth=0.32,
            zorder=0,
        )

    for j in range(1, N_COLS):
        ax.vlines(
            j - 0.5,
            matrix_top,
            matrix_bottom,
            color=COL_GUIDE_COLOR,
            linewidth=0.32,
            zorder=0,
        )

    # ------------------------------------------
    # C. 绘制每个 Kendall 棒棒糖
    # ------------------------------------------
    for i, network in enumerate(NETWORK_ORDER):

        y = y_centers[i]
        row = row_lookup.get(network)

        if row is None:
            continue

        for j, (source_col, short_name) in enumerate(COMPARATORS):

            cx = x_centers[j]
            tau = float(row[source_col])

            left_x = local_x(cx, -1.0)
            zero_x = cx
            right_x = local_x(cx, 1.0)

            # 完整局部参考轴。
            ax.plot(
                [left_x, right_x],
                [y, y],
                color=LOCAL_AXIS_COLOR,
                linewidth=0.78,
                solid_capstyle="round",
                zorder=1,
            )

            # 两端空心圆。
            ax.plot(
                [left_x, right_x],
                [y, y],
                linestyle="None",
                marker="o",
                markersize=ENDPOINT_SIZE,
                markerfacecolor="white",
                markeredgecolor=LOCAL_AXIS_COLOR,
                markeredgewidth=0.60,
                zorder=2,
            )

            # 零点短竖线。
            ax.vlines(
                zero_x,
                y - 0.095,
                y + 0.095,
                color=ZERO_COLOR,
                linewidth=0.72,
                zorder=2,
            )

            if not np.isfinite(tau):
                continue

            tau_x = local_x(
                cx,
                tau,
            )

            color = COLORS[short_name]

            # 0 -> tau_b 彩色线段。
            ax.plot(
                [zero_x, tau_x],
                [y, y],
                color=color,
                linewidth=STEM_WIDTH,
                solid_capstyle="round",
                zorder=3,
            )

            # 所有结果统一圆形。
            ax.plot(
                tau_x,
                y,
                marker=MARKER,
                markersize=MARKER_SIZE,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=MARKER_EDGE_WIDTH,
                linestyle="None",
                zorder=4,
            )

            # 精确 Kendall 系数。
            if tau > 0.78:
                text_x = tau_x - 0.020
                ha = "right"

            elif tau < -0.78:
                text_x = tau_x + 0.020
                ha = "left"

            else:
                text_x = tau_x
                ha = "center"

            ax.text(
                text_x,
                y - VALUE_OFFSET_Y,
                fmt_tau(tau),
                ha=ha,
                va="bottom",
                fontsize=8.0,
                color=TEXT_COLOR,
                zorder=5,
            )

    # ==========================================
    # 5. 坐标轴：与 SIR 图保持一致
    # ==========================================

    ax.set_yticks(
        y_centers
    )

    ax.set_yticklabels(
        NETWORK_ORDER,
        fontsize=9,
    )

    ax.set_xticks(
        x_centers
    )

    # 底部方法全部采用缩写。
    ax.set_xticklabels(
        [
            "DC",
            "CP",
            "SH",
            "ISH",
            "SNIM",
        ],
        rotation=0,
        ha="center",
        fontsize=9,
    )

    ax.set_ylabel(
        "Network",
        fontsize=11,
        labelpad=7,
    )

    ax.set_xlabel("")

    ax.tick_params(
        axis="x",
        bottom=True,
        top=False,
        labelbottom=True,
        direction="out",
        length=3.0,
        width=0.7,
        colors="black",
        pad=5,
    )

    ax.tick_params(
        axis="y",
        left=True,
        right=False,
        direction="out",
        length=3.0,
        width=0.7,
        colors="black",
        pad=4,
    )

    # Lesmis 在顶部，Power 在底部。
    ax.set_ylim(
        N_ROWS - 0.52,
        -0.52,
    )

    # 删除右端 Table 数据后，x 范围严格围绕矩阵。
    ax.set_xlim(
        matrix_left,
        matrix_right,
    )

    # ==========================================
    # 6. 完整四边框
    # ==========================================

    frame_lw = 0.8

    # Top
    ax.plot(
        [matrix_left, matrix_right],
        [matrix_top, matrix_top],
        color="black",
        linewidth=frame_lw,
        zorder=10,
        clip_on=False,
    )

    # Bottom
    ax.plot(
        [matrix_left, matrix_right],
        [matrix_bottom, matrix_bottom],
        color="black",
        linewidth=frame_lw,
        zorder=10,
        clip_on=False,
    )

    # Left
    ax.plot(
        [matrix_left, matrix_left],
        [matrix_top, matrix_bottom],
        color="black",
        linewidth=frame_lw,
        zorder=10,
        clip_on=False,
    )

    # Right
    ax.plot(
        [matrix_right, matrix_right],
        [matrix_top, matrix_bottom],
        color="black",
        linewidth=frame_lw,
        zorder=10,
        clip_on=False,
    )

    # ==========================================
    # 7. 清理与保存
    # ==========================================

    # 自定义矩阵边框，因此默认 spines 隐藏。
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.grid(False)

    # 紧凑布局，比例接近正文单栏/双栏图。
    fig.subplots_adjust(
        left=0.13,
        right=0.985,
        bottom=0.105,
        top=0.985,
    )

    fig.savefig(
        out_pdf,
        format="pdf",
    )

    fig.savefig(
        out_png,
        format="png",
        dpi=600,
    )

    plt.close(fig)


# ==========================================
# 8. 保存正文数值
# ==========================================

def save_manuscript_table(
    df: pd.DataFrame,
    path: Path,
):

    table = pd.DataFrame({
        "Network": df["Network"],
    })

    for source_col, short_name in COMPARATORS:
        table[short_name] = pd.to_numeric(
            df[source_col],
            errors="coerce",
        )

    table.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )


# ==========================================
# 9. 主流程
# ==========================================

def main(
    input_path: Path,
    output_dir: Path,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 82)
    print(
        "Final manuscript Kendall lollipop matrix"
    )
    print("=" * 82)

    df = load_kendall_results(
        input_path
    )

    out_pdf = (
        output_dir
        / "Fig_C2_MSH_Kendall_lollipop_matrix_manuscript.pdf"
    )

    out_png = (
        output_dir
        / "Fig_C2_MSH_Kendall_lollipop_matrix_manuscript.png"
    )

    out_csv = (
        output_dir
        / "Table_C2_Kendall_manuscript_values.csv"
    )

    plot_matrix_lollipop(
        df,
        out_pdf,
        out_png,
    )

    save_manuscript_table(
        df,
        out_csv,
    )

    print("\nSaved:")
    print(f"  PDF : {out_pdf}")
    print(f"  PNG : {out_png}")
    print(f"  CSV : {out_csv}")

    print()
    print(
        "Final manuscript layout: 9x5 Kendall matrix only; "
        "no Table-2 context columns."
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Create the final manuscript Kendall lollipop matrix "
            "without Table-2 context columns."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "CSV produced by the Kendall correlation experiment."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory.",
    )

    args = parser.parse_args()

    main(
        input_path=args.input,
        output_dir=args.output_dir,
    )
