"""
Experiment: Spreader Separation / Spatial Dispersion Analysis (revised)

Purpose
-------
Evaluate whether the top-ranked nodes selected by different methods are spatially
separated in the network, using the average shortest-path distance among selected
seed nodes:

    L_s(S) = 2 / (|S|(|S|-1)) * sum_{i<j, i,j in S} d(i,j)

Revision points
---------------
1. Seed ratios are fixed to p = 2%, 4%, 6%, 8%, 10%.
2. Three additional baseline methods are included: VoteRank, CHBC, and SNIM.
3. Figures are saved immediately after each network is processed.
4. Each network produces editable PDF and high-resolution PNG figures.
5. A standalone legend is exported, and individual network figures do not contain legends.
6. The network name is displayed above each figure.
7. The output style follows the revised SIR template: Times New Roman, black borders,
   editable PDF, separate Excel data for each network, and checkpoint summary export.

Notes
-----
- The graph loader is expected to return the preprocessed graph used by other experiments.
- The actual number of selected seeds is k = max(2, ceil(pN)) to make pairwise
  distance well-defined for small networks.
- Higher L_s indicates a more spatially dispersed seed set. This metric should be
  interpreted as a dispersion diagnostic rather than direct evidence of node influence.
"""

import argparse
import os
import warnings
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from hosh_methods import get_node_scores
from network_loader import download_and_load_graph, get_network_list
from precompute_rankings import load_precomputed_rankings, get_standardized_ranked_nodes


# ==========================================
# 0. Plot configuration, consistent with SIR template
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
    'lines.linewidth': 1.5,
    'lines.markersize': 4.5,
    'axes.grid': False,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
})


DEFAULT_METHODS = [
    'HOSH', 'VoteRank', 'SNIM', 'CHBC',
    'ISH', 'DC', 'BC', 'CC', 'K-Shell', 'SH', 'CI', 'SNC'
]

COLORS = {
    'HOSH': '#D63230',
    'VoteRank': '#2CA02C',
    'SNIM': '#7F7F7F',
    'CHBC': '#5D3FD3',
    'ISH': '#F08C3D',
    'DC': '#E5B25D',
    'BC': '#4FA3D1',
    'CC': '#4364B8',
    'K-Shell': '#A855A8',
    'SH': '#E2739F',
    'CI': '#8D6E63',
    'SNC': '#4DB6AC',
}

MARKERS = {
    'HOSH': 'o',
    'VoteRank': 'o',
    'SNIM': 'p',
    'CHBC': '*',
    'ISH': 's',
    'DC': '^',
    'BC': 'D',
    'CC': 'X',
    'K-Shell': 'P',
    'SH': 'v',
    'CI': 'h',
    'SNC': 'H',
}

SEED_RATIOS = np.array([0.02, 0.04, 0.06, 0.08, 0.10], dtype=float)


# ==========================================
# 1. Utility functions
# ==========================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_networks(network_arg: str | None) -> List[str]:
    if network_arg is None or network_arg.strip() == '':
        return get_network_list()
    return [x.strip() for x in network_arg.split(',') if x.strip()]


def export_standalone_legend(methods: Sequence[str], colors: Dict[str, str],
                             markers: Dict[str, str], output_dir: str) -> None:
    """Export a standalone legend so individual panels remain clean."""
    fig, ax = plt.subplots(figsize=(8.5, 1.2))
    ax.axis('off')

    handles = []
    for m in methods:
        line, = ax.plot(
            [], [],
            label=m,
            color=colors.get(m, '#000000'),
            marker=markers.get(m, 'o'),
            linestyle='--',
            linewidth=1.5,
            markersize=5.0,
            markerfacecolor=colors.get(m, '#000000'),
            markeredgecolor='black',
            markeredgewidth=0.5,
        )
        handles.append(line)

    ax.legend(
        handles=handles,
        loc='center',
        ncol=6,
        frameon=True,
        fancybox=False,
        shadow=False,
        edgecolor='black',
        framealpha=1.0,
        fontsize=10,
        columnspacing=0.9,
        handlelength=1.4,
        handletextpad=0.4,
        borderpad=0.4,
    )

    pdf_path = os.path.join(output_dir, 'Spreader_Separation_Legend_Standalone.pdf')
    png_path = os.path.join(output_dir, 'Spreader_Separation_Legend_Standalone.png')
    plt.savefig(pdf_path, format='pdf')
    plt.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"[Output] Standalone legend saved: {pdf_path}")


def get_ranked_nodes(method_name: str, graph: nx.Graph, precomputed: Dict | None) -> List:
    """Load precomputed ranking if possible; otherwise compute scores on the fly."""
    if precomputed and method_name in precomputed and precomputed[method_name]:
        scores = precomputed[method_name]
    else:
        scores = get_node_scores(method_name, graph)

    # Prefer the same deterministic ranking utility used in the revised SIR scripts.
    try:
        return list(get_standardized_ranked_nodes(scores))
    except Exception:
        # Fallback: score descending, then node id/string ascending for reproducible tie breaking.
        return [node for node, _ in sorted(scores.items(), key=lambda x: (-x[1], str(x[0])))]


# ==========================================
# 2. Spreader separation metric
# ==========================================
def calculate_spreader_separation(graph: nx.Graph, spreaders: Sequence) -> float:
    """
    Average shortest-path distance among selected spreaders.

    For a selected seed set S, this function computes the average over unordered pairs:
        L_s(S) = 2 / (|S|(|S|-1)) * sum_{i<j} d(i,j)

    If the graph is disconnected, unreachable pairs are assigned delta + 1, where
    delta is the largest connected-component diameter when available. In the standard
    preprocessing pipeline, graphs are expected to be connected after extracting LCC.
    """
    valid = [node for node in spreaders if graph.has_node(node)]
    if len(valid) <= 1:
        return 0.0

    valid_set = set(valid)

    # Standard loader should produce LCC. Keep a defensive unreachable distance.
    if nx.is_connected(graph):
        disconnected_distance = None
    else:
        try:
            largest_cc = max(nx.connected_components(graph), key=len)
            diameter = nx.diameter(graph.subgraph(largest_cc))
            disconnected_distance = diameter + 1
        except Exception:
            disconnected_distance = graph.number_of_nodes()

    total_distance = 0.0
    pair_count = 0

    # More efficient than calling shortest_path_length separately for every pair.
    for idx, source in enumerate(valid):
        lengths = nx.single_source_shortest_path_length(graph, source)
        for target in valid[idx + 1:]:
            distance = lengths.get(target, disconnected_distance)
            if distance is None:
                # This branch should not occur for connected graphs, but keeps the
                # function safe if preprocessing changes.
                distance = graph.number_of_nodes()
            total_distance += float(distance)
            pair_count += 1

    return total_distance / pair_count if pair_count > 0 else 0.0


def run_spreader_separation(methods: Sequence[str], graph: nx.Graph,
                            network_name: str | None = None,
                            seed_ratios: np.ndarray = SEED_RATIOS) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    """Run the separation analysis for all methods and specified seed ratios."""
    print("  [Exp: Spreader Separation] Running spatial dispersion analysis...")

    num_nodes = graph.number_of_nodes()
    precomputed = load_precomputed_rankings(network_name) if network_name else None

    # k is ratio-specific but method-independent.
    k_values = [max(2, int(np.ceil(num_nodes * ratio))) for ratio in seed_ratios]

    method_values: Dict[str, List[float]] = {}
    seed_records = []

    for method_name in methods:
        print(f"    Evaluating: {method_name}")
        ranked_nodes = get_ranked_nodes(method_name, graph, precomputed)

        values = []
        for ratio, k in tqdm(list(zip(seed_ratios, k_values)), desc=f"    {method_name}", leave=False):
            seeds = ranked_nodes[:k]
            avg_distance = calculate_spreader_separation(graph, seeds)
            values.append(avg_distance)

            seed_records.append({
                'Method': method_name,
                'Seed_Ratio_%': ratio * 100,
                'k': k,
                'AvgShortestPathDistance': avg_distance,
                'Selected_Seeds': ','.join(map(str, seeds)),
            })

        method_values[method_name] = values

    df_records = pd.DataFrame(seed_records)
    return df_records, method_values


# ==========================================
# 3. Plotting
# ==========================================
def plot_radar_results(network_name: str, method_values: Dict[str, List[float]],
                       methods: Sequence[str], colors: Dict[str, str], markers: Dict[str, str],
                       output_dir: str, normalize: bool = True) -> None:
    """Draw radar chart for p = 2%, 4%, 6%, 8%, 10%."""
    labels = ['2%', '4%', '6%', '8%', '10%']
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    # Extract available methods.
    radar_data = {m: list(method_values[m]) for m in methods if m in method_values}
    if not radar_data:
        return

    all_values = [v for vals in radar_data.values() for v in vals]
    raw_min = min(all_values)
    raw_max = max(all_values)

    plot_data = {}
    if normalize:
        value_range = raw_max - raw_min
        if value_range <= 1e-12:
            plot_data = {m: [0.5] * len(vals) for m, vals in radar_data.items()}
        else:
            plot_data = {m: [(v - raw_min) / value_range for v in vals] for m, vals in radar_data.items()}
        radial_upper = 1.03
        scale_note = f"normalized; raw range [{raw_min:.2f}, {raw_max:.2f}]"
    else:
        plot_data = radar_data
        radial_upper = raw_max * 1.10 if raw_max > 0 else 1.0
        scale_note = f"raw scale; range [{raw_min:.2f}, {raw_max:.2f}]"

    print(f"      [Scale] {network_name}: {scale_note}")

    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111, projection='polar')

    # Plot baselines first, HOSH last.
    plot_order = [m for m in methods if m in plot_data and m != 'HOSH']
    if 'HOSH' in plot_data:
        plot_order.append('HOSH')

    for method_name in plot_order:
        values = plot_data[method_name] + plot_data[method_name][:1]
        is_hosh = method_name == 'HOSH'
        ax.plot(
            angles,
            values,
            color=colors.get(method_name, '#000000'),
            linewidth=2.0 if is_hosh else 1.25,
            linestyle='--',
            marker=markers.get(method_name, 'o'),
            markersize=5.8 if is_hosh else 4.2,
            markerfacecolor=colors.get(method_name, '#000000'),
            markeredgecolor='black',
            markeredgewidth=0.5,
            alpha=1.0 if is_hosh else 0.78,
            zorder=100 if is_hosh else 40,
        )
        ax.fill(
            angles,
            values,
            color=colors.get(method_name, '#000000'),
            alpha=0.18 if is_hosh else 0.08,
            linewidth=0,
            zorder=80 if is_hosh else 20,
        )

    ax.set_title(network_name, y=1.12, fontsize=11, fontweight='bold')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9, fontweight='bold')
    ax.set_ylim(0, radial_upper)
    ax.set_yticklabels([])
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.35, color='gray')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color('black')

    plt.subplots_adjust(left=0.12, right=0.88, top=0.86, bottom=0.10)

    pdf_path = os.path.join(output_dir, f"Spreader_Separation_Radar_{network_name}.pdf")
    png_path = os.path.join(output_dir, f"Spreader_Separation_Radar_{network_name}.png")
    plt.savefig(pdf_path, format='pdf')
    plt.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"    [Output] Radar PDF saved: {pdf_path}")


def plot_line_results(network_name: str, method_values: Dict[str, List[float]],
                      methods: Sequence[str], colors: Dict[str, str], markers: Dict[str, str],
                      output_dir: str) -> None:
    """Draw a SIR-template-like line figure across seed ratios."""
    x_percent = SEED_RATIOS * 100
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    for method_name in methods:
        if method_name not in method_values:
            continue
        y = np.asarray(method_values[method_name], dtype=float)
        is_hosh = method_name == 'HOSH'
        ax.plot(
            x_percent,
            y,
            label=method_name,
            color=colors.get(method_name, '#000000'),
            linestyle='--',
            marker=markers.get(method_name, 'o'),
            linewidth=1.8 if is_hosh else 1.2,
            markersize=5.0 if is_hosh else 4.0,
            markerfacecolor=colors.get(method_name, '#000000'),
            markeredgecolor='black',
            markeredgewidth=0.5,
            alpha=0.98 if is_hosh else 0.85,
            zorder=100 if is_hosh else 40,
        )

    ax.set_title(network_name, fontsize=11, fontweight='bold', pad=7)
    ax.set_xlabel('$p$ (%)')
    ax.set_ylabel('$L_s$')
    ax.set_xlim(1.5, 10.5)
    ax.set_xticks([2, 4, 6, 8, 10])

    all_values = [v for vals in method_values.values() for v in vals]
    if all_values:
        y_min, y_max = min(all_values), max(all_values)
        y_range = max(y_max - y_min, 1e-6)
        ax.set_ylim(max(0, y_min - 0.08 * y_range), y_max + 0.10 * y_range)

    for spine in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color('black')
    ax.tick_params(direction='out', which='major', length=3.0, width=0.7)
    plt.tight_layout(pad=0.2)

    pdf_path = os.path.join(output_dir, f"Spreader_Separation_Line_{network_name}.pdf")
    png_path = os.path.join(output_dir, f"Spreader_Separation_Line_{network_name}.png")
    plt.savefig(pdf_path, format='pdf')
    plt.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"    [Output] Line PDF saved: {pdf_path}")


# ==========================================
# 4. Data export
# ==========================================
def save_network_excel(network_name: str, df_records: pd.DataFrame,
                       method_values: Dict[str, List[float]], methods: Sequence[str],
                       output_dir: str) -> str:
    """Save per-network detailed and wide-format results immediately."""
    wide_data = {
        'Seed_Ratio_%': SEED_RATIOS * 100,
    }

    # k is identical across methods; retrieve from any method if available.
    if not df_records.empty:
        k_by_ratio = (
            df_records[['Seed_Ratio_%', 'k']]
            .drop_duplicates()
            .sort_values('Seed_Ratio_%')
        )
        wide_data['k'] = k_by_ratio['k'].to_numpy()

    for method_name in methods:
        if method_name in method_values:
            wide_data[f'{method_name}_AvgDistance'] = method_values[method_name]

    df_wide = pd.DataFrame(wide_data)

    excel_path = os.path.join(output_dir, f"Spreader_Separation_Data_{network_name}.xlsx")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_wide.to_excel(writer, sheet_name='Wide_Results', index=False)
        df_records.to_excel(writer, sheet_name='Long_Results_and_Seeds', index=False)

        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for col in worksheet.columns:
                max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col)
                worksheet.column_dimensions[col[0].column_letter].width = min(max(max_len + 2, 12), 45)

    print(f"    [Output] Excel data saved: {excel_path}")
    return excel_path


def save_combined_checkpoint(all_network_records: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """Save a combined checkpoint after each processed network."""
    if not all_network_records:
        return

    combined_long = []
    combined_summary = []

    for network_name, df in all_network_records.items():
        temp = df.copy()
        temp.insert(0, 'Network', network_name)
        combined_long.append(temp)

        summary = (
            temp.groupby(['Network', 'Method'], as_index=False)['AvgShortestPathDistance']
            .mean()
            .rename(columns={'AvgShortestPathDistance': 'Mean_AvgDistance_over_2_4_6_8_10'})
        )
        combined_summary.append(summary)

    checkpoint_path = os.path.join(output_dir, 'Spreader_Separation_Combined_Checkpoint.xlsx')
    with pd.ExcelWriter(checkpoint_path, engine='openpyxl') as writer:
        pd.concat(combined_long, ignore_index=True).to_excel(writer, sheet_name='Long_All', index=False)
        pd.concat(combined_summary, ignore_index=True).to_excel(writer, sheet_name='Mean_By_Network_Method', index=False)

    print(f"    [Checkpoint] Combined results updated: {checkpoint_path}")


# ==========================================
# 5. Main workflow
# ==========================================
def main() -> None:
    parser = argparse.ArgumentParser(description='Spreader separation analysis with p=2,4,6,8,10 and expanded baselines.')
    parser.add_argument('--networks', type=str, default=None,
                        help='Comma-separated network names. Default: all networks from get_network_list().')
    parser.add_argument('--output-dir', type=str, default='results/exp_spreader_separation_revised',
                        help='Output directory.')
    parser.add_argument('--plot-type', type=str, default='both', choices=['radar', 'line', 'both'],
                        help='Figure type to generate for each network.')
    parser.add_argument('--no-normalize-radar', action='store_true',
                        help='Use raw radial scale for radar plots instead of within-network normalization.')
    args = parser.parse_args()

    print('=' * 70)
    print(' Experiment: Spreader Separation Analysis (Revised)')
    print(' Seed ratios: 2%, 4%, 6%, 8%, 10%')
    print(' Methods: ' + ', '.join(DEFAULT_METHODS))
    print('=' * 70)

    output_dir = args.output_dir
    ensure_dir(output_dir)

    methods = DEFAULT_METHODS
    networks = parse_networks(args.networks)

    export_standalone_legend(methods, COLORS, MARKERS, output_dir)

    all_network_records: Dict[str, pd.DataFrame] = {}

    for idx, network_name in enumerate(networks, 1):
        print(f"\n[{idx}/{len(networks)}] Processing network: {network_name}")
        print('-' * 70)

        try:
            graph = download_and_load_graph(network_name)
            if graph is None or graph.number_of_nodes() == 0:
                print(f"  [Skip] Network {network_name} is empty or failed to load")
                continue

            print(f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
            print(f"  Seed ratios: {[int(r * 100) for r in SEED_RATIOS]}%")
            print(f"  Actual k values: {[max(2, int(np.ceil(graph.number_of_nodes() * r))) for r in SEED_RATIOS]}")

            df_records, method_values = run_spreader_separation(
                methods=methods,
                graph=graph,
                network_name=network_name,
                seed_ratios=SEED_RATIOS,
            )

            # Save data and figures immediately after each network.
            save_network_excel(network_name, df_records, method_values, methods, output_dir)

            if args.plot_type in ('radar', 'both'):
                plot_radar_results(
                    network_name,
                    method_values,
                    methods,
                    COLORS,
                    MARKERS,
                    output_dir,
                    normalize=not args.no_normalize_radar,
                )

            if args.plot_type in ('line', 'both'):
                plot_line_results(
                    network_name,
                    method_values,
                    methods,
                    COLORS,
                    MARKERS,
                    output_dir,
                )

            all_network_records[network_name] = df_records
            save_combined_checkpoint(all_network_records, output_dir)

            print(f"  [Done] Finished network: {network_name}")

        except Exception as exc:
            print(f"  [Error] Failed to process {network_name}: {exc}")
            import traceback
            traceback.print_exc()
            continue

    print('\n' + '=' * 70)
    print(' Experiment: Spreader Separation Completed!')
    print('=' * 70)
    print(f"Results saved to: {output_dir}/")


if __name__ == '__main__':
    # Suppress harmless font warnings when Times New Roman is unavailable locally.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main()
