import os
import argparse
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from utils import read_jsonl
import matplotlib.patheffects as pe
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

# --- Config ---
ITERATIONS = list(range(1, 20))
LANGS = ["de", "fr", "zh", "ja"]

# marker set (cycled if more langs)
MARKERS = ["o"]

PALETTES = {
    "okabe_ito": [
        "#0072B2", "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#D55E00", "#CC79A7", "#000000"
    ],
    "dark2": [
        "#1B9E77", "#D95F02", "#7570B3", "#E7298A",
        "#66A61E", "#E6AB02", "#A6761D", "#666666"
    ],
    "tableau10": [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"
    ]
}

def use_palette(name: str, n: int):
    if name in PALETTES:
        base = PALETTES[name]
        if n <= len(base):
            return base[:n]
        return (base * ((n + len(base) - 1) // len(base)))[:n]
    return sns.color_palette(name, n)


# --- Helpers ---
def compute_auc(hm_file, wm_file):
    """Compute AUC from two jsonl zscore files using read_jsonl."""
    if not os.path.exists(hm_file) or not os.path.exists(wm_file):
        return None
    hm_list = read_jsonl(hm_file)
    wm_list = read_jsonl(wm_file)
    if len(hm_list) != len(wm_list):
        print(f"⚠️ zscore lengths differ: {hm_file} ({len(hm_list)}) vs {wm_file} ({len(wm_list)})")
        return None
    hm_z = [x.get("z_score", 0) if x.get("z_score") is not None else 0 for x in hm_list]
    wm_z = [x.get("z_score", 0) if x.get("z_score") is not None else 0 for x in wm_list]
    y_true = [0] * len(hm_list) + [1] * len(wm_list)
    y_scores = hm_z + wm_z
    try:
        return roc_auc_score(y_true, y_scores)
    except Exception as e:
        print(f"⚠️ Failed to compute AUC for {hm_file} / {wm_file}: {e}")
        return None

def get_clusters_count(clusters_file):
    """Return number of clusters (length of JSON array/object)."""
    if not os.path.exists(clusters_file):
        return None
    with open(clusters_file, "r") as f:
        clusters = json.load(f)
    if isinstance(clusters, dict):
        return len(clusters)
    return len(clusters)


def lighten(color, amount=0.6):
    """Lighten a color by mixing with white."""
    try:
        r, g, b = mcolors.to_rgb(color)
    except Exception:
        r, g, b = (0.0, 0.0, 0.0)
    r = 1 - (1 - r) * (1 - amount)
    g = 1 - (1 - g) * (1 - amount)
    b = 1 - (1 - b) * (1 - amount)
    return (r, g, b)


# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Plot AUC vs cluster count per language (one line per language) with SIR glow.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base dir with sir/xsir results.")
    parser.add_argument("--model_abbr", type=str, required=True, help="Model abbreviation used in mapping files.")
    parser.add_argument("--mapping_dir", type=str, required=True, help="Directory containing iteration subdirs with mapping JSONs.")
    parser.add_argument("--palette", type=str, default="dark2", help="Palette name key in PALETTES or seaborn palette name.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--latex_width_in", type=float, default=6.5)
    parser.add_argument("--latex_height_in", type=float, default=5.0)
    parser.add_argument("--marker_size", type=float, default=30)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--axis_fontsize", type=int, default=10)
    parser.add_argument("--sir_glow_mult", type=float, default=3.2)
    parser.add_argument("--sir_glow_alpha", type=float, default=0.28)
    parser.add_argument("--sir_glow_lighten", type=float, default=0.66)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelsize": args.axis_fontsize,
        "xtick.labelsize": args.axis_fontsize - 1,
        "ytick.labelsize": args.axis_fontsize - 1,
        "figure.dpi": 300,
    })

    palette = use_palette(args.palette, n=len(LANGS))
    fig, ax = plt.subplots(figsize=(args.latex_width_in, args.latex_height_in))

    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--", linewidth=0.85, alpha=0.45)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.22)

    all_x = []

    for li, lang in enumerate(LANGS):
        color = palette[li % len(palette)]
        marker = MARKERS[li % len(MARKERS)]

        # SIR baseline AUC
        sir_hm = os.path.join(args.base_dir, "sir", f"mc4.en-{lang}.hum.z_score.jsonl")
        sir_wm = os.path.join(args.base_dir, "sir", f"mc4.en-{lang}.mod.z_score.jsonl")
        sir_auc = compute_auc(sir_hm, sir_wm)

        # collect X-SIR points
        points = []
        for it in ITERATIONS:
            clusters_file = os.path.join(args.mapping_dir, str(it), f"300_mapping_{args.model_abbr}_clusters.json")
            clusters_count = get_clusters_count(clusters_file)
            if clusters_count is None:
                continue
            it_hm = os.path.join(args.base_dir, "xsir", str(it), f"mc4.en-{lang}.hum.z_score.jsonl")
            it_wm = os.path.join(args.base_dir, "xsir", str(it), f"mc4.en-{lang}.mod.z_score.jsonl")
            it_auc = compute_auc(it_hm, it_wm)
            if it_auc is None:
                continue
            points.append((clusters_count, it_auc))

        if not points:
            continue

        points.sort(key=lambda x: x[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        all_x.extend(xs)

        # X-SIR line + markers
        ax.plot(xs, ys, marker=marker, linestyle='-', linewidth=1.6, markersize=0,
                label=lang, color=color, alpha=0.65, zorder=3)
        ax.scatter(xs, ys, s=args.marker_size, marker=marker, facecolors=color,
                   edgecolors="white", linewidths=0.6, alpha=args.alpha, zorder=4)

        # SIR horizontal line
        if sir_auc is not None:
            xmin, xmax = min(xs), max(xs)
            glow_color = lighten(color, amount=args.sir_glow_lighten)
            ax.plot([xmin, xmax], [sir_auc, sir_auc],
                    color=glow_color, linewidth=args.sir_glow_mult,
                    solid_capstyle="round", alpha=args.sir_glow_alpha, zorder=1)
            ax.plot([xmin, xmax], [sir_auc, sir_auc],
                    color=color, linestyle='--', linewidth=1.25, alpha=0.75, zorder=2)

    # widen x axis
    if all_x:
        xmin, xmax = min(all_x), max(all_x)
        pad = 0.06 * (xmax - xmin) if xmax > xmin else 10
        ax.set_xlim(xmin - pad, xmax + pad)

    # Axis labels with larger, bold font
    ax.set_xlabel("Cluster count (number of clusters)",
                  fontsize=args.axis_fontsize + 3, labelpad=10)
    ax.set_ylabel("AUC",
                  fontsize=args.axis_fontsize + 3, labelpad=10)

    # Tick parameters for better visibility
    ax.tick_params(axis="both", which="major",
                   labelsize=args.axis_fontsize + 2,
                   width=1.4, length=6, direction="out")
    ax.tick_params(axis="both", which="minor",
                   labelsize=args.axis_fontsize + 1,
                   width=1.0, length=4, direction="out")

    # Legend for languages
    lang_legend = ax.legend(frameon=True, fontsize=args.axis_fontsize + 1, loc="lower right")

    # Custom legend for SIR vs X-SIR
    custom_lines = [
        Line2D([0], [0], color="black", linestyle='--', lw=1.25, label="SIR"),
        Line2D([0], [0], color="black", linestyle='-', lw=1.6, label="X-SIR"),
    ]
    ax.add_artist(lang_legend)  # keep lang legend
    ax.legend(handles=custom_lines, frameon=True, fontsize=args.axis_fontsize + 1, loc="best")

    sns.despine(ax=ax)
    fig.tight_layout()

    png_path = os.path.join(args.output_dir, "auc_vs_cluster_count_per_lang_with_sir_glow.png")
    pdf_path = os.path.join(args.output_dir, "auc_vs_cluster_count_per_lang_with_sir_glow.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    print(f"✅ Saved {png_path}")
    print(f"✅ Saved {pdf_path}")


if __name__ == "__main__":
    main()
