import argparse
import os
import json
import matplotlib.pyplot as plt
from utils import read_jsonl
from sklearn.metrics import roc_auc_score
import math

# === Config you provided ===
TGT_LANGS = [
    "fr", "de", "it", "es", "pt",
    "pl", "nl", "ru", "hi", "ko", "ja",
    "bn", "fa", "vi", "iw", "uk", "ta",
]

MODEL_ABBR = [
    "llama-3.2-1B",
    "aya-23-8B",
    "llamax3-8B"
]

SEEDS = ["0", "42", "123"]

# Distinct marker styles per model (no explicit colors)
MARKERS = ["o", "s", "D", "^", "v", "P", "X"]


def compute_auc(base_dir, model_abbr, tgt_lang, seed):
    hm_zscore = os.path.join(base_dir, model_abbr, f"xsir_seed{seed}", f"mc4.en-{tgt_lang}.hum.z_score.jsonl")
    wm_zscore = os.path.join(base_dir, model_abbr, f"xsir_seed{seed}", f"mc4.en-{tgt_lang}.mod.z_score.jsonl")
    hm_list = read_jsonl(hm_zscore)
    wm_list = read_jsonl(wm_zscore)

    if len(hm_list) != len(wm_list):
        print(f"⚠️ zscore lengths differ for {model_abbr} {tgt_lang} seed={seed}")
        return None

    hm_z = [x["z_score"] if x["z_score"] is not None else 0 for x in hm_list]
    wm_z = [x["z_score"] if x["z_score"] is not None else 0 for x in wm_list]

    y_true = [0] * len(hm_list) + [1] * len(wm_list)
    y_scores = hm_z + wm_z

    return roc_auc_score(y_true, y_scores)


def repel_text(ax, texts, anchors, max_iter=250, step=0.01):
    """
    Simple force-based label repulsion:
    - texts: list of Text objects
    - anchors: list of (x,y) anchor points for leader lines (same order as texts)
    """
    fig = ax.figure
    fig.canvas.draw()  # ensure we have a renderer

    for _ in range(max_iter):
        moved = False
        # get bboxes in display coords
        bboxes = [t.get_window_extent(renderer=fig.canvas.get_renderer()).expanded(1.05, 1.2)
                  for t in texts]

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if not bboxes[i].overlaps(bboxes[j]):
                    continue
                # push i and j apart in data coords
                xi, yi = texts[i].get_position()
                xj, yj = texts[j].get_position()
                dx, dy = xi - xj, yi - yj
                dist = math.hypot(dx, dy) or 1e-6
                ux, uy = dx / dist, dy / dist  # unit vector
                # move both half a step in opposite directions
                texts[i].set_position((xi + ux * step, yi + uy * step))
                texts[j].set_position((xj - ux * step, yj - uy * step))
                moved = True

        if not moved:
            break

    # After settling, draw leader lines from final label pos -> anchor
    for t, (xa, ya) in zip(texts, anchors):
        xt, yt = t.get_position()
        ax.plot([xa, xt], [ya, yt], lw=0.6, alpha=0.5)


def main():
    parser = argparse.ArgumentParser(description="Line plot: avg AUC vs tokenizer words (per language) for multiple models.")
    parser.add_argument("--base_dir", type=str, required=True, help="Directory containing model results.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing per-model stats.json.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save plots.")
    parser.add_argument("--stats_file", type=str, default="stats.json", help="Stats filename inside each model dir.")
    # LaTeX-friendly sizing options
    parser.add_argument("--latex_width_in", type=float, default=6.5, help="Figure width in inches.")
    parser.add_argument("--latex_height_in", type=float, default=5.0, help="Figure height in inches.")
    # Aesthetics
    parser.add_argument("--marker_size", type=float, default=70, help="Marker size.")
    parser.add_argument("--lang_fontsize", type=int, default=9, help="Font size for language labels.")
    parser.add_argument("--axis_fontsize", type=int, default=10, help="Font size for axis labels/ticks.")
    parser.add_argument("--title_fontsize", type=int, default=12, help="Font size for title.")
    parser.add_argument("--line_width", type=float, default=1.8, help="Line width for model curves.")
    # Small x-jitter to separate different models’ points at similar x
    parser.add_argument("--x_jitter", type=float, default=50.0, help="Per-model x jitter in tokenizer-word units.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Paper-ish style
    plt.rcParams.update({
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelsize": args.axis_fontsize,
        "xtick.labelsize": args.axis_fontsize - 1,
        "ytick.labelsize": args.axis_fontsize - 1,
        "legend.fontsize": args.axis_fontsize - 1,
        "figure.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(args.latex_width_in, args.latex_height_in))

    labeled_models = set()
    all_texts = []
    all_anchors = []

    for midx, model_abbr in enumerate(MODEL_ABBR):
        marker = MARKERS[midx % len(MARKERS)]

        stats_path = os.path.join(args.data_dir, model_abbr, args.stats_file)
        if not os.path.exists(stats_path):
            print(f"Stats file not found: {stats_path}")
            continue

        with open(stats_path, "r") as f:
            stats = json.load(f)

        # Collect rows for sorting by x
        rows = []
        for tgt_lang in TGT_LANGS:
            auc_values = []
            for seed in SEEDS:
                auc = compute_auc(args.base_dir, model_abbr, tgt_lang, seed)
                if auc is not None:
                    auc_values.append(auc)
            if not auc_values:
                continue
            avg_auc = sum(auc_values) / len(auc_values)
            x_tokens = stats["per_language"][tgt_lang]["unique_words_hit_single_token"]
            # Apply a tiny x-jitter per model to reduce direct overlap of markers/labels
            jitter = (midx - (len(MODEL_ABBR) - 1) / 2.0) * args.x_jitter
            rows.append((x_tokens + jitter, avg_auc, tgt_lang, x_tokens))  # keep raw x_tokens for the leader anchor

        if not rows:
            continue

        rows.sort(key=lambda r: r[0])  # sort by jittered x so lines go left->right

        x_vals = [r[0] for r in rows]
        y_vals = [r[1] for r in rows]
        langs  = [r[2] for r in rows]
        x_raw  = [r[3] for r in rows]

        label = model_abbr if model_abbr not in labeled_models else None
        # marker size in plot uses points; approximate from area arg
        ax.plot(x_vals, y_vals, marker=marker, linewidth=args.line_width,
                markersize=(args.marker_size ** 0.5), label=label)
        if label is not None:
            labeled_models.add(model_abbr)

        # Add labels now; we will repel them later and then add leader lines.
        for xj, y, lang, xr in zip(x_vals, y_vals, langs, x_raw):
            t = ax.text(xj, y, lang, fontsize=args.lang_fontsize, ha="center", va="center")
            all_texts.append(t)
            all_anchors.append((xj, y))  # anchor is the jittered point location

    # Repel overlapping labels and add leader lines
    repel_text(ax, all_texts, all_anchors, max_iter=300, step=0.012)

    ax.set_xlabel("Tokenizer single-token vocabulary per language")
    ax.set_ylabel("Average AUC across seeds")
    ax.set_title("Average AUC vs Tokenizer Coverage (per Language)", fontsize=args.title_fontsize)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Models", loc="best", frameon=True)

    fig.tight_layout()

    png_path = os.path.join(args.output_dir, "avg_auc_vs_tokens_lines_repel.png")
    pdf_path = os.path.join(args.output_dir, "avg_auc_vs_tokens_lines_repel.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)  # for LaTeX
    print(f"✅ Saved {png_path}")
    print(f"✅ Saved {pdf_path}")


if __name__ == "__main__":
    main()
