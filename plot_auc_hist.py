#!/usr/bin/env python3
# auc_hist_acl_stats.py
import os, json, argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ─── Helpers ────────────────────────────────────────────────────────────────
def load_zscores(path):
    with open(path) as f:
        return [float(json.loads(line)["z_score"]) for line in f]

def compute_auc(hum_path, atk_path):
    h = load_zscores(hum_path)
    a = load_zscores(atk_path)
    y_true  = [0]*len(h) + [1]*len(a)
    y_score =  h        +  a
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)

def nice_rc(dpi=300, dark=False):
    """ACL-style fonts and clean look (no LaTeX dependency)."""
    bg = "#0d0f12" if dark else "white"
    fg = "white" if dark else "black"
    mpl.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",   # Computer Modern
        "font.family": "serif",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.edgecolor": fg,
        "text.color": fg,
        "axes.labelcolor": fg,
        "xtick.color": fg,
        "ytick.color": fg,
        "grid.color": "gray",
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "figure.facecolor": bg,
        "axes.facecolor": bg,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "pdf.fonttype": 42, "ps.fonttype": 42
    })
    return bg, fg

def pick_colors(name: str, dark=False):
    """Return (bar_color, line_color, band_color_rgba) by palette name."""
    name = (name or "blue").lower()
    if dark:
        base = {
            "blue":   ("#60a5fa", "#1d4ed8"),  # bar, line
            "teal":   ("#5eead4", "#0f766e"),
            "purple": ("#c4b5fd", "#6d28d9"),
            "green":  ("#86efac", "#15803d"),
            "orange": ("#fdba74", "#c2410c"),
        }
    else:
        base = {
            "blue":   ("#2563eb", "#1e40af"),
            "teal":   ("#0d9488", "#115e59"),
            "purple": ("#7c3aed", "#5b21b6"),
            "green":  ("#16a34a", "#166534"),
            "orange": ("#ea580c", "#9a3412"),
        }
    bar, line = base.get(name, base["blue"])
    # translucent band color (line color with alpha)
    band = (*mpl.colors.to_rgb(line), 0.12)
    return bar, line, band

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Plot AUC histogram per language (ACL style) with μ/σ")
    parser.add_argument("--model_abbr", required=True)
    parser.add_argument("--base_dir",   required=True)
    parser.add_argument("--seeds",      nargs="+", required=True)
    parser.add_argument("--langs",      nargs="+", required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--bins",       type=int, default=18, help="Histogram bins")
    parser.add_argument("--palette",    default="teal", choices=["blue","teal","purple","green","orange"])
    parser.add_argument("--dark",       action="store_true", help="Dark theme text/axes")
    parser.add_argument("--xlim",       nargs=2, type=float, default=None, help="Optional x-limits, e.g. 0.5 1.0")
    args = parser.parse_args()

    _, fg = nice_rc(dark=args.dark)  # apply ACL-style fonts and params
    seeds = sorted(int(s) for s in args.seeds)
    bar_c, line_c, band_c = pick_colors(args.palette, dark=args.dark)

    for lang in args.langs:
        auc_vals = []
        for seed in seeds:
            subdir   = os.path.join(args.base_dir, "xsir", f"seed_{seed}")
            hum_path = os.path.join(subdir, f"mc4.en-{lang}.hum.z_score.jsonl")
            atk_path = os.path.join(subdir, f"mc4.en-{lang}.mod.z_score.jsonl")
            if os.path.exists(hum_path) and os.path.exists(atk_path):
                try:
                    auc_vals.append(compute_auc(hum_path, atk_path))
                except Exception as e:
                    print(f"[Error] seed={seed}, lang={lang}: {e}")
            else:
                print(f"[Missing] seed={seed}, lang={lang}")

        if not auc_vals:
            print(f"No data for {lang}, skipping.")
            continue

        auc_vals = np.array(auc_vals, dtype=float)
        mu  = float(np.mean(auc_vals))
        std = float(np.std(auc_vals, ddof=1)) if len(auc_vals) > 1 else 0.0

        # ─── Plot ────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(6.0, 4.0))

        # Histogram
        ax.hist(
            auc_vals,
            bins=args.bins,
            color=bar_c,
            edgecolor="black",
            alpha=0.90
        )

        # Stats overlays: mean line and ±1σ band
        # ax.axvline(mu, color=line_c, linewidth=2.0, linestyle="-", zorder=3)
        # ax.axvspan(mu - std, mu + std, color=band_c, zorder=2)

        # Neat stats box
        stats_text = rf"$\mu={mu:.3f}$" + (rf", $\sigma={std:.3f}$" if std > 0 else "")
        ax.text(
            0.98, 0.95, stats_text,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.9) if fg=="black"
                 else dict(boxstyle="round,pad=0.25", facecolor="#111827", edgecolor="none", alpha=0.8),
            color=fg
        )

        # Cosmetics
        ax.set_xlabel("AUC")
        ax.set_ylabel("Frequency")
        if args.xlim:
            ax.set_xlim(args.xlim[0], args.xlim[1])
        ax.grid(axis="y")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.tight_layout()

        # Save PNG + PDF
        out_root = args.output.replace(".png", f"_{lang}")
        os.makedirs(os.path.dirname(out_root), exist_ok=True)
        fig.savefig(out_root + ".png", bbox_inches="tight")
        fig.savefig(out_root + ".pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"→ saved {out_root}.png / .pdf")

if __name__ == "__main__":
    main()
