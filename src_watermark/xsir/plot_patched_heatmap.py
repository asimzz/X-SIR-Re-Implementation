#!/usr/bin/env python3
# auc_heatmap_acl_colorful.py
import os, json, argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, auc

# ----------------------------- Data utils -----------------------------
def load_zscores(path):
    """Read a .jsonl where each line is either a float or {'z_score': ...}."""
    with open(path) as f:
        out = []
        for line in f:
            j = json.loads(line)
            out.append(float(j["z_score"] if isinstance(j, dict) and "z_score" in j else j))
    return out

def compute_auc(hum_path, atk_path):
    h = load_zscores(hum_path)
    a = load_zscores(atk_path)
    y_true  = [0]*len(h) + [1]*len(a)
    y_score =  h        +  a
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)

# ----------------------------- Style helpers -----------------------------
def get_colormap(name: str):
    """
    Saturated palettes for eye-catchy heatmaps.
    - reds_deep: deeper low end (no white). Great if you want a 'Reds' vibe without pale cells.
    - turbo / inferno / magma: perceptually uniform, vivid.
    """
    name = (name or "reds_deep").lower()
    if name == "reds_deep":
        # Darker low end than ColorBrewer Reds (avoid near-white)
        stops = ["#f3b3a0", "#f07b5a", "#e34a33", "#b30000", "#67000d"]
        return LinearSegmentedColormap.from_list("reds_deep", stops, N=256)
    if name in plt.colormaps:
        return plt.colormaps[name]
    # fallback
    return plt.colormaps["turbo"]

def nice_rc(dpi=300, dark=False):
    """ACL-like typography without requiring system LaTeX."""
    bg = "#0d0f12" if dark else "white"
    fg = "white" if dark else "black"
    mpl.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",   # Computer Modern look
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
        "figure.facecolor": bg,
        "axes.facecolor": bg,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "pdf.fonttype": 42, "ps.fonttype": 42
    })
    return bg, fg

# ----------------------------- Plot -----------------------------
def plot_auc_heatmap(
    mat, seeds, model_abbr, tgt_lang, output_path,
    annot="diag", decimals=3, percent=False,
    vmin=0.50, vmax=1.00, palette="reds_deep",
    auto_vmin=None, auto_vmax=None, dark=False
):
    """
    annot: 'none' | 'diag' | 'all'
    percent: annotate cells as % of [vmin, vmax].
    auto_vmin/auto_vmax: percentiles (e.g., 5 / 99) to override vmin/vmax from data.
    """
    _, fg = nice_rc(dark=dark)
    cmap = get_colormap(palette)

    # Mask NaNs so they render cleanly
    masked = np.ma.masked_invalid(mat)

    # Optional auto scaling from data percentiles
    if auto_vmin is not None:
        vmin = float(np.nanpercentile(mat, auto_vmin))
    if auto_vmax is not None:
        vmax = float(np.nanpercentile(mat, auto_vmax))
    # guard in case vmin==vmax
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = 0.5, 1.0

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(masked, origin="upper", vmin=vmin, vmax=vmax,
                   cmap=cmap, interpolation="nearest")

    # ticks
    n = len(seeds)
    ax.set_xticks(np.arange(n), labels=[str(s) for s in seeds])
    ax.set_yticks(np.arange(n), labels=[str(s) for s in seeds])
    ax.set_xlabel("Patched seed")
    ax.set_ylabel("Reference seed")

    # subtle grid (lighter so the color pops more)
    gridc = (0, 0, 0, 0.08) if fg == "black" else (1, 1, 1, 0.14)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color=gridc, linewidth=0.8)
    ax.tick_params(which="both", length=0)

    # annotations
    def format_val(v):
        if percent:
            p = np.clip((v - vmin) / max(1e-9, (vmax - vmin)), 0, 1)
            return f"{int(round(p*100))}%"
        return f"{v:.{decimals}f}"

    def text_color(v):
        t = (v - vmin) / max(1e-9, (vmax - vmin))
        return "white" if t > 0.55 else "black"

    if annot in ("diag", "all"):
        for i in range(n):
            for j in range(n):
                if np.isnan(mat[i, j]):
                    continue
                if annot == "diag" and i != j:
                    continue
                ax.text(j, i, format_val(mat[i, j]),
                        ha="center", va="center",
                        fontsize=9,
                        color=text_color(mat[i, j]))

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
    cbar.set_label("AUC", rotation=90, labelpad=10, color=fg)
    for tick in cbar.ax.get_yticklabels():
        tick.set_color(fg)
    cbar.outline.set_edgecolor(fg)

    fig.tight_layout()

    # Save PNG + PDF
    root, ext = os.path.splitext(output_path)
    png_path = root + ".png" if ext.lower() not in (".png", ".pdf") else (root + ".png")
    pdf_path = root + ".pdf" if ext.lower() not in (".png", ".pdf") else (root + ".pdf")
    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}\nSaved: {pdf_path}")

# ----------------------------- CLI -----------------------------
def main():
    p = argparse.ArgumentParser(description="ACL-styled, colorful AUC heatmap.")
    p.add_argument("--model_abbr", required=True)
    p.add_argument("--base_dir",   required=True, help="gen/<model_abbr>/xsir/seed_<ref_seed>/…")
    p.add_argument("--seeds",      nargs="+", required=True, help="List of seeds, e.g., 0 1 2 ...")
    p.add_argument("--tgt_lang",   required=True, help="Target language code, e.g., it")
    p.add_argument("--output",     required=True, help="Output path (no ext or .png/.pdf)")
    p.add_argument("--annot",      choices=["none","diag","all"], default="none")
    p.add_argument("--decimals",   type=int, default=3)
    p.add_argument("--vmin",       type=float, default=0.50)
    p.add_argument("--vmax",       type=float, default=1.00)
    p.add_argument("--auto_vmin",  type=float, default=None, help="Percentile for vmin (e.g., 5)")
    p.add_argument("--auto_vmax",  type=float, default=None, help="Percentile for vmax (e.g., 99)")
    p.add_argument("--percent",    action="store_true")
    p.add_argument("--palette",    default="reds_deep",
                   choices=["reds_deep","turbo","inferno","magma"])
    p.add_argument("--dark",       action="store_true")
    args = p.parse_args()

    seeds = sorted(int(s) for s in args.seeds)
    n = len(seeds)
    mat = np.full((n, n), np.nan, dtype=float)

    # Build matrix (rows=ref seed, cols=patched seed)
    for i, ref in enumerate(seeds):
        for j, patch in enumerate(seeds):
            subdir = os.path.join(args.base_dir, args.model_abbr, "xsir", f"seed_{ref}")
            hum_fn = "mc4.en.hum.z_score.jsonl"
            atk_fn = (
                f"mc4.en-{args.tgt_lang}.mod.z_score.jsonl"
                if ref == patch else
                f"mc4.en-{args.tgt_lang}-seed-{patch}.mod.z_score.jsonl"
            )
            hum_path = os.path.join(subdir, hum_fn)
            atk_path = os.path.join(subdir, atk_fn)

            if not (os.path.exists(hum_path) and os.path.exists(atk_path)):
                print(f"[WARN] missing files for ref={ref}, patch={patch}")
                continue
            try:
                mat[i, j] = compute_auc(hum_path, atk_path)
            except Exception as e:
                print(f"[ERROR] ref={ref}, patch={patch}: {e}")

    plot_auc_heatmap(
        mat=mat,
        seeds=seeds,
        model_abbr=args.model_abbr,
        tgt_lang=args.tgt_lang,
        output_path=args.output,
        annot=args.annot,
        decimals=args.decimals,
        percent=args.percent,
        vmin=args.vmin,
        vmax=args.vmax,
        palette=args.palette,
        auto_vmin=args.auto_vmin,
        auto_vmax=args.auto_vmax,
        dark=args.dark
    )

if __name__ == "__main__":
    main()
