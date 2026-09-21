import argparse
import glob
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

MODELS = ["stgcn", "graphwavenet", "agcrn"]
METHODS = ["rs", "bode"]

MODEL_LABELS = {
    "stgcn": "STGCN",
    "graphwavenet": "Graph WaveNet",
    "agcrn": "AGCRN",
}
METHOD_LABELS = {
    "rs": "Random Search",
    "bode": "BO-DE",
}

METHOD_COLORS = {
    "rs": "#7F7F7F",
    "bode": "#D62728",
}
METHOD_MARKERS = {
    "rs": "o",
    "bode": "s",
}


def parse_filename(path):
    base = os.path.basename(path)
    base = re.sub(r"^\d+_", "", base)
    base = base.rsplit(".", 1)[0]
    parts = base.split("_")
    if len(parts) < 4:
        return None
    model, dataset, method = parts[0].lower(), parts[1].lower(), parts[2].lower()
    if model not in MODELS or method not in METHODS:
        return None
    return model, dataset, method


def load_all(input_dir):
    records = {}
    for path in glob.glob(os.path.join(input_dir, "*.xlsx")):
        parsed = parse_filename(path)
        if parsed is None:
            continue
        model, dataset, method = parsed
        df = pd.read_excel(path)
        if "trial" not in df.columns or "best_val_mae" not in df.columns:
            continue
        records.setdefault(dataset, []).append((model, method, df))
    return records


def plot_dataset(dataset, entries, output_dir, mark_bode_start=True):
    models_present = sorted({model for model, _, _ in entries})
    n = len(models_present)

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.5), sharey=True)
    if n == 1:
        axes = [axes]

    y_all = pd.concat([df["best_val_mae"] for _, _, df in entries])
    y_min, y_max = y_all.min(), y_all.max()
    pad = (y_max - y_min) * 0.08
    ylim = (y_min - pad, y_max + pad)

    for ax, model in zip(axes, models_present):
        model_entries = [(m, meth, df) for m, meth, df in entries if m == model]

        for _, method, df in sorted(model_entries, key=lambda e: e[1]):
            df = df.sort_values("trial")
            color = METHOD_COLORS.get(method, "#333333")
            marker = METHOD_MARKERS.get(method, "o")
            label = METHOD_LABELS.get(method, method)

            ax.plot(df["trial"], df["best_val_mae"], color=color, linestyle="-",
                    marker=marker, markersize=5, linewidth=1.6, label=label, zorder=2)

            # mark this line's single best (lowest) val_mae point
            min_idx = df["best_val_mae"].idxmin()
            min_trial = df.loc[min_idx, "trial"]
            min_val = df.loc[min_idx, "best_val_mae"]
            ax.plot(min_trial, min_val, marker="*", markersize=10,
                    markerfacecolor=color, markeredgecolor="black",
                    markeredgewidth=1.0, linestyle="none", zorder=5)

            if mark_bode_start and "phase" in df.columns:
                init_rows = df[df["phase"] == "init"]
                if not init_rows.empty:
                    n_init = len(init_rows)
                    if n_init < len(df):
                        x = df["trial"].iloc[n_init]
                        ax.axvline(x=x - 0.5, color=color, linestyle=":",
                                   linewidth=1.2, alpha=0.5, zorder=1)
        if MODEL_LABELS.get(model, model).lower() == 'stgcn':
            ax.set_title("STCN", fontsize=13, fontweight="bold")
        else:
            ax.set_title(MODEL_LABELS.get(model, model), fontsize=13, fontweight="bold")
        ax.set_xlabel("Trial", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(*ylim)
        ax.legend(fontsize=9, loc="best", frameon=True)

    axes[0].set_ylabel("Validation MAE", fontsize=12)
    fig.suptitle(f"Trial vs. Validation MAE by Model \u2010 {dataset.upper()}",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_path = os.path.join(output_dir, f"trial_vs_mae_{dataset}_bymodel.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path} ({len(entries)} lines across {n} panels)")

def plotting(res_dir="search_results/", out_dir="plots/"):
    os.makedirs(out_dir, exist_ok=True)
    records = load_all(res_dir)

    if not records:
        print("No matching result files found.")
        return

    for dataset, entries in records.items():
        plot_dataset(dataset, entries, out_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Plot trial vs. validation MAE from search-result xlsx files."
    )
    parser.add_argument("--resdir", type=str, default="search_results/",
                        help="Directory containing the search-result .xlsx files.")
    parser.add_argument("--outpath", type=str, default="plots/",
                        help="Directory to save the generated plots to.")
    args = parser.parse_args()

    plotting(res_dir=args.resdir, out_dir=args.outpath)


if __name__ == "__main__":
    main()