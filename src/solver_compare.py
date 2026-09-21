import argparse
import os
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

SOLVER_STYLES = {
    "bo-de":  {"color": "#B23A48", "marker": "s"},
    "dehb":   {"color": "#1E8449", "marker": "o"},
    "bohb":   {"color": "#1F4FD1", "marker": "^"},
    "rs":     {"color": "#7F7F7F", "marker": "D"},
}
DEFAULT_STYLE_CYCLE = ["#B23A48", "#1E8449", "#1F4FD1", "#7F7F7F", "#CA6F1E", "#8E44AD"]
DEFAULT_MARKER_CYCLE = ["s", "o", "^", "D", "v", "P"]


def infer_label(filename):
    base = os.path.basename(filename).lower()
    if "bode" in base or "bo_de" in base or "bo-de" in base:
        return "BO-DE"
    if "dehb" in base:
        return "DEHB"
    if "bohb" in base:
        return "BOHB"
    if re.search(r"(^|[_\-])rs([_\-]|$)", base) or "random" in base:
        return "RS"
    return os.path.splitext(os.path.basename(filename))[0]


def style_for(label, idx):
    key = label.lower()
    if key in SOLVER_STYLES:
        return SOLVER_STYLES[key]
    return {
        "color": DEFAULT_STYLE_CYCLE[idx % len(DEFAULT_STYLE_CYCLE)],
        "marker": DEFAULT_MARKER_CYCLE[idx % len(DEFAULT_MARKER_CYCLE)],
    }


def load_and_process(filepath, time_col="trial_duration_sec",
                      value_col="val_mae", order_col="trial"):
    df = pd.read_excel(filepath)
    if order_col not in df.columns:
        raise ValueError(f"'{order_col}' column not found in {filepath}")
    if time_col not in df.columns:
        raise ValueError(f"'{time_col}' column not found in {filepath}")
    if value_col not in df.columns:
        raise ValueError(f"'{value_col}' column not found in {filepath}")


    trial_numeric = pd.to_numeric(df[order_col], errors="coerce")
    n_dropped = trial_numeric.isna().sum()
    if n_dropped > 0:
        dropped_vals = df.loc[trial_numeric.isna(), order_col].tolist()
        print(f"  Note: dropped {n_dropped} non-trial row(s) from "
              f"{os.path.basename(filepath)}: {dropped_vals}")
    df = df.loc[trial_numeric.notna()].copy()
    df[order_col] = trial_numeric[trial_numeric.notna()]

    df = df.sort_values(order_col).reset_index(drop=True)
    df["cum_time_sec"] = df[time_col].cumsum()
    df["best_so_far"] = df[value_col].cummin()
    return df


def plot_anytime_performance(files, labels, title, output_path,
                              value_col="val_mae", ylabel="Best validation MAE found so far"):
    fig, ax = plt.subplots(figsize=(9, 6))

    summary = []
    for idx, (filepath, label) in enumerate(zip(files, labels)):
        df = load_and_process(filepath, value_col=value_col)
        style = style_for(label, idx)
        x = df["cum_time_sec"] / 3600  # hours

        ax.step(x, df["best_so_far"], where="post",
                color=style["color"], marker=style["marker"], markersize=4,
                linewidth=2.2, label=label)

        summary.append({
            "label": label,
            "n_trials": len(df),
            "total_hours": x.iloc[-1],
            "final_best": df["best_so_far"].iloc[-1],
        })

    ax.set_xlabel("Cumulative training time (hours)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"Anytime Performance\n{title}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    if output_path.lower().endswith(".png"):
        fig.savefig(output_path[:-4] + ".pdf", bbox_inches="tight", facecolor="white")

    print(f"Saved: {output_path}")
    print()
    print(f"{'Solver':<10} {'Trials':>7} {'Total (h)':>10} {'Final best':>12}")
    for s in summary:
        print(f"{s['label']:<10} {s['n_trials']:>7} {s['total_hours']:>10.2f} {s['final_best']:>12.4f}")

    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resdir", type=Path, default=Path("."),
                    help="Base directory the --xlsx entries are resolved against.")
    ap.add_argument("--outpath", type=Path, default=Path("plots/"),
                    help="Directory to save the anytime-performance plot to.")
    ap.add_argument("--title", type=str, required=True,
                    help="Plot title, e.g. 'Graph WaveNet on PEMS-BAY'.")
    ap.add_argument("--xlsx", nargs="+", required=True,
                    help="xlsx result files to compare, resolved against --resdir.")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Solver label per --xlsx file. If omitted, inferred from filenames.")
    ap.add_argument("--value_col", type=str, default="val_mae")
    args = ap.parse_args()

    files = [str(args.resdir / f) for f in args.xlsx]
    labels = args.labels if args.labels is not None else [infer_label(f) for f in files]

    if len(labels) != len(files):
        raise ValueError("--labels must have the same number of entries as --xlsx")

    os.makedirs(args.outpath, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", args.title.lower()).strip("_")
    output_path = args.outpath / f"anytime_{slug}.png"

    plot_anytime_performance(
        files=files,
        labels=labels,
        title=args.title,
        output_path=str(output_path),
        value_col=args.value_col,
    )


if __name__ == "__main__":
    main()