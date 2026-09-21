import argparse
import re
import sys
from pathlib import Path
import os

import pandas as pd

FNAME_RE = re.compile(
    r"^(?:\d+_)?(?P<model>[A-Za-z0-9]+)_(?P<dataset>[A-Za-z0-9]+)_"
    r"(?P<method>BODE|RS)_+(?P<date>\d+)\.xlsx$",
    re.IGNORECASE,
)


def parse_filename(path: Path):
    m = FNAME_RE.match(path.name)
    if not m:
        return None
    return m.group("model").lower(), m.group("dataset").lower(), m.group("method").upper()


def value_column(df: pd.DataFrame) -> str:
    if "val_mae" in df.columns:
        return "val_mae"
    if "best_val_mae" in df.columns:
        return "best_val_mae"
    raise KeyError("Neither 'val_mae' nor 'best_val_mae' found in columns")


def summarize_run(path: Path) -> dict:
    df = pd.read_excel(path)
    col = value_column(df)
    vals = df[col].astype(float)

    best_idx = vals.idxmin()
    best_val = vals.loc[best_idx]
    trial_at_best = df.loc[best_idx, "trial"] if "trial" in df.columns else best_idx

    q1, q3 = vals.quantile([0.25, 0.75])

    dur = df["trial_duration_sec"].astype(float) if "trial_duration_sec" in df.columns else pd.Series(dtype=float)
    total_elapsed = (
        df["elapsed_since_start_sec"].astype(float).max()
        if "elapsed_since_start_sec" in df.columns
        else float("nan")
    )

    return {
        "n_trials": len(df),
        "mean_val_mae": vals.mean(),
        "std_val_mae": vals.std(),
        "median_val_mae": vals.median(),
        "iqr_val_mae": q3 - q1,
        "best_val_mae": best_val,
        "trial_at_best": trial_at_best,
        "mean_trial_duration_sec": dur.mean() if not dur.empty else float("nan"),
        "std_trial_duration_sec": dur.std() if not dur.empty else float("nan"),
        "total_elapsed_sec": total_elapsed,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resdir", type=Path, default=Path("search_results/") ,help="Directory containing the .xlsx result files")
    ap.add_argument("--outpath", type=Path, default=Path("plots/"),
                    help="Directory to save search_comparison_summary.csv to.")
    args = ap.parse_args()
    os.makedirs(args.outpath, exist_ok=True)
    csv_path = args.outpath / "search_comparison_summary.csv"
    rows = []
    skipped = []
    for path in sorted(args.resdir.glob("*.xlsx")):
        parsed = parse_filename(path)
        if parsed is None:
            skipped.append(path.name)
            continue
        model, dataset, method = parsed
        try:
            stats = summarize_run(path)
        except Exception as e:
            print(f"WARNING: failed to summarize {path.name}: {e}", file=sys.stderr)
            continue
        stats.update({"model": model, "dataset": dataset, "method": method, "file": path.name})
        rows.append(stats)

    if skipped:
        print("Skipped (filename didn't match expected pattern):", file=sys.stderr)
        for name in skipped:
            print(f"  {name}", file=sys.stderr)

    if not rows:
        print("No matching result files found.", file=sys.stderr)
        sys.exit(1)

    summary = pd.DataFrame(rows)

    col_order = [
        "model", "dataset", "method", "n_trials",
        "mean_val_mae", "std_val_mae", "median_val_mae", "iqr_val_mae",
        "best_val_mae", "trial_at_best",
        "mean_trial_duration_sec", "std_trial_duration_sec", "total_elapsed_sec",
        "file",
    ]
    summary = summary[col_order].sort_values(["model", "dataset", "method"]).reset_index(drop=True)

    summary.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}\n")

    display_cols = [
        "method", "n_trials", "mean_val_mae", "std_val_mae", "median_val_mae",
        "iqr_val_mae", "best_val_mae", "trial_at_best",
        "mean_trial_duration_sec", "std_trial_duration_sec",
    ]
    for (model, dataset), group in summary.groupby(["model", "dataset"]):
        print(f"=== {model} / {dataset} ===")
        print(group[display_cols].round(4).to_string(index=False))
        print()


if __name__ == "__main__":
    main()