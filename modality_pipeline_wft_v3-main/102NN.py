import pandas as pd
from pathlib import Path


def _read_test_csv_with_inferred_prefix(csv_path):
    """
    Read a CSV. If it lacks 'weight_prefix', add a column
    with an inferred prefix taken from the file stem (unique per file).
    """
    csv_path = Path(csv_path)
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, sep="\t")

    if "weight_prefix" not in df.columns:
        inferred = csv_path.stem  # safe unique identifier per-file
        df["weight_prefix"] = inferred
    # ensure sample_index exists (if not, create a placeholder)
    if "sample_index" not in df.columns:
        df.insert(0, "sample_index", range(len(df)))
    return df


def compute_experiment_metrics(experiment_dir):
    """
    Given an experiment folder containing a 'metrics/' subfolder with *_testing_.csv files,
    compute per-weight average metrics, avoiding any leakage when 'weight_prefix' is absent.
    Returns list[dict] with keys: experiment_name, weight_prefix, average_<metric>.
    """
    exp_path = Path(experiment_dir)
    metrics_dir = exp_path / "metrics"
    all_files = sorted(metrics_dir.glob("*_testing_.csv"))

    if not all_files:
        raise FileNotFoundError(f"No files ending with '_testing_.csv' found in {metrics_dir}")

    dfs = []
    for f in all_files:
        df = _read_test_csv_with_inferred_prefix(f)
        # Tag rows with the source filename too (optional but useful for tracing)
        df["_source_file"] = f.name
        dfs.append(df)

    big_df = pd.concat(dfs, ignore_index=True)

    # Required id columns (we ensured both exist above)
    id_cols = ["sample_index", "weight_prefix"]
    metric_cols = [c for c in big_df.columns if c not in id_cols + ["_source_file"]]

    # Convert metrics to numeric; non-numeric -> NaN
    for c in metric_cols:
        big_df[c] = pd.to_numeric(big_df[c], errors="coerce")

    summaries = []
    # Group by weight_prefix (this prevents mixing different inferred prefixes)
    for prefix, sub_df in big_df.groupby("weight_prefix", dropna=False):
        prefix_str = str(prefix) if pd.notna(prefix) else "none"
        mean_vals = sub_df[metric_cols].mean().to_dict()
        # Build summary with 'average_<metric>' keys
        summary = {
            "experiment_name": exp_path.name,
            "weight_prefix": prefix_str,
        }
        for k, v in mean_vals.items():
            summary[f"average_{k}"] = float(f"{v:.6f}") if pd.notna(v) else None
        summaries.append(summary)

    return summaries


def merge_all_experiments(experiment_dirs, ID):
    """
    Process multiple experiment root directories (each containing a 'metrics' subfolder).
    Produces a single overall_results.csv at metrics/{ID}/overall_results.csv with rows:
        experiment_name, weight_prefix, average_<metric1>, average_<metric2>, ...
    """
    all_summaries = []
    for exp_dir in experiment_dirs:
        print(f"Processing: {exp_dir}")
        summaries = compute_experiment_metrics(exp_dir)
        all_summaries.extend(summaries)

    if not all_summaries:
        print("No summaries found; skipping overall_results.csv")
        return

    out_dir = Path(f"metrics/{ID}")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_overall = pd.DataFrame(all_summaries)

    # Order columns: experiment_name, weight_prefix, then sorted metric columns
    metric_cols = sorted([c for c in df_overall.columns if c not in ["experiment_name", "weight_prefix"]])
    cols = ["experiment_name", "weight_prefix"] + metric_cols
    df_overall = df_overall[cols]

    out_csv = out_dir / "overall_results.csv"
    df_overall.to_csv(out_csv, index=False)
    print(f"Overall summary written to: {out_csv}")
    return out_csv

# Example usage:
paths = [
    "logs/VNET_L1",
    "logs/VNET_L1_GAN",
    "logs/VNET_L1_Percept",
    
    "logs/251112_UNET_3D_L1_32_nearest_150",
    "logs/251112_UNET_3D_L1_32_trilinear_150",
    "logs/251112_UNET_3D_L1_32_conv_150",
    
    "logs/251112_UNET_3D_L1_32_nearest",
    "logs/251112_UNET_3D_L1_32_trilinear",
    "logs/251112_UNET_3D_L1_32_conv",
    

    
    "logs/151112_SEResidual_3D_L1_32_conv",
    
    

    "logs/30_ESAU_3D_L1_32_nearest_150",
    "logs/30_ESAU_3D_L1_32_nearest",
    
    "logs/30_ESAU_3D_L1_32_conv_150",
    "logs/30_ESAU_3D_L1_32_conv",
    
    "logs/30_ESAU_3D_L1_32_trilinear_150",
    "logs/30_ESAU_3D_L1_32_trilinear",
    "logs/251201_ESAU_3D_L1_48_trilinear",
    "logs/251201_ESAU_3D_L1_64_trilinear",

    
    "logs/30_ESAU_3D_L1_32_trilinear",
    "logs/30_ESAU_3D_L1_32_trilinear_150",
    "logs/251205_ESAU_TL_3D_L1_64_trilinear",
    "logs/251205_ESAU_TL_3D_L1_64_trilinear_TOP1",
    "logs/251205_ESAU_TL_3D_L1_96_trilinear",
    "logs/251205_ESAU_TL_3D_L1_128_trilinear",
]
ID = "ISBI_Final0"
merge_all_experiments(paths, ID)
