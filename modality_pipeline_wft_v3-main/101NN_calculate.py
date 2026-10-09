import pandas as pd
from pathlib import Path


def compute_experiment_metrics(experiment_dir):
    """
    Given an experiment folder containing a 'metrics/' subfolder with *_testing_.csv files,
    compute per-weight average metrics.

    Returns
    -------
    list[dict]
        Each dict contains: experiment_name, weight_prefix, <metric>: <average>
    """
    exp_path = Path(experiment_dir)
    metrics_dir = exp_path / "metrics"
    all_files = sorted(metrics_dir.glob("*_testing_.csv"))

    if not all_files:
        raise FileNotFoundError(f"No files ending with '_testing_.csv' found in {metrics_dir}")

    # Combine all test CSVs for this experiment
    dfs = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
        except Exception:
            df = pd.read_csv(f, sep="\t")
        dfs.append(df)

    big_df = pd.concat(dfs, ignore_index=True)

    # Identify columns
    id_cols = ["sample_index", "weight_prefix"]
    metric_cols = [c for c in big_df.columns if c not in id_cols]
    for c in metric_cols:
        big_df[c] = pd.to_numeric(big_df[c], errors="coerce")

    summaries = []
    for prefix, sub_df in big_df.groupby("weight_prefix", dropna=False):
        prefix_str = str(prefix) if pd.notna(prefix) else "none"
        mean_vals = sub_df[metric_cols].mean().to_dict()
        summary = {
            "experiment_name": exp_path.name,
            "weight_prefix": prefix_str,
            **{f"average_{k}": float(f"{v:.6f}") for k, v in mean_vals.items()},
        }
        summaries.append(summary)

    return summaries


def merge_all_experiments(experiment_dirs, ID):
    """
    Process multiple experiment folders, each containing a 'metrics' subfolder.
    Create one overall_results.csv with averaged metrics per (experiment, weight_prefix).
    """
    all_summaries = []
    for exp_dir in experiment_dirs:
        print(f"--- Processing: {exp_dir} ---")
        summaries = compute_experiment_metrics(exp_dir)
        all_summaries.extend(summaries)

    if not all_summaries:
        print("⚠️ No summaries found — no file created.")
        return

    out_dir = Path(f"metrics/{ID}")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_overall = pd.DataFrame(all_summaries)
    # Order columns
    cols = ["experiment_name", "weight_prefix"] + [c for c in df_overall.columns if c not in ["experiment_name", "weight_prefix"]]
    df_overall = df_overall[cols]

    out_csv = out_dir / "overall_results.csv"
    df_overall.to_csv(out_csv, index=False)
    print(f"🏁 Created overall summary file → {out_csv}")


# === Example usage ===
# experiment_dirs = [
#     "experiment_A",
#     "experiment_B",
#     "experiment_C",
# ]
# ID = "2025_11_14"
# merge_all_experiments(experiment_dirs, ID)

# # Example usage:
# experiment_metric_dirs = [
#     "exp1/metrics",
#     "exp2/metrics",
#     "exp3/metrics",
# ]
# ID = "2025_11_14"
# merge_all_experiments(experiment_metric_dirs, ID)

# Example usage:
paths = [
#    "logs/22_ESAU_3D_Conv_Bounded_Full",
   "logs/22_ESAU_3D_Conv_Bounded_Tanh",
   "logs/22_ESAU_3D_Nearsest_Bounded",
   "logs/22_ESAU_3D_Nearsest_Tanh",
   "logs/22_ESAU_3D_Trilinear_Bounded",
   "logs/22_ESAU_3D_Trilinear_Tanh",
   "logs/22_ESAU_RA_3D_Conv_Bounded_Tanh",
   "logs/23_ESAU_RA_1_3D_Trilinear_Tanh",
   "logs/23_ESAU_RA_3D_Nearsest_Tanh",
   "logs/ESAU_3D_Conv_Bounded_Full",
   "logs/ESAU_3D_Nearsest_Bounded_Full",
   "logs/ESAU_3D_Trilinear_Bounded_Full",
#    "logs/30_CCallESAU_3D_L1_32_CONV",
#    "logs/30_CCallESAU_3D_L1_32_nearest",
#    "logs/30_ESAU_3D_L1_8_nearest",
#    "logs/30_ESAU_3D_L1_16_nearest",
#    "logs/30_ESAU_3D_L1_32_conv",
#    "logs/30_ESAU_3D_L1_32_nearest",
#    "logs/30_ESAU_3D_L1_32_trilinear"
   "logs/251112_UNET_3D_L1_32_conv",
   "logs/251112_UNET_3D_L1_32_nearest",
   "logs/251112_UNET_3D_L1_32_trilinear",
   "logs/VNET_L1",
   "logs/VNET_L1_GAN",
   "logs/VNET_L1_Percept"
]
ID = "ISBI_2"
merge_all_experiments(paths, ID)
