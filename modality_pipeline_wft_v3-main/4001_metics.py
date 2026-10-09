import pandas as pd
from pathlib import Path
import os
import re


def _read_csv_flexible(f):
    try:
        return pd.read_csv(f)
    except Exception:
        return pd.read_csv(f, sep="\t")


def _epoch_from_prefix(prefix):
    """Extract the epoch number from strings like 'saveEvery_epoch_epoch=100'."""
    s = str(prefix)
    m = re.search(r"epoch\s*=?\s*(\d+)", s)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", s)
    return int(nums[-1]) if nums else -1


ID_COLS = ["sample_index", "weight_prefix"]
ORDER_PRIORITY = ["mse", "mae", "psnr", "ssim_3d", "ssim_2d"]


def _order_metrics(metric_cols):
    ordered = []
    for key in ORDER_PRIORITY:
        ordered.extend([c for c in metric_cols if key.lower() in c.lower()])
    other = [c for c in metric_cols if c not in ordered]
    return ordered + other


def merge_testing_metrics(experiments_dir, out_csv):
    """
    Merge all *_testing metric files into a single CSV, reorder metric columns,
    and add average/std rows. Returns a numeric dataframe (with a '__split__'
    column identifying the source file) for downstream cross-experiment stats.
    """
    experiments_dir = Path(experiments_dir)
    all_files = sorted(experiments_dir.glob("*_testing_.csv"))
    if not all_files:
        raise FileNotFoundError(f"No files ending with '_testing' found in {experiments_dir}")

    dfs = []
    for f in all_files:
        df = _read_csv_flexible(f)
        df["__split__"] = f.stem  # each source file = one split
        dfs.append(df)
    big_df = pd.concat(dfs, ignore_index=True)

    # Identify id vs metric columns
    metric_cols = [c for c in big_df.columns if c not in ID_COLS + ["__split__"]]

    # Convert to numeric
    for c in metric_cols:
        big_df[c] = pd.to_numeric(big_df[c], errors="coerce")

    # Keep a numeric copy (with split info) for cross-experiment aggregation
    numeric_df = big_df.copy()

    # ---- Compute stats ----
    mean_vals = big_df[metric_cols].mean().tolist()
    std_vals = big_df[metric_cols].std().tolist()
    mean_row = ["AVERAGE", ""] + [f"{v:0.6f}" for v in mean_vals]
    std_row = ["STD", ""] + [f"{v:0.6f}" for v in std_vals]

    # Format per-scan rows
    out_df = big_df.drop(columns="__split__").copy()
    for c in metric_cols:
        out_df[c] = out_df[c].map(lambda v: f"{v:0.6f}" if pd.notna(v) else "")

    # Append summary rows
    summary_df = pd.DataFrame([mean_row, std_row], columns=out_df.columns)
    final_df = pd.concat([out_df, summary_df], ignore_index=True)

    # ---- Reorder columns ----
    ordered_metrics = _order_metrics(metric_cols)
    final_cols = ID_COLS + ordered_metrics
    final_df = final_df[final_cols]

    # Save
    final_df.to_csv(out_csv, index=False)
    print(f"✅ Merged {len(all_files)} files, reordered metrics, added averages & std → {out_csv}")

    return numeric_df, metric_cols


def build_combined_summary(per_experiment, out_csv):
    """
    per_experiment: dict of experiment_name -> (numeric_df, metric_cols)
    For each experiment, each split is evaluated on multiple weights (epochs).
    Take the highest-epoch weight per split as that split's representative,
    average its samples, then average across splits to get one row per
    experiment. Also record the epoch used for each split.
    """
    rows = []
    all_metric_cols = []

    for exp_name, (df, metric_cols) in per_experiment.items():
        df = df.copy()
        df["__epoch__"] = df["weight_prefix"].map(_epoch_from_prefix)

        split_reps = []
        epochs_used = []
        for split_name in sorted(df["__split__"].unique()):
            sdf = df[df["__split__"] == split_name]
            max_ep = sdf["__epoch__"].max()
            rep = sdf[sdf["__epoch__"] == max_ep]
            split_reps.append(rep[metric_cols].mean())
            epochs_used.append(int(max_ep))

        split_reps = pd.DataFrame(split_reps)
        exp_avg = split_reps.mean()

        row = {
            "Experiment Name": exp_name,
            "Number_of_Splits_used": len(split_reps),
            "Epochs_used_per_split": epochs_used,  # [epochUsedSplit1, epochUsedSplit2, ...]
        }
        for c in metric_cols:
            row[f"Average_{c}"] = round(float(exp_avg[c]), 6)
        rows.append(row)

        for c in metric_cols:
            if c not in all_metric_cols:
                all_metric_cols.append(c)

    combined = pd.DataFrame(rows)
    ordered_cols = (
        ["Experiment Name", "Number_of_Splits_used", "Epochs_used_per_split"]
        + [f"Average_{c}" for c in _order_metrics(all_metric_cols)]
    )
    combined = combined[[c for c in ordered_cols if c in combined.columns]]
    combined.to_csv(out_csv, index=False)
    print(f"✅ Combined summary across {len(rows)} experiments → {out_csv}")
    return combined


paths = [
    "logs/MRIxFields_T1W0.1Tto7T_3DESAU_L1",
    "logs/MRIxFields_SE_T1w_3Tto7T_L1Wavelet",
    "logs/MRIxFields_SE_T1w_3Tto7T_L1WaveletDiffusionLoss",
    "logs/MRIxFields_T1w_3Tto7T",
    "logs/MRIxFields_T1W0.1Tto7T_3DESAU_L1",
    "logs/MRIxFields_T1W1.5Tto7T_3DESAU_L1",
    "logs/MRIxFields_T1W3Tto7T_3DESAU_L1",
    "logs/MRIxFields_T1W5Tto7T_3DESAU_L1",
    "logs/MRIxFields_T1WL1PretrainGIN_FineTune",
    "logs/MRIxFields_T1WL1PretrainGIN_FineTune_L1GIN",
    "logs/MRIxFields_T1WL1PretrainGIN_FineTune_L1Wavelet",
    "logs/MRIxFields_T1WL1PretrainGIN_FineTune_L1Wavelet0.5",
    "logs/MRIxFields_T1WL1PretrainGIN_FineTune_L1Wavelet1.0",
    "logs/MRIxFields_T1WL1PretrainGIN_FineTune_L1WaveletLvl2",
    "logs/MRIxFields_T1WL1PretrainGIN_FineTune_L1WaveletLvl2_MEAN",
    "logs/MRIxFields_T1WL1PretrainGIN_FineTune_L1WaveletLvl2_minmax",
    "logs/MRIxFields_T1WL1PretrainGIN_FineTune_L1WaveletSynthSeg",
    "logs/T1W_0.1_7T_ESAU_Mask_Downsample_aug",
    "logs/T1W_0.1_7T_ESAU_Mask_Downsample_auglrx2",
    "logs/T1W_0.1_7T_ESAU_Mask_Downsample_noaug",
    "logs/T1W_0.1_7T_ESAU_Mask_Downsample_noaug_lrx2",
    "logs/T1W_1.5_7T_ESAU_Mask_Downsample_auglrx0.5",
    "logs/T1W_1.5_7T_ESAU_Mask_Downsample_auglrx2",
    "logs/T1W_3_7T_ESAU_Mask_Downsample_auglrx0.5",
    "logs/T1W_3_7T_ESAU_Mask_Downsample_auglrx2",
    "logs/T1W_5_7T_ESAU_Mask_Downsample_auglrx0.5",
    "logs/T1W_5_7T_ESAU_Mask_Downsample_auglrx2",
        "logs/T1W_FINETUNE_L1_HFWavelet_GIN_MASK_minmaxSource_MAE_style_FinetuneAugsEnabled",
    "logs/T1W_FINETUNE_L1_HFWavelet_GIN_MASK_minmaxSource_MAE_style_NoFinetuneAugs",
    "logs/T1W_FINETUNE_L1_HFWavelet_GIN_MASK_minmaxSource_WeightedMAE_FinetuneAugsEnabled",
    "logs/T1W_FINETUNE_L1_HFWavelet_GIN_MASK_minmaxSource_WeightedMAE_NoFinetuneAugs"
]

ID = "Pre_train_pre_aug3_2"
os.makedirs(f"metrics/{ID}/", exist_ok=True)

per_experiment = {}
for ele in paths:
    exp_name = ele.split("/")[-1]
    numeric_df, metric_cols = merge_testing_metrics(
        f"{ele}/metrics", f"metrics/{ID}/{exp_name}.csv"
    )
    per_experiment[exp_name] = (numeric_df, metric_cols)

if per_experiment:
    build_combined_summary(per_experiment, f"metrics/{ID}/combined_summary.csv")