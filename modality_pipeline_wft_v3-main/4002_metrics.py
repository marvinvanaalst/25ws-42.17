import pandas as pd
import numpy as np
from pathlib import Path
import os
import re
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.interpolate import make_interp_spline
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


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


def _safe_get(d, *keys, default="missing"):
    """Walk nested dict/list keys; return `default` if anything is missing."""
    cur = d
    for k in keys:
        try:
            cur = cur[k]
        except (KeyError, IndexError, TypeError):
            return default
    if cur is None:
        return default
    return cur


def _trend_interpolation(x, y):
    """Return (xs, ys) for a smooth average-trend line over unique-x means."""
    order = np.argsort(x)
    xu, yu = x[order], y[order]
    ux = np.unique(xu)
    uy = np.array([yu[xu == v].mean() for v in ux])
    if len(ux) < 2:
        return ux, uy
    xs = np.linspace(ux.min(), ux.max(), 200)
    if _HAS_SCIPY and len(ux) >= 4:
        k = min(3, len(ux) - 1)
        try:
            spline = make_interp_spline(ux, uy, k=k)
            return xs, spline(xs)
        except Exception:
            pass
    deg = min(3, len(ux) - 1)
    coeffs = np.polyfit(ux, uy, deg)
    return xs, np.polyval(coeffs, xs)


def plot_epoch_graphs(numeric_df, metric_cols, out_dir, exp_name):
    """Individual per-metric plots + one combined plot, with avg trend lines."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = numeric_df.copy()
    df["__epoch__"] = df["weight_prefix"].map(_epoch_from_prefix)
    df = df[df["__epoch__"] >= 0]
    if df.empty:
        print(f"⚠️  No epoch info parsed for {exp_name}; skipping graphs")
        return

    ordered = _order_metrics(metric_cols)

    # ---- Individual plots ----
    for c in ordered:
        sub = df[["__epoch__", c]].dropna()
        if sub.empty:
            continue
        x = sub["__epoch__"].to_numpy(dtype=float)
        y = sub[c].to_numpy(dtype=float)

        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, s=18, alpha=0.4, label="raw points")

        ux = np.unique(x)
        uy = np.array([y[x == v].mean() for v in ux])
        plt.plot(ux, uy, marker="o", linewidth=1.2, label="mean per epoch")

        xs, ys = _trend_interpolation(x, y)
        if len(xs) >= 2:
            plt.plot(xs, ys, linewidth=2.2, linestyle="--", label="avg trend")

        plt.title(f"{exp_name} — {c} vs epoch")
        plt.xlabel("epoch")
        plt.ylabel(c)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{c}.png", dpi=150)
        plt.close()

    # ---- Combined plot (mean-per-epoch trend for every metric) ----
    plt.figure(figsize=(10, 6))
    plotted = False
    for c in ordered:
        sub = df[["__epoch__", c]].dropna()
        if sub.empty:
            continue
        x = sub["__epoch__"].to_numpy(dtype=float)
        y = sub[c].to_numpy(dtype=float)
        xs, ys = _trend_interpolation(x, y)
        if len(xs) >= 2:
            plt.plot(xs, ys, linewidth=2.0, label=c)
            plotted = True
    if plotted:
        plt.title(f"{exp_name} — combined metric trends vs epoch")
        plt.xlabel("epoch")
        plt.ylabel("metric value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "combined.png", dpi=150)
    plt.close()

    print(f"📈 Saved epoch graphs → {out_dir}")


def load_experiment_json(experiment_path):
    """Load the first *.json in the experiment dir and extract requested fields."""
    experiment_path = Path(experiment_path)
    json_files = sorted(experiment_path.glob("*.json"))
    fields = {
        "model_name": "missing",
        "block_augments_apply": "missing",
        "block_augments_downsample_ratio": "missing",
        "block_augments_mask_ratio": "missing",
        "loss_types": "missing",
        "finetunning_do": "missing",
        "finetunning_weight_path": "missing",
        "training_epochs": "missing",
    }
    if not json_files:
        print(f"⚠️  No JSON found in {experiment_path}")
        return fields

    with open(json_files[0], "r") as fh:
        JV = json.load(fh)

    fields["model_name"] = _safe_get(JV, "model_info", "name")
    fields["block_augments_apply"] = _safe_get(JV, "pretrain_augments", "block_augments", "apply")
    fields["block_augments_downsample_ratio"] = _safe_get(
        JV, "pretrain_augments", "block_augments", "config", "downsample_ratio")
    fields["block_augments_mask_ratio"] = _safe_get(
        JV, "pretrain_augments", "block_augments", "config", "mask_ratio")
    fields["loss_types"] = _safe_get(JV, "loss", "types")
    fields["finetunning_do"] = _safe_get(JV, "finetunning", "do")
    fields["finetunning_weight_path"] = _safe_get(JV, "finetunning", "weight_path")
    fields["training_epochs"] = _safe_get(JV, "training", "epochs")
    return fields


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

    metric_cols = [c for c in big_df.columns if c not in ID_COLS + ["__split__"]]

    for c in metric_cols:
        big_df[c] = pd.to_numeric(big_df[c], errors="coerce")

    numeric_df = big_df.copy()

    mean_vals = big_df[metric_cols].mean().tolist()
    std_vals = big_df[metric_cols].std().tolist()
    mean_row = ["AVERAGE", ""] + [f"{v:0.6f}" for v in mean_vals]
    std_row = ["STD", ""] + [f"{v:0.6f}" for v in std_vals]

    out_df = big_df.drop(columns="__split__").copy()
    for c in metric_cols:
        out_df[c] = out_df[c].map(lambda v: f"{v:0.6f}" if pd.notna(v) else "")

    summary_df = pd.DataFrame([mean_row, std_row], columns=out_df.columns)
    final_df = pd.concat([out_df, summary_df], ignore_index=True)

    ordered_metrics = _order_metrics(metric_cols)
    final_cols = ID_COLS + ordered_metrics
    final_df = final_df[final_cols]

    final_df.to_csv(out_csv, index=False)
    print(f"✅ Merged {len(all_files)} files, reordered metrics, added averages & std → {out_csv}")

    return numeric_df, metric_cols


def build_combined_summary(per_experiment, out_csv, verbose_csv=None):
    """
    per_experiment: dict of experiment_name -> (numeric_df, metric_cols, json_fields)
    For each experiment, each split is evaluated on multiple weights (epochs).
    Take the highest-epoch weight per split as that split's representative,
    average its samples, then average across splits to get one row per
    experiment. Also record the epoch used for each split. Optionally writes a
    second 'verbose' CSV with the extracted JSON metadata merged in.
    """
    rows = []
    verbose_rows = []
    all_metric_cols = []

    for exp_name, (df, metric_cols, json_fields) in per_experiment.items():
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

        verbose_rows.append({**row, **json_fields})

        for c in metric_cols:
            if c not in all_metric_cols:
                all_metric_cols.append(c)

    metric_avg_cols = [f"Average_{c}" for c in _order_metrics(all_metric_cols)]

    combined = pd.DataFrame(rows)
    ordered_cols = ["Experiment Name", "Number_of_Splits_used", "Epochs_used_per_split"] + metric_avg_cols
    combined = combined[[c for c in ordered_cols if c in combined.columns]]
    combined.to_csv(out_csv, index=False)
    print(f"✅ Combined summary across {len(rows)} experiments → {out_csv}")

    if verbose_csv is not None:
        json_cols = [
            "model_name", "block_augments_apply", "block_augments_downsample_ratio",
            "block_augments_mask_ratio", "loss_types", "finetunning_do",
            "finetunning_weight_path", "training_epochs",
        ]
        verbose = pd.DataFrame(verbose_rows)
        verbose_ordered = (
            ["Experiment Name", "Number_of_Splits_used", "Epochs_used_per_split"]
            + json_cols + metric_avg_cols
        )
        verbose = verbose[[c for c in verbose_ordered if c in verbose.columns]]
        verbose.to_csv(verbose_csv, index=False)
        print(f"✅ Verbose summary (with JSON metadata) → {verbose_csv}")

    return combined


paths = [
    "logs/0_1Tto7TNoPretraining_SynthSegl15LevelWavelet",
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
    "logs/T1W_FINETUNE_L1_HFWavelet_GIN_MASK_minmaxSource_WeightedMAE_NoFinetuneAugs",
    "logs/T1W_FINETUNE_L1_HFWavelet_GIN_MASK_minmaxSource_MAE_style_3Levels",
    "logs/T1W_FINETUNE_L1_HFWavelet_GIN_MASK_minmaxSource_MAE_style_5Levels",
    "logs/0_1Tto7TNoPretraining_l12LevelWavelet",
    # "logs/0_1Tto7TNoPretraining_l15LevelWavelet",
    "logs/0_1Tto7TNoPretraining_SynthSegl12LevelWavelet",
]

ID = "13aug26_meeting"
os.makedirs(f"metrics/{ID}/", exist_ok=True)
graphs_root = f"metrics/{ID}/graphs"
os.makedirs(graphs_root, exist_ok=True)

per_experiment = {}
for ele in paths:
    exp_name = ele.split("/")[-1]
    numeric_df, metric_cols = merge_testing_metrics(
        f"{ele}/metrics", f"metrics/{ID}/{exp_name}.csv"
    )
    plot_epoch_graphs(numeric_df, metric_cols, f"{graphs_root}/{exp_name}", exp_name)
    json_fields = load_experiment_json(ele)
    per_experiment[exp_name] = (numeric_df, metric_cols, json_fields)

if per_experiment:
    build_combined_summary(
        per_experiment,
        f"metrics/{ID}/combined_summary.csv",
        verbose_csv=f"metrics/{ID}/combined_summary_verbose.csv",
    )