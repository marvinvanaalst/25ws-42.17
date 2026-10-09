import pandas as pd
from pathlib import Path

import pandas as pd
from pathlib import Path

def merge_testing_metrics(experiments_dir, out_csv):
    """
    Merge all *_testing metric files into a single CSV,
    reorder metric columns, and add average/std rows.
    """
    experiments_dir = Path(experiments_dir)
    all_files = sorted(experiments_dir.glob("*_testing_.csv"))

    if not all_files:
        raise FileNotFoundError(f"No files ending with '_testing' found in {experiments_dir}")

    dfs = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
        except Exception:
            df = pd.read_csv(f, sep="\t")
        dfs.append(df)

    big_df = pd.concat(dfs, ignore_index=True)

    # Identify id vs metric columns
    id_cols = ["sample_index", "weight_prefix"]
    metric_cols = [c for c in big_df.columns if c not in id_cols]

    # Convert to numeric
    for c in metric_cols:
        big_df[c] = pd.to_numeric(big_df[c], errors="coerce")

    # ---- Compute stats ----
    mean_vals = big_df[metric_cols].mean().tolist()
    std_vals  = big_df[metric_cols].std().tolist()

    mean_row = ["AVERAGE", ""] + [f"{v:0.6f}" for v in mean_vals]
    std_row  = ["STD", ""] + [f"{v:0.6f}" for v in std_vals]

    # Format per-scan rows
    for c in metric_cols:
        big_df[c] = big_df[c].map(lambda v: f"{v:0.6f}" if pd.notna(v) else "")

    # Append summary rows
    summary_df = pd.DataFrame([mean_row, std_row], columns=big_df.columns)
    final_df = pd.concat([big_df, summary_df], ignore_index=True)

    # ---- Reorder columns ----
    order_priority = ["mse", "mae", "psnr", "ssim_3d", "ssim_2d"]
    ordered_metrics = []
    for key in order_priority:
        ordered_metrics.extend([c for c in metric_cols if key.lower() in c.lower()])

    # Add any remaining metrics at the end
    other_metrics = [c for c in metric_cols if c not in ordered_metrics]

    final_cols = id_cols + ordered_metrics + other_metrics
    final_df = final_df[final_cols]

    # Save
    final_df.to_csv(out_csv, index=False)
    print(f"✅ Merged {len(all_files)} files, reordered metrics, added averages & std → {out_csv}")

paths = [
    
    # "logs/VNET_L1",
    # "logs/VNET_L1_GAN",
    # "logs/VNET_L1_Percept",
    
    # "logs/251112_UNET_3D_L1_32_nearest_150",
    # "logs/251112_UNET_3D_L1_32_trilinear_150",
    # "logs/251112_UNET_3D_L1_32_conv_150",
    
    # "logs/251112_UNET_3D_L1_32_nearest",
    # "logs/251112_UNET_3D_L1_32_trilinear",
    # "logs/251112_UNET_3D_L1_32_conv",
    

    
    # "logs/151112_SEResidual_3D_L1_32_conv",
    
    

    # "logs/30_ESAU_3D_L1_32_nearest_150",
    # "logs/30_ESAU_3D_L1_32_nearest",
    
    # "logs/30_ESAU_3D_L1_32_conv_150",
    # "logs/30_ESAU_3D_L1_32_conv",
    
    # "logs/30_ESAU_3D_L1_32_trilinear_150",
    # "logs/30_ESAU_3D_L1_32_trilinear",
    # "logs/251201_ESAU_3D_L1_48_trilinear",
    # "logs/251201_ESAU_3D_L1_64_trilinear",

    
    # "logs/30_ESAU_3D_L1_32_trilinear",
    # "logs/30_ESAU_3D_L1_32_trilinear_150",
    # "logs/251205_ESAU_TL_3D_L1_64_trilinear",
    # "logs/251205_ESAU_TL_3D_L1_64_trilinear_TOP1",
    # "logs/251205_ESAU_TL_3D_L1_96_trilinear",
    # "logs/251205_ESAU_TL_3D_L1_128_trilinear",


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
    "T1W_FINETUNE_L1_HFWavelet_GIN_MASK_minmaxSource_MAE_style_FinetuneAugsEnabled",
    "T1W_FINETUNE_L1_HFWavelet_GIN_MASK_minmaxSource_MAE_style_NoFinetuneAugs",
    "T1W_FINETUNE_L1_HFWavelet_GIN_MASK_minmaxSource_WeightedMAE_FinetuneAugsEnabled",
    "T1W_FINETUNE_L1_HFWavelet_GIN_MASK_minmaxSource_WeightedMAE_NoFinetuneAugs"
]

# paths = [
#     # "logs/t2_20251121_ESAU_3D_L1_32_conv",
#     # "logs/t2_20251121_SEResidual_3D_L1_32_conv_gcr",
#     # "logs/t2_20251121_SEResidual_3D_L1_32_conv_gcr",
#     # "logs/t2_20251121_SEResidual_3D_L1_32_conv_gcr_0.5",
#     # "logs/t2_20251121_SEResidual_3D_L1_32_conv_gcr_0.598_percentile",
#     # "logs/t2_20251121_UNET_3D_L1_32_conv_bcr",
#     # "logs/t2_20251121_UNET_3D_L1_32_conv_gcr",
#     # "logs/t2_20251121_UNET_3D_L1_32_conv_gcr_0.5Overlap",
#     # "logs/t2_20251121_UNET_3D_L1_32_conv_gcr_0.5Overlap98_percentile"
# ]

ID = "Pre_train_pre_aug3"
import os
os.makedirs(f"metrics/{ID}/",exist_ok=True)
for ele in paths:
    merge_testing_metrics(f"{ele}/metrics", f"metrics/{ID}/{ele.split('/')[-1]}.csv")