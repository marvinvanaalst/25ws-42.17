#!/usr/bin/env python3
import os
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm

# ============================================================
# 🔧 1. User configuration
# ============================================================
REF_DIR = "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/post_processing/SEG_MASKS/7T"
MODEL_DIRS = {
    "VNETL1": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/post_processing/SEG_MASKS/VNET_L1",
    "VNETGAN": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/post_processing/SEG_MASKS/VNET_L1_GAN",
    "VENTPercept": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/post_processing/SEG_MASKS/VNET_L1_Percept",
    "UNET_Conv": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/post_processing/SEG_MASKS/251112_UNET_3D_L1_32_conv",
    "UNET_Nearest" : "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/post_processing/SEG_MASKS/251112_UNET_3D_L1_32_nearest",
    "UNET_trilinear": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/post_processing/SEG_MASKS/251112_UNET_3D_L1_32_trilinear",
    "7T_Self": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/post_processing/SEG_MASKS/7T",
    "3T": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/post_processing/SEG_MASKS/10_Pat_t1",
    "3DESAU_conv" : "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/post_processing/SEG_MASKS/30_ESAU_3D_L1_32_conv",
    "3DESAU_nearest": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/post_processing/SEG_MASKS/30_ESAU_3D_L1_32_nearest",
    "3DESAU_trilinear" : "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/post_processing/SEG_MASKS/30_ESAU_3D_L1_32_trilinear",
}

MODEL_DIRS = {
    "VNETL1": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/FULL_LOG_SYNTH_ISBI/VNET_L1",
    "VNETGAN": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/FULL_LOG_SYNTH_ISBI/VNET_L1_GAN",
    "VENTPercept": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/FULL_LOG_SYNTH_ISBI/VNET_L1_Percept",
    
    "3DEsaUNET(Conv)": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/FULL_LOG_SYNTH_ISBI/ISBI_SUB_30_ESAU_3D_L1_32_conv",
    "3DEsaUNET(Near)": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/FULL_LOG_SYNTH_ISBI/ISBI_SUB_30_ESAU_3D_L1_32_nearest",
    "3DEsaUNET(Tri)": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/FULL_LOG_SYNTH_ISBI/30_ESAU_3D_L1_32_trilinear",
    
    "SwinUNETR(Conv)": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/FULL_LOG_SYNTH_ISBI/2026_01_18_SwinUNETR_Max_Conv",
    "SwinUNETR(Near)": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/FULL_LOG_SYNTH_ISBI/2026_01_18_SwinUNETR_Max_Nearest",
    "SwinUNETR(Tri)": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/FULL_LOG_SYNTH_ISBI/2026_01_18_SwinUNETR_Max_trilinear",
    
    "UNETR(Conv)": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/FULL_LOG_SYNTH_ISBI/2026_01_20_UNETR_32_Conv",
    "UNETR(Near)": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/FULL_LOG_SYNTH_ISBI/2026_01_20_UNETR_32_nearest",
    "UNETR(Tri)": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/FULL_LOG_SYNTH_ISBI/2026_01_20_UNETR_32_trilinear",
    
    
    "SE-ResidualUNET(Tri)": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/FULL_LOG_SYNTH_ISBI/ISBI_fin_SE-Residual_3D_trilinear",
    "SE-ResidualUNET(Near)": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/FULL_LOG_SYNTH_ISBI/ISBI_fin_SE-Residual_3D_nearest",
    "SE-ResidualUNET(Conv)": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/FULL_LOG_SYNTH_ISBI/151112_SEResidual_3D_L1_32_conv",
    #pending
    "3D_UNET(Tri)": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/FULL_LOG_SYNTH_ISBI/UNET_3D_L1_32_trilinear",
    "3D_UNET(Near)": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/FULL_LOG_SYNTH_ISBI/UNET_3D_L1_32_nearest",
    "3D_UNET(Conv)": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/FULL_LOG_SYNTH_ISBI/UNET_3D_L1_32_conv",
    
    "BBDM": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/post_processing/SEG_MASKS/BBDM"
    
}
OUT_CSV = "ISBI_Fin_DICEAGGREGATE.csv"
OUT_CSV_SCANWISE = "ISBI_OUT_CSV_SCANWISE.csv"
OUT_CSV_SUMMARY  = "ISBI_OUT_CSV_SUMMARY.csv"

# --- Tissue group definitions (FreeSurfer/SynthSeg label IDs) ---
#!/usr/bin/env python3

# --- Tissue group definitions (FreeSurfer/SynthSeg label IDs) ---
# LABELS_GM = {
#     3, 42, 8, 47, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54,
#     26, 58, 27, 59, 28, 60, 10, 49, 16
# }
# LABELS_WM = {2, 41, 7, 46, 250, 251, 252, 253, 254, 255}
# LABELS_CSF = {4, 43, 14, 15, 24, 31, 63}
# LABELS_BRAINSTEM = {16, 60} | set(range(173, 179))  # brainstem + cerebellar nuclei

# SynthSeg v2.0 label definitions
LABELS_WM  = {2, 7, 41, 46}  # White matter
LABELS_GM  = {
    3, 8, 10, 11, 12, 13, 17, 18,
    26, 28, 42, 47, 49, 50, 51, 52,
    53, 54, 58, 60
}  # Gray matter
LABELS_CSF = {24}             # Subarachnoid CSF (SynthSeg 2.0)
LABELS_BRAINSTEM = {16}       # Brainstem

EPS = 1e-8

# ============================================================
# 🧠 2. Dice helper functions
# ============================================================
def dice_binary(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    denom = mask1.sum() + mask2.sum()
    return 2.0 * inter / (denom + EPS)

def dice_per_label(seg1, seg2, labels):
    vals = [
        dice_binary(seg1 == l, seg2 == l)
        for l in labels if (seg1 == l).any() or (seg2 == l).any()
    ]
    return np.mean(vals) if vals else np.nan

def dice_macro(seg1, seg2):
    labels = np.unique(np.concatenate([np.unique(seg1), np.unique(seg2)]))
    labels = labels[labels != 0]
    vals = [dice_binary(seg1 == l, seg2 == l) for l in labels]
    return np.mean(vals) if len(vals) else np.nan

def load_seg(path):
    return nib.load(path).get_fdata().astype(np.int32)

# ============================================================
# 🧮 3. Core computation
# ============================================================
def compute_all():
    ref_files = sorted(f for f in os.listdir(REF_DIR) if f.endswith((".nii.gz", ".mgz", ".nii")))
    if not ref_files:
        raise RuntimeError("No reference files found in REF_DIR")

    all_results = []
    for model_name, model_dir in MODEL_DIRS.items():
        print(f"\n🧩 Evaluating model: {model_name}")
        for fname in tqdm(ref_files, desc=model_name, ncols=80):
            ref_path = os.path.join(REF_DIR, fname)
            all_files_model_dir = os.listdir(model_dir)
            for ele in all_files_model_dir:
                # print(ele)
                if (fname[:10] and "300") in ele:
                    fname_model = ele
                    print("mogged", fname_model, fname)
                    input
            # print("AAAA",all_files_model_dir)
            # input(f"BBBBB{fname}, {fname} ")
            # continue
            cmp_path = os.path.join(model_dir, fname_model)
            if not os.path.exists(cmp_path):
                continue
            print("\nCompared ", ref_path, cmp_path)        
            seg_ref = load_seg(ref_path)
            seg_cmp = load_seg(cmp_path)

            dice_gm  = dice_per_label(seg_ref, seg_cmp, LABELS_GM)
            dice_wm  = dice_per_label(seg_ref, seg_cmp, LABELS_WM)
            dice_csf = dice_per_label(seg_ref, seg_cmp, LABELS_CSF)
            dice_bs  = dice_per_label(seg_ref, seg_cmp, LABELS_BRAINSTEM)
            dice_avg = dice_macro(seg_ref, seg_cmp)

            all_results.append({
                "model": model_name,
                "scan": fname,
                "dice_GM": float(dice_gm) if not np.isnan(dice_gm) else np.nan,
                "dice_WM": float(dice_wm) if not np.isnan(dice_wm) else np.nan,
                "dice_CSF": float(dice_csf) if not np.isnan(dice_csf) else np.nan,
                "dice_Brainstem_CerebellarNuclei": float(dice_bs) if not np.isnan(dice_bs) else np.nan,
                "dice_AvgAll": float(dice_avg) if not np.isnan(dice_avg) else np.nan,
            })

    df = pd.DataFrame(all_results)

    # 🧩 Force numeric conversion before aggregation
    numeric_cols = [c for c in df.columns if c.startswith("dice_")]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    df.to_csv(OUT_CSV_SCANWISE, index=False)
    print(f"\n✅ Saved scan-wise Dice results to: {OUT_CSV_SCANWISE}")

    # ============================================================
    # 📊 4. Summary per model (mean ± std)
    # ============================================================
    agg = df.groupby("model")[numeric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = [f"{col}_{stat}" if stat else col for col, stat in agg.columns]
    agg.to_csv(OUT_CSV_SUMMARY, index=False)

    print(f"✅ Saved per-model summary to: {OUT_CSV_SUMMARY}\n")

    # Clean terminal summary print
    print("📈 Mean ± SD per model:\n")
    for _, row in agg.iterrows():
        print(f"Model: {row['model']}")
        for c in numeric_cols:
            m, s = row[f"{c}_mean"], row[f"{c}_std"]
            print(f"  {c:35s}: {m:.4f} ± {s:.4f}")
        print("-" * 50)

    return df, agg

# ============================================================
# 🚀 5. Run
# ============================================================
if __name__ == "__main__":
    compute_all()
