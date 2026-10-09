#!/usr/bin/env python3
import os
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm

# -----------------------
# 🔧 1. User configuration
# -----------------------
REF_DIR = "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/post_processing/SEG_MASKS/7T"
MODEL_DIRS = {
    "VNETL1": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/post_processing/SEG_MASKS/VNET_L1",
    "VNETGAN": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/post_processing/SEG_MASKS/VNET_L1_GAN",
    "VENTPercept": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/post_processing/SEG_MASKS/VNET_L1_Percept",
    "UNET_Conv": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/post_processing/SEG_MASKS/251112_UNET_3D_L1_32_conv",
    "UNET_Nearest" : "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/post_processing/SEG_MASKS/251112_UNET_3D_L1_32_nearest",
    "UNET_trilinear": "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/post_processing/SEG_MASKS/251112_UNET_3D_L1_32_trilinear"
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
    
}

OUT_CSV = "ISBI_Final_dice_summary.csv"

# --- Tissue group definitions (SynthSeg / FreeSurfer label IDs) ---
# Adjust as needed — these come from FreeSurferColorLUT conventions.
LABELS_GM = {
    # Cortical + subcortical gray
    3, 42, 8, 47, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54, 26, 58, 27, 59,
    28, 60, 10, 49, 16  # Thalamus, Pallidum, Hippocampus, Amygdala, etc.
}
LABELS_WM = {2, 41, 7, 46, 250, 251, 252, 253, 254, 255}  # Cerebral + cerebellar WM
LABELS_CSF = {4, 43, 14, 15, 24, 31, 63}  # Ventricles, CSF
LABELS_BRAINSTEM = {16, 60} | set(range(173, 179))  # Brainstem, cerebellar nuclei (approx)

# -----------------------
# 🧠 2. Dice helpers
# -----------------------
EPS = 1e-8

def dice_binary(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    denom = mask1.sum() + mask2.sum()
    return 2.0 * inter / (denom + EPS)

def dice_per_label(seg1, seg2, labels):
    return np.mean([
        dice_binary(seg1 == l, seg2 == l)
        for l in labels if (seg1 == l).any() or (seg2 == l).any()
    ]) if labels else np.nan

def dice_macro(seg1, seg2):
    labels = np.unique(np.concatenate([np.unique(seg1), np.unique(seg2)]))
    labels = labels[labels != 0]
    return np.mean([dice_binary(seg1 == l, seg2 == l) for l in labels]) if labels.size else np.nan

def load_seg(path):
    return nib.load(path).get_fdata().astype(np.int32)

# -----------------------
# 🧮 3. Main computation
# -----------------------
def compute_all():
    ref_files = sorted(f for f in os.listdir(REF_DIR) if f.endswith((".nii.gz", ".mgz", ".nii")))
    # input(ref_files)
    if not ref_files:
        raise RuntimeError("No reference files found.")

    results = []
    for model_name, model_dir in MODEL_DIRS.items():
        print(f"\n🧩 Evaluating model: {model_name}")
        for fname in tqdm(ref_files, desc=model_name, ncols=80):
            
            ref_path = os.path.join(REF_DIR, fname)
            all_files_model_dir = os.listdir(model_dir)
            for ele in all_files_model_dir:
                # print(ele)
                if fname[:10] in ele:
                    fname_model = ele
                    # print("mogged", fname_model, fname)
            # print("AAAA",all_files_model_dir)
            # input(f"BBBBB{fname}, {fname} ")
            # continue
            cmp_path = os.path.join(model_dir, fname_model)
            if not os.path.exists(cmp_path):
                continue
            print("Compared ", ref_path, cmp_path)        
            seg_ref = load_seg(ref_path)
            seg_cmp = load_seg(cmp_path)

            dice_gm = dice_per_label(seg_ref, seg_cmp, LABELS_GM)
            dice_wm = dice_per_label(seg_ref, seg_cmp, LABELS_WM)
            dice_csf = dice_per_label(seg_ref, seg_cmp, LABELS_CSF)
            dice_bs  = dice_per_label(seg_ref, seg_cmp, LABELS_BRAINSTEM)
            dice_avg = dice_macro(seg_ref, seg_cmp)

            results.append({
                "model": model_name,
                "scan": fname,
                "dice_GM": dice_gm,
                "dice_WM": dice_wm,
                "dice_CSF": dice_csf,
                "dice_Brainstem_CerebellarNuclei": dice_bs,
                "dice_AvgAll": dice_avg,
            })

    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print((results))
    print(f"\n✅ Saved results to {OUT_CSV}")
    print(df.groupby("model")[["dice_GM","dice_WM","dice_CSF","dice_Brainstem_CerebellarNuclei","dice_AvgAll"]].mean())
    return df

# -----------------------
# 🚀 4. Run
# -----------------------
if __name__ == "__main__":
    compute_all()
