import os
import re
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchmetrics.functional import mean_absolute_error, mean_squared_error

def compute_psnr(x, y, max_val=1.0):
    mse = F.mse_loss(x, y)
    psnr = 20 * torch.log10(max_val) - 10 * torch.log10(mse + 1e-8)
    return psnr

def compute_ssim(x, y, window_size=11, C1=0.01**2, C2=0.03**2):
    """3D SSIM (simplified version, patch-based using 3D convs)"""
    if x.ndim == 1:
        x = x.unsqueeze(0).unsqueeze(0)
        y = y.unsqueeze(0).unsqueeze(0)
    mu_x = F.avg_pool3d(x, window_size, 1, window_size//2)
    mu_y = F.avg_pool3d(y, window_size, 1, window_size//2)
    sigma_x = F.avg_pool3d(x * x, window_size, 1, window_size//2) - mu_x ** 2
    sigma_y = F.avg_pool3d(y * y, window_size, 1, window_size//2) - mu_y ** 2
    sigma_xy = F.avg_pool3d(x * y, window_size, 1, window_size//2) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return ssim_map.mean()

def compute_csim(x, y):
    """Contrast similarity index (CSIM)"""
    C = 1e-8
    sigma_x = torch.std(x)
    sigma_y = torch.std(y)
    return (2 * sigma_x * sigma_y + C) / (sigma_x ** 2 + sigma_y ** 2 + C)

def masked_metrics(orig, recon, mask):
    mask = mask > 0
    x = torch.tensor(orig[mask], dtype=torch.float32)
    y = torch.tensor(recon[mask], dtype=torch.float32)

    # normalize
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)

    mse = mean_squared_error(x, y)
    mae = mean_absolute_error(x, y)
    psnr = compute_psnr(x, y)
    csim = compute_csim(x, y)

    # reshape for SSIM
    x_3d = x.view(1, 1, int(len(x) ** (1/3)), -1, -1) if len(x) > 1000 else x.view(1, 1, 1, -1, -1)
    y_3d = y.view_as(x_3d)
    try:
        ssim = compute_ssim(x_3d, y_3d)
    except Exception:
        ssim = torch.tensor(float('nan'))

    return float(psnr), float(ssim), float(csim), float(mae), float(mse)

def evaluate_directories(dir_list, mask_dir, output_csv="metrics.csv"):
    rows = []
    for d in dir_list:
        for f in os.listdir(d):
            if "epoch=300" in f and f.endswith(".nii.gz"):
                sub_match = re.search(r"sub_\d+", f)
                if not sub_match:
                    continue
                sub_id = sub_match.group(0)

                orig_file = next((os.path.join(d, x) for x in os.listdir(d) if "orig" in x and sub_id in x), None)
                if not orig_file:
                    continue

                epoch300_file = os.path.join(d, f)
                mask_file = next((os.path.join(mask_dir, x) for x in os.listdir(mask_dir) if sub_id in x), None)
                if not mask_file:
                    continue

                orig = nib.load(orig_file).get_fdata()
                recon = nib.load(epoch300_file).get_fdata()
                mask = nib.load(mask_file).get_fdata()

                psnr, ssim, csim, mae, mse = masked_metrics(orig, recon, mask)
                rows.append({
                    "path_dir": d,
                    "subject_id": sub_id,
                    "PSNR": psnr,
                    "SSIM": ssim,
                    "CSIM": csim,
                    "MAE": mae,
                    "MSE": mse
                })
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved metrics to {output_csv}")
    return df


dir_list = [
    "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/30_ESAU_3D_L1_32_conv/generations",
    "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/30_ESAU_3D_L1_32_nearest/generations",
    "//storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/30_ESAU_3D_L1_32_trilinear/generations",
    "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/251112_UNET_3D_L1_32_conv/generations",
    "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/251112_UNET_3D_L1_32_nearest/generations",
    "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/251112_UNET_3D_L1_32_trilinear/generations",
    "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/VNET_L1/generations",
    "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/VNET_L1_GAN/generations"
   " /storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/VNET_L1_Percept/generations"
]
mask_dir = "/storage/an_inam/MR2MR/Data/10_Pat_t1/MASK7T_MNIReg"

df = evaluate_directories(dir_list, mask_dir, output_csv="comparison_metrics.csv")
