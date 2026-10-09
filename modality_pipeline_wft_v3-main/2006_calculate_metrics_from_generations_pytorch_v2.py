import os
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from colorama import Fore, Style, init

# Torch & MONAI metrics
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from monai.metrics import SSIMMetric

init(autoreset=True)

# --- HEADERS CONFIG ---
# These are the explicit headers for the individual subject-level CSVs
METRIC_COLUMNS = [
    'subject', 'fold', 'weight_prefix', 
    'test_psnr', 'test_ssim_3D', 'test_ssim_2D', 
    'test_mae', 'test_mse'
]

EXPECTED_FOLDS = 5 
# ----------------------

def calculate_2d_ssim_torch(img_t, ref_t, ssim_2d_m):
    """Calculates average SSIM across Axial, Coronal, and Sagittal planes."""
    try:
        # Axial (D)
        axial = torch.mean(torch.stack([ssim_2d_m(img_t[:,:,i,:,:], ref_t[:,:,i,:,:]) for i in range(img_t.shape[2])]))
        # Coronal (H)
        coronal = torch.mean(torch.stack([ssim_2d_m(img_t[:,:,:,i,:], ref_t[:,:,:,i,:]) for i in range(img_t.shape[3])]))
        # Sagittal (W)
        sagittal = torch.mean(torch.stack([ssim_2d_m(img_t[:,:,:,:,i], ref_t[:,:,:,:,i]) for i in range(img_t.shape[4])]))
        return ((axial + coronal + sagittal) / 3).item()
    except Exception:
        return np.nan

@torch.no_grad()
def eval_single_pair(file_info):
    """Worker function: Nibabel I/O + TorchMetrics computation."""
    try:
        fake_path, orig_path, subject_id, weight_id, fold = file_info
        
        # 1. Load volumes
        fake_np = nib.load(fake_path).get_fdata().astype(np.float32)
        orig_np = nib.load(orig_path).get_fdata().astype(np.float32)

        # 2. Convert to Tensors [B, C, D, H, W]
        fake_t = torch.from_numpy(fake_np).unsqueeze(0).unsqueeze(0)
        orig_t = torch.from_numpy(orig_np).unsqueeze(0).unsqueeze(0)

        # 3. Initialize local metrics (Required for multiprocessing safety)
        psnr_m = PeakSignalNoiseRatio(data_range=1.0)
        mae_m = MeanAbsoluteError()
        mse_m = MeanSquaredError()
        ssim_3d_m = SSIMMetric(spatial_dims=3, data_range=1.0, kernel_type='gaussian', win_size=11)
        ssim_2d_m = SSIMMetric(spatial_dims=2, data_range=1.0, kernel_type='gaussian', win_size=11)

        return {
            'subject': subject_id,
            'fold': fold,
            'weight_prefix': weight_id,
            'test_psnr': psnr_m(fake_t, orig_t).item(),
            'test_ssim_3D': ssim_3d_m(fake_t, orig_t).item(),
            'test_ssim_2D': calculate_2d_ssim_torch(fake_t, orig_t, ssim_2d_m),
            'test_mae': mae_m(fake_t, orig_t).item(),
            'test_mse': mse_m(fake_t, orig_t).item()
        }
    except Exception as e:
        return {"error": f"Error in {subject_id}: {str(e)}"}

def run_evaluation(target_dir, dest_dir, id_token):
    target_path = Path(target_dir)
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    all_summary_data = []
    error_list = []
    fold_warnings = []

    experiments = [d for d in target_path.iterdir() if d.is_dir()]
    
    # --- LEVEL 1: EXPERIMENTS ---
    for exp_folder in tqdm(experiments, desc="Overall Experiments", unit="exp"):
        gen_dir = exp_folder / "generations"
        if not gen_dir.exists():
            error_list.append(f"Skipping {exp_folder.name}: No 'generations' folder.")
            continue

        fake_files = list(gen_dir.glob("*_fake.nii.gz"))
        weight_groups = {}

        for f_path in fake_files:
            try:
                parts = f_path.name.split('_')
                fold = parts[0]
                subject_id = f"{parts[1]}_{parts[2]}" 
                weight_id = f_path.name.replace(f"{fold}_{subject_id}_", "").replace("_fake.nii.gz", "")
                orig_path = gen_dir / f"{fold}_{subject_id}_orig.nii.gz"
                
                if not orig_path.exists():
                    error_list.append(f"Missing GT: {orig_path.name} in {exp_folder.name}")
                    continue

                if weight_id not in weight_groups: weight_groups[weight_id] = []
                weight_groups[weight_id].append((str(f_path), str(orig_path), subject_id, weight_id, fold))
            except Exception as e:
                error_list.append(f"Filename parse error in {f_path.name}: {e}")

        # --- LEVEL 2: WEIGHTS ---
        for weight_name, tasks in tqdm(weight_groups.items(), desc=f"  Weights in {exp_folder.name}", leave=False):
            valid_results = []
            
            # --- LEVEL 3: VOLUMES (Parallel) ---
            with Pool(processes=max(1, cpu_count() - 1)) as pool:
                pbar = tqdm(total=len(tasks), desc="    Volumes", leave=False, unit="vol")
                for result in pool.imap_unordered(eval_single_pair, tasks):
                    if "error" in result:
                        error_list.append(result["error"])
                    else:
                        valid_results.append(result)
                    pbar.update(1)
                pbar.close()

            if not valid_results: continue

            # Individual Weight CSV (Subject-level)
            df_weight = pd.DataFrame(valid_results, columns=METRIC_COLUMNS)
            u_folds = df_weight['fold'].nunique()
            
            if u_folds < EXPECTED_FOLDS:
                fold_warnings.append(f"{exp_folder.name} ({weight_name}): {u_folds}/{EXPECTED_FOLDS} folds")

            # Save file immediately
            weight_csv_name = f"{exp_folder.name}_{weight_name}.csv"
            df_weight.to_csv(dest_path / weight_csv_name, index=False)

            # --- PREPARE AGGREGATE STATS ---
            stats_row = {
                'experiment': exp_folder.name,
                'weight_prefix': weight_name,
                'folds_count': u_folds,
                'sample_count': len(valid_results)
            }

            # Calculate Mean and Std for all calculated metrics
            metrics_to_agg = ['test_psnr', 'test_ssim_3D', 'test_ssim_2D', 'test_mae', 'test_mse']
            for m in metrics_to_agg:
                stats_row[f'{m}_avg'] = df_weight[m].mean()
                stats_row[f'{m}_std'] = df_weight[m].std()

            all_summary_data.append(stats_row)

    # --- SAVE GLOBAL AGGREGATE ---
    if all_summary_data:
        df_aggregate = pd.DataFrame(all_summary_data)
        # Leaderboard sorted by PSNR Average
        if 'test_psnr_avg' in df_aggregate.columns:
            df_aggregate = df_aggregate.sort_values(by='test_psnr_avg', ascending=False)
        
        df_aggregate.to_csv(dest_path / f"{id_token}_aggregate.csv", index=False)

    # --- FINAL CONSOLE REPORT ---
    print(f"\n{'-'*40}\nEVALUATION COMPLETE\n{'-'*40}")
    print(f"Results saved to: {dest_path}")
    
    if fold_warnings:
        print(f"\n{Fore.RED}{Style.BRIGHT}🚨 FOLD DEFICIT WARNINGS (Under {EXPECTED_FOLDS}):")
        for warn in fold_warnings:
            print(f"  {Fore.RED}✖ {warn}")
    
    if error_list:
        print(f"\n{Fore.YELLOW}LOGGED ERRORS (First 10):")
        for err in list(set(error_list))[:10]:
            print(f"  - {err}")

# --- CONFIGURATION ---
TARGET_EXPERIMENTS = "ISBI_Final" # Folder with experiment subfolders
DESTINATION = "ISBI_TORCH"           # Where to save CSVs
TOKEN = "ASDASD"                        # Filename identifier

if __name__ == "__main__":
    run_evaluation(TARGET_EXPERIMENTS, DESTINATION, TOKEN)