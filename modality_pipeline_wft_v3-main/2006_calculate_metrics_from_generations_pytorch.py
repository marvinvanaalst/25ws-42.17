import os
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from monai.metrics import SSIMMetric
from colorama import Fore, Style, init

# Initialize colorama for colored terminal output
init(autoreset=True)

# --- CONFIGURATION ---
SOURCE_DIR = "ISBI_Final"
DEST_DIR = "ISBI_Final_Metrics_torch"
ID_TOKEN = "Final"
EXPECTED_FOLDS = 10  # The variable threshold for fold validation
# ---------------------

def get_triplanar_ssim_2d(preds, targets, ssim_2d_func):
    """Calculates 2D SSIM across Axial, Coronal, and Sagittal planes."""
    try:
        # Axial (D) - Dim 2
        axial = torch.mean(torch.stack([ssim_2d_func(preds[:,:,i,:,:], targets[:,:,i,:,:]) for i in range(preds.shape[2])]))
        # Coronal (H) - Dim 3
        coronal = torch.mean(torch.stack([ssim_2d_func(preds[:,:,:,i,:], targets[:,:,:,i,:]) for i in range(preds.shape[3])]))
        # Sagittal (W) - Dim 4
        sagittal = torch.mean(torch.stack([ssim_2d_func(preds[:,:,:,:,i], targets[:,:,:,:,i]) for i in range(preds.shape[4])]))
        return (axial + coronal + sagittal) / 3
    except Exception:
        return torch.tensor(0.0)

@torch.no_grad()
def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_path = Path(SOURCE_DIR)
    dest_path = Path(DEST_DIR)
    dest_path.mkdir(parents=True, exist_ok=True)

    # Initialize Metrics
    psnr_f = PeakSignalNoiseRatio(data_range=1.0).to(device)
    mae_f = MeanAbsoluteError().to(device)
    mse_f = MeanSquaredError().to(device)
    ssim_3d_f = SSIMMetric(spatial_dims=3, data_range=1.0, kernel_type='gaussian', win_size=11)
    ssim_2d_f = SSIMMetric(spatial_dims=2, data_range=1.0, kernel_type='gaussian', win_size=11)

    all_summary_data = []
    global_errors = []
    fold_warnings = []

    experiments = [d for d in target_path.iterdir() if d.is_dir()]
    
    # LEVEL 1: OVERALL EXPERIMENTS
    main_pbar = tqdm(experiments, desc=f"{Fore.CYAN}Experiments", unit="exp")
    for exp_folder in main_pbar:
        gen_dir = exp_folder / "generations"
        if not gen_dir.exists():
            global_errors.append(f"Folder Error: Missing 'generations' in {exp_folder.name}")
            continue

        fake_files = list(gen_dir.glob("*_fake.nii.gz"))
        if not fake_files:
            global_errors.append(f"File Error: No fake images in {exp_folder.name}")
            continue

        # Grouping by weight suffix
        weight_groups = {}
        for f_path in fake_files:
            try:
                parts = f_path.name.split('_')
                fold, sub_id = parts[0], f"{parts[1]}_{parts[2]}"
                weight_id = f_path.name.replace(f"{fold}_{sub_id}_", "").replace("_fake.nii.gz", "")
                orig_path = gen_dir / f"{fold}_{sub_id}_orig.nii.gz"
                
                if not orig_path.exists():
                    global_errors.append(f"Missing GT: {orig_path.name} in {exp_folder.name}")
                    continue

                if weight_id not in weight_groups: weight_groups[weight_id] = []
                weight_groups[weight_id].append({'fake': f_path, 'orig': orig_path, 'sub': sub_id, 'fold': fold})
            except Exception as e:
                global_errors.append(f"Parsing Error: {f_path.name} - {str(e)}")

        # LEVEL 2: WEIGHTS
        weight_pbar = tqdm(weight_groups.items(), desc=f"  {Fore.YELLOW}Weights", leave=False)
        for weight_name, tasks in weight_pbar:
            valid_results = []
            
            # LEVEL 3: VOLUMES
            vol_pbar = tqdm(tasks, desc=f"    {Fore.GREEN}Volumes", leave=False, unit="vol")
            for task in vol_pbar:
                try:
                    fake_v = torch.from_numpy(nib.load(task['fake']).get_fdata()).float().to(device).unsqueeze(0).unsqueeze(0)
                    orig_v = torch.from_numpy(nib.load(task['orig']).get_fdata()).float().to(device).unsqueeze(0).unsqueeze(0)

                    res = {
                        'subject': task['sub'], 'fold': task['fold'], 'weight_prefix': weight_name,
                        'test_psnr': psnr_f(fake_v, orig_v).item(),
                        'test_ssim_3D': ssim_3d_f(fake_v, orig_v).item(),
                        'test_ssim_2D': get_triplanar_ssim_2d(fake_v, orig_v, ssim_2d_f).item(),
                        'test_mae': mae_f(fake_v, orig_v).item(),
                        'test_mse': mse_f(fake_v, orig_v).item()
                    }
                    valid_results.append(res)
                except Exception as e:
                    global_errors.append(f"Runtime Error: {task['sub']} @ {weight_name}: {str(e)}")

            if not valid_results: continue

            # Compile Per-Weight CSV
            df_w = pd.DataFrame(valid_results)
            u_folds = df_w['fold'].nunique()
            total_s = len(df_w)
            
            # Record warning if fold count is low
            if u_folds < EXPECTED_FOLDS:
                fold_warnings.append(f"Experiment: {exp_folder.name} | Weight: {weight_name} | Folds: {u_folds}/{EXPECTED_FOLDS}")

            df_w['file_count_in_weight'] = total_s
            df_w['fold_count_in_weight'] = u_folds
            df_w.to_csv(dest_path / f"{exp_folder.name}_{weight_name}.csv", index=False)

            all_summary_data.append({
                'experiment': exp_folder.name, 'weight_prefix': weight_name,
                'folds': u_folds, 'samples': total_s,
                'test_psnr': df_w['test_psnr'].mean(),
                'test_ssim_3D': df_w['test_ssim_3D'].mean(),
                'test_ssim_2D': df_w['test_ssim_2D'].mean(),
                'test_mae': df_w['test_mae'].mean(),
                'test_mse': df_w['test_mse'].mean()
            })

    # FINAL AGGREGATION
    if all_summary_data:
        df_agg = pd.DataFrame(all_summary_data).sort_values(by='test_psnr', ascending=False)
        df_agg.to_csv(dest_path / f"{ID_TOKEN}_aggregate.csv", index=False)
        
        print(f"\n{Style.BRIGHT}{Fore.MAGENTA}🏆 LEADERBOARD (Sorted by PSNR):")
        print(df_agg[['experiment', 'weight_prefix', 'folds', 'test_psnr']].head(10).to_string(index=False))

    # ERROR & WARNING REPORT
    print(f"\n{'-'*60}")
    if global_errors:
        print(f"{Fore.YELLOW}LOGGED ERRORS ({len(global_errors)}):")
        for err in list(set(global_errors))[:10]: print(f"  - {err}")
    
    if fold_warnings:
        print(f"\n{Style.BRIGHT}{Fore.RED}🚨 FOLD COUNT ALERTS (Less than {EXPECTED_FOLDS} folds):")
        for warn in fold_warnings:
            print(f"  {Fore.RED}✖ {warn}")
    else:
        print(f"\n{Fore.GREEN}✔ All experiments met the {EXPECTED_FOLDS}-fold requirement.")
    print(f"{'-'*60}")

if __name__ == "__main__":
    run_evaluation()