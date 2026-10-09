import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


def compute_metrics(real, synth):
    """Compute MSE, MAE, PSNR, SSIM between two masked volumes."""
    mse = mean_squared_error(real, synth)
    mae = np.mean(np.abs(real - synth))
    data_range =  1.0
    psnr = peak_signal_noise_ratio(real, synth, data_range=data_range)
    ssim = structural_similarity(real, synth, data_range=data_range)
    return mse, mae, psnr, ssim


def evaluate_all(FF, mask_dir, output_csv="metrics_summary_datarange_bbdm.csv"):
    """
    For each folder in FF:
        - Compare matching scans from generations_orig and generations_synth
        - Apply corresponding mask
        - Compute MSE, MAE, PSNR, SSIM for each pair
        - Report mean and std across all scans
        - Save final summary as CSV
    """
    mask_dir = Path(mask_dir)
    all_folder_results = []

    for folder in FF:
        folder = Path(folder)
        orig_dir = folder / "generations_orig"
        synth_dir = folder / "generations_synth"

        if not (orig_dir.exists() and synth_dir.exists()):
            print(f"Skipping {folder} — missing required subfolders.")
            continue

        mse_list, mae_list, psnr_list, ssim_list = [], [], [], []

        for real_path in orig_dir.glob("*.nii.gz"):
            name = real_path.name
            synth_path = synth_dir / name
            mask_path = mask_dir / name

            if not (synth_path.exists() and mask_path.exists()):
                print(f"Skipping {name} — missing synth or mask.")
                continue

            # Load data
            real_img = nib.load(real_path).get_fdata().astype(np.float32)
            synth_img = nib.load(synth_path).get_fdata().astype(np.float32)
            mask = nib.load(mask_path).get_fdata().astype(bool)

            # Apply mask
            real_masked = real_img * mask
            synth_masked = synth_img * mask

            # Compute metrics
            mse, mae, psnr, ssim = compute_metrics(real_masked, synth_masked)
            mse_list.append(mse)
            mae_list.append(mae)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

        if mse_list:
            folder_metrics = {
                "Folder": folder.name,
                "MSE_mean": np.mean(mse_list),
                "MSE_std": np.std(mse_list),
                "MAE_mean": np.mean(mae_list),
                "MAE_std": np.std(mae_list),
                "PSNR_mean": np.mean(psnr_list),
                "PSNR_std": np.std(psnr_list),
                "SSIM_mean": np.mean(ssim_list),
                "SSIM_std": np.std(ssim_list),
            }
            all_folder_results.append(folder_metrics)

            print(f"\n📂 {folder.name}")
            for k, v in folder_metrics.items():
                if k != "Folder":
                    print(f"  {k}: {v:.6f}")
        else:
            print(f"No valid pairs found in {folder}")

    # --- Save all results to CSV ---
    if all_folder_results:
        df = pd.DataFrame(all_folder_results)
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Metrics summary saved to: {output_csv}")
    else:
        print("\n⚠️ No valid metrics to save.")

    return all_folder_results

# --- Example usage ---
if __name__ == "__main__":
    import os
    base_path = "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/isbi_logs_C"
    FF = [
       f"{base_path}/{file}" for file in os.listdir(base_path)
    ]
    
    FF = [
        '/storage/an_inam/datasets/BBDM_LANCZOS_EXP_1_shape_match'
    ]
    mask_dir = "/storage/an_inam/MR2MR/Data/10_Pat_t1/MASK3T_7TReg"

    results = evaluate_all(FF, mask_dir)
