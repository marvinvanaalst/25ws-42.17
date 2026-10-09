import nibabel as nib
import numpy as np
from pathlib import Path
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

def evaluate_scans(scan_paths, mask_dir):
    """
    Given a list of paths containing paired medical scans (real vs synthetic),
    compute masked MSE, MAE, PSNR, and SSIM for each pair.

    Args:
        scan_paths (list[str] or list[Path]): Paths to all scans (.nii.gz)
        mask_dir (str or Path): Directory containing masks with matching names

    Returns:
        dict: subject-wise metrics
    """
    mask_dir = Path(mask_dir)
    results = {}

    # Group scans by subject ID (based on 'sub-0?')
    grouped = {}
    for path in scan_paths:
        path = Path(path)
        subject_id = path.stem.split("_")[0]  # e.g. sub-01
        grouped.setdefault(subject_id, []).append(path)

    for subject, paths in grouped.items():
        # Identify real and synthetic scans
        real = next((p for p in paths if "orig" in p.name), None)
        synth = next((p for p in paths if "orig" not in p.name), None)

        if not (real and synth):
            print(f"Skipping {subject} — missing pair")
            continue

        # Load NIfTI volumes
        real_img = nib.load(real).get_fdata().astype(np.float32)
        synth_img = nib.load(synth).get_fdata().astype(np.float32)

        # Normalize to 0–1 range
        real_img = (real_img - real_img.min()) / (real_img.max() - real_img.min() + 1e-8)
        synth_img = (synth_img - synth_img.min()) / (synth_img.max() - synth_img.min() + 1e-8)

        # Load corresponding mask
        mask_path = mask_dir / f"{subject}_mask.nii.gz"
        if not mask_path.exists():
            print(f"Mask missing for {subject}")
            continue
        mask = nib.load(mask_path).get_fdata().astype(bool)

        # Apply mask
        real_masked = real_img * mask
        synth_masked = synth_img * mask

        # --- Compute metrics ---
        mse = mean_squared_error(real_masked, synth_masked)
        mae = np.mean(np.abs(real_masked - synth_masked))  # <- replaced sklearn
        psnr = peak_signal_noise_ratio(real_masked, synth_masked, data_range=1.0)
        ssim = structural_similarity(real_masked, synth_masked, data_range=1.0)

        results[subject] = {
            "MSE": float(mse),
            "MAE": float(mae),
            "PSNR": float(psnr),
            "SSIM": float(ssim)
        }

    return results


# # --- Example usage ---
# if __name__ == "__main__":
#     all_scans = list(Path("/path/to/scans").rglob("*.nii.gz"))
#     mask_dir = "/path/to/masks"

#     metrics = evaluate_scans(all_scans, mask_dir)

#     for subj, vals in metrics.items():
#         print(f"{subj}: {vals}")


# --- Example usage ---
if __name__ == "__main__":
    import os
    pathh = "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/isbi_logs"
    for ppaatthh in [f"{pathh}/{hehe}" for hehe in os.listdir(pathh)]:
    # Example: read all nii.gz under a folder
        all_scans = list(Path(ppaatthh).rglob("*.nii.gz"))
        mask_dir = "/storage/an_inam/MR2MR/Data/10_Pat_t1/MASK3T_7TReg"

        metrics = evaluate_scans(all_scans, mask_dir)

        # Print results
        for subj, vals in metrics.items():
            print(f"{subj}: {vals}")
