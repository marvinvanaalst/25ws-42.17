#!/usr/bin/env python3
"""
Check FSL BET masks by creating residual images and computing validation metrics.

Usage:
  python check_mask_residuals.py /path/to/input /path/to/output
"""

import os
import sys
import nibabel as nib
import numpy as np
import csv

input_dir = sys.argv[1]
input_dir_2 = sys.argv[1]

output_dir = sys.argv[3]
os.makedirs(output_dir, exist_ok=True)

report_path = os.path.join(output_dir, "mask_validation_report.csv")


mod1_files = sorted(os.listdir(input_dir))
mod_2files = sorted(os.listdir(input_dir_2))

with open(report_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Scan", "Coverage", "ResidualFraction", "MaskVolume", "ScanVolume", "Flag"])

    for idx, fname in enumerate(sorted(os.listdir(input_dir))):
        if fname.endswith(".nii.gz"):
            base = fname.replace(".nii.gz", "")
            scan_path = os.path.join(input_dir, mod1_files[idx])
            mask_path = os.path.join(input_dir_2, mod_2files[idx])

            if not os.path.exists(mask_path):
                print(f"⚠️ No mask for {scan_path}, skipping")
                continue

            print(f"Processing {mod1_files[idx]} and {mod_2files[idx]} || saved as {base}")

            # Load scan and mask
            scan_nii = nib.load(scan_path)
            scan = scan_nii.get_fdata()
            maxx, minn = scan.max(), scan.min()
            scan = (scan-minn)/(maxx-minn)
            
            mask = nib.load(mask_path).get_fdata()

            # Compute residual
            residual = scan - (scan*mask)
            # residual = mask
            

            # Save residual image
            residual_nii = nib.Nifti1Image(residual, scan_nii.affine, scan_nii.header)
            nib.save(residual_nii, os.path.join(output_dir, f"{base}_residual.nii.gz"))

            # Metrics
            scan_nonzero = scan > 0
            mask_nonzero = mask > 0
            covered = np.logical_and(scan_nonzero, mask_nonzero)

            coverage = covered.sum() / scan_nonzero.sum() if scan_nonzero.sum() > 0 else 0
            residual_fraction = residual.sum() / scan.sum() if scan.sum() > 0 else 0
            mask_volume = mask_nonzero.sum()
            scan_volume = scan_nonzero.sum()

            # Helper flag
            flag = ""
            if coverage < 0.999 or residual_fraction > 0.05:
                flag = "⚠️ check"

            writer.writerow([base, f"{coverage:.4f}", f"{residual_fraction:.4f}",
                             mask_volume, scan_volume, flag])

print("✅ Done. Residuals + CSV report saved in", output_dir)
