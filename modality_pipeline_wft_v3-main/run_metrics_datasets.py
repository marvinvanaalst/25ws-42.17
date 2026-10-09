import numpy as np
import nibabel as nib
from pathlib import Path
import pandas as pd


def analyze_dataset(root_dir):
    root = Path(root_dir)
    results = []

    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue

        # Collect per-dataset shape & spacing stats
        sagittal_sizes, coronal_sizes, axial_sizes = [], [], []
        spacing_x, spacing_y, spacing_z = [], [], []

        dataset_modality_results = []

        for modality_dir in sorted(dataset_dir.iterdir()):
            if not modality_dir.is_dir():
                continue

            modality_name = modality_dir.name
            min_vals, max_vals = [], []
            mr_p05, mr_p995 = [], []

            for scan_path in modality_dir.glob("*.nii*"):
                try:
                    img = nib.load(str(scan_path))
                    arr = img.get_fdata()
                    hdr = img.header

                    # Intensity stats
                    min_vals.append(np.min(arr))
                    max_vals.append(np.max(arr))

                    if "mr" in modality_name.lower():
                        p05, p995 = np.percentile(arr, [0.5, 99.5])
                        mr_p05.append(p05)
                        mr_p995.append(p995)

                    # Shape stats
                    sx, sy, sz = arr.shape
                    sagittal_sizes.append(sx)
                    coronal_sizes.append(sy)
                    axial_sizes.append(sz)

                    # Spacing stats
                    zooms = hdr.get_zooms()[:3]
                    spacing_x.append(zooms[0])
                    spacing_y.append(zooms[1])
                    spacing_z.append(zooms[2])

                except Exception as e:
                    print(f"Error reading {scan_path}: {e}")

            if min_vals and max_vals:
                dataset_modality_results.append({
                    "dataset": dataset_dir.name,
                    "modality": modality_name,
                    "intensity_min": float(np.min(min_vals)),
                    "intensity_max": float(np.max(max_vals)),
                    "mr_p0.5": float(np.min(mr_p05)) if mr_p05 else np.nan,
                    "mr_p99.5": float(np.max(mr_p995)) if mr_p995 else np.nan
                })

        # Dataset-level shape + spacing stats
        if sagittal_sizes:
            sag_min, sag_max = int(np.min(sagittal_sizes)), int(np.max(sagittal_sizes))
            cor_min, cor_max = int(np.min(coronal_sizes)), int(np.max(coronal_sizes))
            ax_min, ax_max = int(np.min(axial_sizes)), int(np.max(axial_sizes))

            spx_min, spx_max = float(np.min(spacing_x)), float(np.max(spacing_x))
            spy_min, spy_max = float(np.min(spacing_y)), float(np.max(spacing_y))
            spz_min, spz_max = float(np.min(spacing_z)), float(np.max(spacing_z))

            # Add dataset-level info to each modality row
            for row in dataset_modality_results:
                row.update({
                    "sagittal_min": sag_min, "sagittal_max": sag_max,
                    "coronal_min": cor_min, "coronal_max": cor_max,
                    "axial_min": ax_min, "axial_max": ax_max,
                    "spacing_x_min": spx_min, "spacing_x_max": spx_max,
                    "spacing_y_min": spy_min, "spacing_y_max": spy_max,
                    "spacing_z_min": spz_min, "spacing_z_max": spz_max,
                })
                results.append(row)

    return pd.DataFrame(results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze min/max intensity, shape, and spacing per dataset & modality")
    parser.add_argument("root_dir", type=str, help="Input root directory")
    parser.add_argument("--out_csv", type=str, default="99_metrics.csv", help="Optional output CSV")

    args = parser.parse_args()
    df = analyze_dataset(args.root_dir)
    print(df)

    df.to_csv(args.out_csv, index=False)
    print(f"Saved results to {args.out_csv}")
