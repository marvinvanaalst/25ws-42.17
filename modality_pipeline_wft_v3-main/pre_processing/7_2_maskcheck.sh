#!/bin/bash
# Usage: ./mask_and_residual.sh /path/to/scans /path/to/masks /path/to/output

scan_dir=$1
mask_dir=$2
out_dir=$3

mkdir -p "$out_dir"

for scan in "$scan_dir"/*.nii.gz; do
    fname=$(basename "$scan")
    mask="$mask_dir/$fname"
    out_masked="$out_dir/${fname%.nii.gz}_masked.nii.gz"
    out_residual="$out_dir/${fname%.nii.gz}_residual.nii.gz"

    # masked brain
    fslmaths "$scan" -mas "$mask" "$out_masked"

    # residual = original - masked
    fslmaths "$scan" -sub "$out_masked" "$out_residual"
done
