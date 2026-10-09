#!/bin/bash
# Usage: ./register_3T_to_7T.sh <3T_folder> <7T_folder> <output_folder>

# Input args
dir3T=$1
dir7T=$2
outdir=$3

# Make output directory if it doesn't exist
mkdir -p $outdir

# Loop through all 3T scans
for file in "$dir3T"/*.nii.gz; do
    # Extract subject ID (filename without extension)
    sub=$(basename "$file" .nii.gz)

    in_3T="$dir3T/$sub.nii.gz"
    ref_7T="$dir7T/$sub.nii.gz"

    # Check if corresponding 7T exists
    if [ ! -f "$ref_7T" ]; then
        echo "⚠️  Skipping $sub — no matching 7T file found."
        continue
    fi

    echo "Registering 3T → 7T for subject: $sub"

    # Run FLIRT affine registration
    flirt \
      -in "$in_3T" \
      -ref "$ref_7T" \
      -out "$outdir/${sub}_3T_in_7T.nii.gz" \
      -omat "$outdir/${sub}_3T_to_7T.mat" \
      -dof 12 \
      -cost mutualinfo \
      -interp trilinear
done
