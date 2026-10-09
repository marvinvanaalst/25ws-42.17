#!/bin/bash
# Usage: ./batch_bet.sh /path/to/input /path/to/output

# Input and output directories
input_dir=$1
output_dir=$2
mkdir -p "$output_dir"

# Loop through all .nii.gz files
for scan in "$input_dir"/*.nii.gz; do
    # Get filename without extension
    fname=$(basename "$scan" .nii.gz)

    echo "Processing $fname ..."

    # Run FSL BET
    bet "$scan" "$output_dir/${fname}_brain" -m -f 0.3

    # This produces:
    #   ${fname}_brain.nii.gz      → skull-stripped image
    #   ${fname}_brain_mask.nii.gz → binary brain mask
done

echo "All done. Results saved in $output_dir"
