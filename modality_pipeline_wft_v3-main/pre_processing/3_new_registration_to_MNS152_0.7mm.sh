#!/bin/bash
# Usage: ./align_all.sh /path/to/input_dir /path/to/output_dir

INPUT_DIR=$1
OUTPUT_DIR=$2
REF=$FSLDIR/data/standard/downloaded/MNI152_T1_0.7mm_brain.nii.gz

mkdir -p ${OUTPUT_DIR}

for file in ${INPUT_DIR}/*.nii.gz; do
    fname=$(basename $file .nii.gz)

    echo "Processing $fname ..."

    flirt -in ${file} \
          -ref ${REF} \
          -out ${OUTPUT_DIR}/${fname}_to_MNI152_0.7mm.nii.gz \
          -omat ${OUTPUT_DIR}/${fname}_to_MNI152_0.7mm.mat \
          -dof 12 \
          -cost mutualinfo \
          -interp trilinear
done

echo "All scans aligned to MNI152 (0.7 mm)."
