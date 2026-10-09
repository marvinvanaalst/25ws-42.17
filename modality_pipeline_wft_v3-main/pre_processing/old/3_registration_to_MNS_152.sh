#!/bin/bash
# Register both 3T and 7T images to MNI152
# Ensure FSL is sourced

source_path_3T="/Drive4T/inam/MRMRData/T1/skull_stripped/3T_t1/n4_corrected"
source_path_7T="/Drive4T/inam/MRMRData/T1/skull_stripped/7T_t1/n4_corrected"

destination_path_3T="/Drive4T/inam/MRMRData/T1/skull_stripped/3T_t1/n4_corrected/MNS_152"
destination_path_7T="/Drive4T/inam/MRMRData/T1/skull_stripped/7T_t1/n4_corrected/MNS_152"

mkdir -p "$destination_path_3T"
mkdir -p "$destination_path_7T"

for img in $source_path_3T/*.nii.gz; do
    fname=$(basename "$img")
    flirt -in $img -ref $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz -out $destination_path_3T/152_$fname
done

for img in $source_path_7T/*.nii.gz; do
    fname=$(basename "$img")
    flirt -in $img -ref $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz -out $destination_path_7T/152_$fname
done
