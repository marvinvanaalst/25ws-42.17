#!/bin/bash
# Register both 3T and 7T images to MNI152
# Ensure FSL is sourced

source_path_3T="/Drive4T/inam/MRMRData/T1/skull_stripped/3T_t1/n4_corrected"
source_path_7T="/Drive4T/inam/MRMRData/T1/skull_stripped/7T_t1/n4_corrected"

destination_path_3T="/Drive4T/inam/MRMRData/T1/skull_stripped/3T_t1/n4_corrected/MNS_152_orig_ref"
destination_path_7T="/Drive4T/inam/MRMRData/T1/skull_stripped/7T_t1/n4_corrected/MNS_152_orig_ref"

mkdir -p "$destination_path_3T"
mkdir -p "$destination_path_7T"

mni_ref="$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz"

# Register 3T
# for img in "$source_path_3T"/*.nii.gz; do
#     fname=$(basename "$img")
#     matfile=$(mktemp)  # Temporary matrix file

#     # Step 1: Estimate transform (native → MNI)
#     flirt -in "$img" -ref "$mni_ref" -omat "$matfile" -out "$destination_path_3T/152_$fname"

#     # Step 2: Apply transform, preserve native resolution
#     flirt -in "$img" -ref "$img" -applyxfm -init "$matfile" -out "$destination_path_3T/native_res_152_$fname"

#     rm "$matfile"
# done

# mni_ref="$FSLDIR/data/standard/MNI152_T1_0.5mm_brain.nii.gz"

# Register 7T
for img in "$source_path_7T"/*.nii.gz; do
    fname=$(basename "$img")
    matfile=$(mktemp)

    flirt -in "$img" -ref "$mni_ref" -omat "$matfile" -out "$destination_path_7T/152_$fname"

    flirt -in "$img" -ref "$img" -applyxfm -init "$matfile" -out "$destination_path_7T/native_res_152_$fname"

    rm "$matfile"
done
