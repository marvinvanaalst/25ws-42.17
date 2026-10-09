#!/bin/bash

# Usage: ./rename_leaf_files.sh /path/to/parent_dir

parent_dir=$1

# Walk all subfolders
find "$parent_dir" -type f -name "*.nii.gz" | while read -r file; do
    dir=$(dirname "$file")                    # directory of file
    base=$(basename "$file" .nii.gz)          # filename without extension
    newname="${base:0:6}.nii.gz"             # first 13 chars + .nii.gz
    mv "$file" "$dir/$newname"                # rename
    echo "Renamed $file -> $dir/$newname"
done