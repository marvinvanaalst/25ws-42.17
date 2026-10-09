#!/usr/bin/env python3
import os
import argparse


def rename_files(folder):
    for fname in os.listdir(folder):
        old_path = os.path.join(folder, fname)
        if not os.path.isfile(old_path):
            continue  # skip directories

        # Split into name and extension (handles .nii.gz correctly)
        if fname.endswith(".nii.gz"):
            name_part, ext = fname[:-7], ".nii.gz"
        else:
            name_part, ext = os.path.splitext(fname)

        # Keep only first 6 characters
        new_name = name_part[:6] + ext
        new_path = os.path.join(folder, new_name)

        # Rename only if new name is different
        if new_path != old_path:
            os.rename(old_path, new_path)
            print(f"Renamed: {fname} → {new_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename files keeping only first 6 characters")
    parser.add_argument("folder", help="Folder containing files")
    args = parser.parse_args()

    rename_files(args.folder)
