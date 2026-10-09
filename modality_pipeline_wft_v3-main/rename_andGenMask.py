"""
Phase 1: strip the field-strength tag (the subdirectory name) from scan filenames.
Phase 2: generate foreground masks from ONE nominated subdirectory, using the
         renamed files.

Layout:
    ROOT_DIR/
        0.1T/  pat1_0.1T_substring.nii.gz  ->  pat1_substring.nii.gz
        3T/    pat9_3T_substring.nii.gz    ->  pat9_substring.nii.gz

Masks are written to MASK_DIR/<MASK_SOURCE_TAG>/.
"""

import os
import re
import numpy as np
import torch
import torchio as tio
from scipy.ndimage import binary_closing, binary_fill_holes, generate_binary_structure

# ---------------------------------------------------------------- arguments
ROOT_DIR           = "/home/ig/in_pgi/MRIxFields/release_20260414/Training_retrospective/T2WR"   # contains 0.1T/, 3T/, ...
MASK_SOURCE_TAG    = "7T"                  # ONLY this subdir is masked
MASK_DIR           = "/home/ig/in_pgi/MRIxFields/release_20260414/Training_retrospective/T2WR/MASK"       # output root for masks
CLOSING_ITERATIONS = 2                     # morphological closing radius
DO_RENAME          = True                  # False = dry run, print only
# ---------------------------------------------------------------------------


def strip_tag(filename: str, tag: str) -> str:
    """Remove the subdirectory tag from a filename, collapsing leftover separators."""
    t = re.escape(tag)
    stem = re.sub(rf"[_-]?{t}[_-]?", "_", filename, count=1)
    stem = re.sub(r"__+", "_", stem).strip("_")
    return stem or filename


def make_mask(filepath: str, closing_iterations: int = 2) -> tio.LabelMap:
    """Binary foreground mask: 1 where intensity > 0, closed and hole-filled."""
    scalar = tio.ScalarImage(filepath)
    data = scalar.data.numpy()
    mask = np.zeros_like(data, dtype=bool)
    structure = generate_binary_structure(rank=3, connectivity=1)

    for c in range(data.shape[0]):
        m = data[c] > 0
        m = binary_closing(m, structure=structure, iterations=closing_iterations)
        m = binary_fill_holes(m, structure=structure)
        mask[c] = m

    return tio.LabelMap(
        tensor=torch.from_numpy(mask).to(torch.uint8),
        affine=scalar.affine,
    )


def rename_pass():
    """Phase 1 — strip the tag from every .nii.gz in every subdirectory."""
    for tag in sorted(os.listdir(ROOT_DIR)):
        subdir_path = os.path.join(ROOT_DIR, tag)
        if not os.path.isdir(subdir_path):
            continue

        for filename in sorted(os.listdir(subdir_path)):
            if not filename.endswith(".nii.gz"):
                continue

            new_name = strip_tag(filename, tag)
            if new_name == filename:
                continue

            src = os.path.join(subdir_path, filename)
            dst = os.path.join(subdir_path, new_name)

            if not DO_RENAME:
                print(f"[dry-run] {tag}/{filename} -> {new_name}")
            elif os.path.exists(dst):
                print(f"[skip] target exists: {tag}/{new_name}")
            else:
                os.rename(src, dst)
                print(f"[rename] {tag}/{filename} -> {new_name}")


def mask_pass():
    """Phase 2 — mask only MASK_SOURCE_TAG, reading the (now renamed) files."""
    subdir_path = os.path.join(ROOT_DIR, MASK_SOURCE_TAG)
    if not os.path.isdir(subdir_path):
        raise FileNotFoundError(f"Mask source directory not found: {subdir_path}")

    out_dir = os.path.join(MASK_DIR)
    os.makedirs(out_dir, exist_ok=True)

    for filename in sorted(os.listdir(subdir_path)):   # already the new names
        if not filename.endswith(".nii.gz"):
            continue

        src = os.path.join(subdir_path, filename)
        try:
            mask = make_mask(src, CLOSING_ITERATIONS)
        except Exception as e:
            print(f"[error] {src}: {e}")
            continue

        mask_path = os.path.join(out_dir, filename)
        mask.save(mask_path)
        print(f"[mask]  {mask_path}")


def main():
    rename_pass()
    mask_pass()


if __name__ == "__main__":
    main()