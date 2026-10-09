import nibabel as nib
import numpy as np

base = "/home/ss_students/mtp/10_Pat_t1-20260830T152711Z-1-001/10_Pat_t1"

src = nib.load(f"{base}/3T_7TReg/sub-01_T1w.nii.gz")
tgt = nib.load(f"{base}/7T/sub-01_T1w.nii.gz")
mask = nib.load(f"{base}/MASK3T_7TReg/sub-01_T1w.nii.gz")

print(f"SOURCE (3T registered) shape: {src.shape}")
print(f"SOURCE zooms (mm/voxel): {src.header.get_zooms()}")
print(f"TARGET (7T) shape: {tgt.shape}")
print(f"TARGET zooms (mm/voxel): {tgt.header.get_zooms()}")
print(f"MASK shape: {mask.shape}")

print(f"\nTotal subjects: 10 (sub-01 to sub-10)")
print(f"Cross-val: 5-fold, 2 test subjects per fold, 8 training subjects per fold")
print(f"Number of slices per subject (axial): {src.shape[2]}")
