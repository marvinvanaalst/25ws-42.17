import numpy as np
import nibabel as nib
from scipy.ndimage import binary_closing, label

def body_mask_ct(ct_path, hu_thresh=-500):
    # 1. Load
    img_nii = nib.load(ct_path)
    img = img_nii.get_fdata()

    # 2. Threshold to keep tissue
    mask = img > hu_thresh

    # 3. Fill small gaps
    mask = binary_closing(mask, structure=np.ones((3,3,3)))

    # 4. Largest connected component
    labels, num = label(mask)
    if num > 0:
        largest_label = np.bincount(labels.ravel())[1:].argmax() + 1
        mask = labels == largest_label

    return mask.astype(np.uint8), img_nii.affine