import os
import nibabel as nib
import numpy as np
from skimage.filters import threshold_otsu
from scipy.ndimage import label
from nibabel import Nifti1Image

def create_brain_mask(volume):
    """Generates binary brain mask using Otsu + LCC filtering"""
    flat = volume[volume > 0.0001].flatten()
    if len(flat) == 0:
        return np.zeros_like(volume, dtype=np.uint8)
    
    thresh = threshold_otsu(flat)
    binary = (volume > thresh).astype(np.uint8)
    
    # Keep only the largest connected component
    labeled, num = label(binary)
    if num == 0:
        return binaryq
    counts = np.bincount(labeled.flatten())
    counts[0] = 0  # ignore background
    largest_label = np.argmax(counts)
    mask = (labeled == largest_label).astype(np.uint8)
    return mask

# Path setup
input_dir = "/Drive4T/inam/datasets/4_3T_registered_to_7T_7TUnchanged/3T_7TReg"  # replace with your .nii.gz folder path
output_dir = "/Drive4T/inam/datasets/4_3T_registered_to_7T_7TUnchanged/MASK3T_7TReg"
os.makedirs(output_dir, exist_ok=True)

from tqdm import tqdm

loopy = tqdm(os.listdir(input_dir))
# Process all .nii.gz files
for idx, filename in enumerate(loopy):
    if not filename.endswith(".nii.gz"):
        continue
    path = os.path.join(input_dir, filename)
    img = nib.load(path)
    data = img.get_fdata()
    
    mask = create_brain_mask(data)
    
    # Save mask as .nii.gz
    mask_img = Nifti1Image(mask.astype(np.uint8), img.affine)
    nib.save(mask_img, os.path.join(output_dir, filename.replace('.nii.gz', '_mask.nii.gz')))

print(f"✅ Done. Masks saved to: {output_dir}")
