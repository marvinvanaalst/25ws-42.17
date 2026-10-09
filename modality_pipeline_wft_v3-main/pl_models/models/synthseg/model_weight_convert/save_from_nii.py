import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# Step 1: Load the .nii.gz file
nifti = nib.load("/Drive4T/inam/3T_7T_Synthesis/pl_models/models/synthseg/synthseg_output.nii.gz")  # Replace with actual path
image = nifti.get_fdata()  # 3D image (H, W, D)

# Step 2: Extract the middle slice along Z-axis
mid_slice_index = image.shape[2] // 2
mid_slice = image[:, :, mid_slice_index]

# Step 3: Normalize the slice for color mapping (0 to 1)
slice_norm = (mid_slice - np.min(mid_slice)) / (np.ptp(mid_slice))

# Step 4: Convert to RGB using a matplotlib colormap
cmap = plt.get_cmap('magma')  # Try 'magma', 'hot', etc.
slice_rgb = cmap(slice_norm)[:, :, :3]  # Drop alpha channel (RGBA → RGB)

# Step 5: Save the RGB image
plt.imsave("middle_slice_colored.png", slice_rgb)
