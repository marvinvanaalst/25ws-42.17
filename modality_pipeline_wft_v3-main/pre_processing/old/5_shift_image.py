import nibabel as nib
import numpy as np
import os
# Load original image
path = "/Drive4T/inam/MRMRData/T1/skull_stripped/7T_t1/n4_corrected/MNS_152_orig_ref"
dest_dir = "/Drive4T/inam/MRMRData/T1/skull_stripped/7T_t1/n4_corrected/MNS_152_orig_ref/shifted/"
os.makedirs(dest_dir, exist_ok=True)
all_scans = sorted([os.path.join(path,f) for f in os.listdir(path) if (f.endswith('.nii.gz') or f.endswith('.nii')) and f.startswith('native_res')])
dest_dir = sorted([os.path.join(dest_dir,os.path.basename(f)) for f in all_scans])


def shift_image(img_path, dest_path, voxel_shift=15):
    """
    
    Shift the image content posteriorly by a specified number of voxels.
    
    Parameters:
    img (nibabel.Nifti1Image): The input NIfTI image.
    voxel_shift (int): Number of voxels to shift posteriorly.
    
    Returns:
    nibabel.Nifti1Image: The shifted image.
    
    """
    img = nib.load(img_path)
    
    # Load the image data and affine
    data = img.get_fdata()
    affine = img.affine.copy()

    # Step 1: Shift the image content posteriorly → along voxel j-axis → axis=1
    shifted_data = np.roll(data, -voxel_shift, axis=1)
    shifted_data[:, -voxel_shift:, :] = 0  # Zero-pad front (anterior) part

    # Step 2: Update the affine to reflect the posterior movement
    affine[:3, 3] -= voxel_shift * affine[:3, 1]  # Adjust origin in j direction

    # Step 3: Create a new NIfTI image with the shifted data and updated affine
    shifted_img = nib.Nifti1Image(shifted_data, affine)
    
    nib.save(shifted_img, dest_path)


if __name__ == "__main__":
    for src, dest in zip(all_scans, dest_dir):
        print(f"Processing {src} to {dest}")
        shift_image(src, dest, voxel_shift=15)
    print("All images processed and saved.")