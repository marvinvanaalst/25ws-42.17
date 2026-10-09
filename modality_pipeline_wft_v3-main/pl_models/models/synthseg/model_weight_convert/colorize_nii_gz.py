import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def colorize_nifti(input_path, output_path, cmap='tab20'):
    nii = nib.load(input_path)
    data = nii.get_fdata()
    affine = nii.affine
    header = nii.header

    # Remove singleton dimensions if any
    data = np.squeeze(data)

    if data.ndim == 3:  # Case 1: Single-channel label image
        print("🧠 Detected: Single-channel label volume")

        data = data.astype(int)
        rgb_volume = np.zeros(data.shape + (3,), dtype=np.uint8)

        cmap_func = plt.get_cmap(cmap)
        labels = np.unique(data)
        max_label = labels.max() if labels.max() > 0 else 1

        for label in labels:
            mask = data == label
            color = (np.array(cmap_func(label / max_label)[:3]) * 255).astype(np.uint8)
            for c in range(3):
                rgb_volume[..., c][mask] = color[c]

        # Transpose to (3, D, H, W)
        rgb_volume = rgb_volume.transpose(3, 0, 1, 2)

    elif data.ndim == 4 and data.shape[0] == 3:  # Case 2: Already 3-channel RGB image
        print("🎨 Detected: 3-channel RGB volume")

        rgb_volume = data
        rgb_volume = np.clip(rgb_volume, 0, 255)

        if rgb_volume.dtype != np.uint8:
            rgb_volume = (rgb_volume / rgb_volume.max()) * 255
            rgb_volume = rgb_volume.astype(np.uint8)

    else:
        raise ValueError(f"Unsupported image shape: {data.shape}. Must be (D,H,W) or (3,D,H,W)")

    # Save RGB NIfTI
    rgb_nii = nib.Nifti1Image(rgb_volume, affine, header)
    nib.save(rgb_nii, output_path)
    print(f"✅ Saved RGB NIfTI to: {output_path}")

colorize_nifti('/Drive4T/inam/3T_7T_Synthesis/pl_models/models/synthseg/synthseg_output_weights.nii.gz', '/Drive4T/inam/3T_7T_Synthesis/pl_models/models/synthseg/synthseg_output_weights_colorized.nii.gz', cmap='tab20')