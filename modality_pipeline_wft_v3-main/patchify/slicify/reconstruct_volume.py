import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm

def reconstruct_scan_from_slices(slice_dir, output_path="reconstructed.nii.gz"):
    slice_dir = Path(slice_dir)
    slice_files = sorted(slice_dir.glob("slice_*.pt"))  # sort ensures correct z-order

    volume_slices = []
    affine = None

    for slice_file in tqdm(slice_files, desc="Loading slices"):
        slice_dict = torch.load(slice_file, map_location='cpu')
        img_slice = slice_dict['data'].float()  # shape: (H, W)
        affine = slice_dict['affine'].float() 
        # print(img_slice.shape)
        # if affine is None and 'affine' in slice_dict:
        #     affine = slice_dict['affine']

        volume_slices.append(img_slice)  # add a new axis: (1, H, W)

    # Stack slices into 3D volume: (Z, H, W)
    volume = torch.cat(volume_slices, dim=0).permute(1, 2, 0).numpy()
    # volume = volume.transpose(2,1,0)
    # Make NIfTI
    nifti_img = nib.Nifti1Image(volume, affine)
    nib.save(nifti_img, output_path)
    print(f"Saved NIfTI: {output_path}")

if __name__ == "__main__":
    import argparse
    slice_dir = "/storage/an_inam/PGI_Project/Original_Shape_Data/SynthRadPGI/slices/256x512/MR/non_empty/1PC000"  # default directory for slices
    output_path = "0s2.nii.gz"
    
    reconstruct_scan_from_slices(slice_dir, output_path)
