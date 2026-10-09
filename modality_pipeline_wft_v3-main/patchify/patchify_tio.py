import os
import json
import torch
import torchio as tio
from glob import glob
from tqdm import tqdm
from pathlib import Path
from scipy.ndimage import binary_closing, binary_fill_holes, generate_binary_structure
import numpy as np
import nibabel as nib
import numpy as np



def foreground_labelmap(scalar: tio.ScalarImage, closing_iterations: int = 2) -> tio.LabelMap:
    """Binary mask: 1 where scan > 0, morphologically closed and hole-filled."""
    data = scalar.data.numpy()                      # (C, W, H, D)
    mask = np.zeros_like(data, dtype=bool)

    structure = generate_binary_structure(rank=3, connectivity=1)   # 6-neighbourhood
    for c in range(data.shape[0]):
        m = data[c] > 0
        m = binary_closing(m, structure=structure, iterations=closing_iterations)
        m = binary_fill_holes(m, structure=structure)
        mask[c] = m

    return tio.LabelMap(
        tensor=torch.from_numpy(mask).to(torch.uint8),
        affine=scalar.affine,                       # must match the scalar image
    )

import re

def make_subject(filepath, config_dict, mask_generate=False, mask=False):
    img = nib.load(filepath)  
    rescale01 = tio.RescaleIntensity(out_min_max=(-1, 1), percentiles=tuple(config_dict['intensity_augmentations']['mri_intensity_percentile']))
    data = img.get_fdata(dtype=np.float32)

    v_min, v_max = data.min(), data.max()
    try:
        if not mask:
            print("normalized", filepath)
            scalar = rescale01(tio.ScalarImage(filepath))
        else:
            scalar = tio.ScalarImage(filepath)
        data = scalar.data.numpy()
        if mask_generate:
            closing_iterations = 2
            mask = np.zeros_like(data, dtype=bool)
            structure = generate_binary_structure(rank=3, connectivity=1)
            for c in range(data.shape[0]):
                m = data[c] > 0
                m = binary_closing(m, structure=structure, iterations=closing_iterations)
                m = binary_fill_holes(m, structure=structure)
                mask[c] = m
            label = tio.ScalarImage(
                tensor=torch.from_numpy(mask).to(torch.float),
                affine=scalar.affine,
            )
            return tio.Subject(image=label)
        subject = tio.Subject(image=scalar)
        return subject, v_min, v_max
    
    except Exception as e:
        raise RuntimeError(f"Failed to load image at {filepath}: {e}")





def maybe_resample(subject, resample_spacing):
    if resample_spacing is not None:
        try:
            resampler = tio.Resample(resample_spacing)
            subject = resampler(subject)
        except Exception as e:
            raise RuntimeError(f"Resampling failed: {e}")
    return subject


def save_patches(subject, subject_name, filepath, patch_size, patch_overlap, dest_dir, v_min, v_max):
    sampler = tio.GridSampler(subject, patch_size, patch_overlap)
    affine = subject.image.affine
    patch_output_dir = os.path.join(dest_dir, subject_name)
    os.makedirs(patch_output_dir, exist_ok=True)

    for i, patch in enumerate(sampler):
        try:
            patch_tensor = patch['image'][tio.DATA]  # (1, D, H, W)
            location = patch[tio.LOCATION].tolist()

            patch_data = {
                'patch': patch_tensor,
                'location': location,
                'v_min' : v_min,
                'v_max' : v_max,
                'affine': torch.from_numpy(affine),
                'filename': filepath
            }

            torch.save(patch_data, os.path.join(patch_output_dir, f'patch_{i:04d}.pt'))

        except Exception as e:
            print(f"[ERROR] Saving patch {i} for {filepath} failed: {e}")


def create_patchify_dataset(config_dict):
    scan_components = [config_dict['data']['direction']['source'], config_dict['data']['direction']['target'], config_dict['data']['direction']['mask_dir_name']]

    patch_size = config_dict['patchify']['patch_size']
    patch_overlap = config_dict['patchify']['patch_overlap']
    if 'intensity_augmentations' not in config_dict:
        config_dict['intensity_augmentations'] = {}
        config_dict['intensity_augmentations']['mri_intensity_percentile'] = [0, 100]
    intensity = config_dict['intensity_augmentations']['mri_intensity_percentile'][1]
    
    print("Target Scan Intensity = ", intensity)
    
    create_all = False
            # os.rmdir(f"{config_dict['data']['training']['subjects_path']}/patches/{patch_size}X{patch_overlap}")
            # print(f"Removed existing patches directory at {config_dict['data']['training']['subjects_path']}/patches/{patch_size}X{patch_overlap}.")
            # print("Recreating patches...")
    for scan_type in scan_components:
        local_continue_flag = False
        if os.path.exists(f"{config_dict['data']['training']['subjects_path']}/patches/{patch_size}X{patch_overlap}_intensity={intensity}/{scan_type}"):
            base_list = os.listdir(f"{config_dict['data']['training']['subjects_path']}/patches/{patch_size}X{patch_overlap}_intensity={intensity}/{scan_type}")
            for idx, ele in enumerate(config_dict['data']['training']['subjects']):
                if ele.split('.')[0] in base_list:
                    if idx+1 == len(config_dict['data']['training']['subjects']):
                        print(f"Skipping patching for {scan_type}. Patch directories for all scans in the config file were found in the destination directory:")
                        print("Path : ", f"{config_dict['data']['training']['subjects_path']}/patches/{patch_size}X{patch_overlap}_intensity={intensity}/{scan_type}\n\n")
                        local_continue_flag = True
                else:
                    print(f"Patches for {ele} not found")
        if local_continue_flag:
            local_continue_flag = False
            continue    
        print(f"Some/ all patches for {scan_type} are missing. Creating patches for {scan_type}.")
        
        
        
        # if scan_type == 'missing_mask': # Will be recalled only of check fails.
        #     print("Detected missing_mask as a modality. Generating mask.")
        #     source_dir = f"{config_dict['data']['training']['subjects_path']}/{scan_components[0]}"
        #     dest_dir = f"{config_dict['data']['training']['subjects_path']}/patches/{patch_size}X{patch_overlap}_intensity={intensity}/missing_mask"
        #     nii_files = sorted(glob(f"{source_dir}/*.nii.gz"))
        #     print(f"Processing {scan_type} scans with patch size {patch_size}, overlap {patch_overlap}, and intensity = {intensity} for mask creation.")
        #     print(f"Found {len(nii_files)} .nii.gz files in {source_dir}")
        #     if not nii_files:
        #         raise FileNotFoundError(f"No .nii.gz files found in {source_dir}")

        #     for filepath in tqdm(nii_files, desc="Processing volumes"):
        #         try:
        #             subject_name = Path(filepath).stem.replace(".nii", "")  # removes .nii.gz safely
        #             subject = make_subject(filepath,config_dict, mask_generate=True)
        #             # subject = maybe_resample(subject, resample_spacing)
        #             save_patches(subject, subject_name, filepath, patch_size, patch_overlap, dest_dir)
        #         except Exception as e:
        #             print(f"[ERROR] Processing {filepath} failed: {e}")
        #     continue
            
        source_dir = f"{config_dict['data']['training']['subjects_path']}/{scan_type}"
        dest_dir = f"{config_dict['data']['training']['subjects_path']}/patches/{patch_size}X{patch_overlap}_intensity={intensity}/{scan_type}"
        print("Source directory:", source_dir)
        nii_files = sorted(glob(f"{source_dir}/*.nii.gz"))
        print(nii_files)
        print(f"Processing {scan_type} scans with patch size {patch_size}, overlap {patch_overlap}, and intensity = {intensity}")
        print(f"Found {len(nii_files)} .nii.gz files in {source_dir}")
        
        if not nii_files:
            raise FileNotFoundError(f"No .nii.gz files found in {source_dir}")

        if re.search(r"\bmask\b", scan_type, re.IGNORECASE):
            this_mask = True
        else:
            this_mask = False
            
        for filepath in tqdm(nii_files, desc="Processing volumes"):
            try:
                subject_name = Path(filepath).stem.replace(".nii", "")  # removes .nii.gz safely
                subject, v_min, v_max = make_subject(filepath,config_dict, mask=this_mask)
                # subject = maybe_resample(subject, resample_spacing)
                save_patches(subject, subject_name, filepath, patch_size, patch_overlap, dest_dir, v_min, v_max)
            except Exception as e:
                print(f"[ERROR] Processing {filepath} failed: {e}")
                
    print(f"Finished patching the scans. Patches saved to {config_dict['data']['training']['subjects_path']}/patches/sizeXoverlap_{patch_size}X{patch_overlap}.")


if __name__ == "__main__":
    import sys
    main("/Drive4T/inam/3T_7T_Synthesis/patchify_tio_params.json")
    # if len(sys.argv) != 2:
    #     print("Usage: python precompute_patches.py config.json")
    # else:
    #     main(sys.argv[1])
