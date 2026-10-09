import os
import glob
import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F

def minmax_masked_normalize(image_volume):
    min_val = image_volume.min()
    max_val = image_volume.max()
    image_volume = (image_volume - min_val) / (max_val - min_val)
    return image_volume

def extract_slice(tensor, axis, index):
    if axis == 0:
        return tensor[:, index, :, :]  # [C, Y, Z]
    elif axis == 1:
        return tensor[:, :, index, :]  # [C, X, Z]
    elif axis == 2:
        return tensor[:, :, :, index]  # [C, X, Y]
    else:
        raise ValueError(f"Invalid axis: {axis}")

def process_scan(config, scan_path, mask_path, output_dir, device='cuda', cache_in_ram=False, axis=2):
    scan_name = Path(scan_path).stem.replace(".nii", "")
    scan_out_dir_all = Path(output_dir) / 'all'/ scan_name
    scan_out_dir_non_empty = Path(output_dir)/ 'non_empty'/scan_name
    
    scan_out_dir_all.mkdir(parents=True, exist_ok=True)
    scan_out_dir_non_empty.mkdir(parents=True, exist_ok=True)
    

    # Load scan and mask
    scan_np = nib.load(scan_path).get_fdata()
    mask_np = nib.load(mask_path).get_fdata()
    affine = nib.load(scan_path).affine

    # Convert to tensor
    scan_tensor = torch.from_numpy(scan_np).float()
    mask_tensor = torch.from_numpy(mask_np).float()

    if not cache_in_ram:
        scan_tensor = scan_tensor.to(device=device, non_blocking=True)
        mask_tensor = mask_tensor.to(device=device, non_blocking=True)
    else:
        device = 'cpu'

    # Ensure shape: [C, X, Y, Z]
    if scan_tensor.ndim != 4:
        scan_tensor = scan_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0)

    scan_tensor = scan_tensor * mask_tensor  # Apply mask
    # scan_tensor = minmax_masked_normalize(scan_tensor)  # Normalize masked region

    num_slices = scan_tensor.shape[axis + 1]
    slice_iter = tqdm(range(num_slices), desc=f"Slicing {scan_name}", leave=False)
    

    for i in slice_iter:
        slice_tensor = extract_slice(scan_tensor, axis, i)  # [C, H, W]
        _, H, W = slice_tensor.shape
        original_size = torch.tensor([H, W], dtype=torch.int16, device=device)  # Save original shape
        if config['training']['change_slice_size']:
            k = config['training']['slice_size']
            if H < k or W < k:
                print(f"Skipping slice {i} of {scan_name}: too small ({H}x{W}) comprared to {k}x{k})")
                continue  # Skip slices smaller than 300x300
            # Crop to center 300x300
            start_y = (H - k) // 2
            start_x = (W - k) // 2
            slice_tensor = slice_tensor[:, start_y:start_y + k, start_x:start_x + k]

        # Slice wise normalization
        slice_tensor = (slice_tensor - slice_tensor.min())/(slice_tensor.max() - slice_tensor.min())  # Normalize to [0, 1]
        # Save data with float16 + original shape
        slice_dict = {
            "data": slice_tensor.to(dtype=torch.float16, device=device),
            "affine": torch.from_numpy(affine).to(dtype=torch.float16, device=device),
            "original_size": original_size  # Shape before cropping
        }


        torch.save(
            slice_dict,
            scan_out_dir_all/ f"slice_{i:04d}.pt",
            _use_new_zipfile_serialization=True
        )
        if slice_dict["data"].any():
            torch.save(
            slice_dict,
            scan_out_dir_non_empty /f"slice_{i:04d}.pt",
            _use_new_zipfile_serialization=True
        )
        slice_iter.set_postfix(slice=i, axis=axis, shape="300x300")


def process_all_scans_with_masks(config, scan_dir, mask_dir, output_dir, device='cuda', cache_in_ram=False, axis=2):
    scan_paths = glob.glob(os.path.join(scan_dir, "*.nii.gz"))
    scan_iter = tqdm(scan_paths, desc="Processing scans")

    for scan_path in scan_iter:
        scan_name = Path(scan_path).stem.replace(".nii", "")
        # print(f"Processing scan: {scan_name}")
        mask_path = os.path.join(mask_dir, f"{scan_name}.nii.gz")

        if not os.path.exists(mask_path):
            scan_iter.write(f"⚠️  Mask missing for: {scan_name}")
            continue

        scan_iter.set_postfix(scan=scan_name)

        process_scan(
            config=config,
            scan_path=scan_path,
            mask_path=mask_path,
            output_dir=output_dir,
            device=device,
            cache_in_ram=cache_in_ram,
            axis=axis
        )

if __name__ == "__main__":
    import argparse

    scan_dir = "/Drive4T/inam/MRMRData/T1/skull_stripped/Fin/3T_t1"
    mask_dir = "/Drive4T/inam/MRMRData/T1/skull_stripped/Fin/mask"
    output_dir = "/Drive4T/inam/MRMRData/T1/skull_stripped/Fin/slices/3T_t1"
    
    device = 'cuda'  # Default device
    cache_in_ram = False  # Default to not caching in RAM
    
    axix = 0  # Default slicing axis
    

    process_all_scans_with_masks(
        scan_dir=scan_dir,
        mask_dir=mask_dir,
        output_dir=output_dir,
        device=device,
        cache_in_ram=cache_in_ram,
        axis=axix
    )