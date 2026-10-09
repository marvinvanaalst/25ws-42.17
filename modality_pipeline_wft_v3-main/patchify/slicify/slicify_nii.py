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


def process_scan(config, scan_path, mask_path, output_dir, device='cuda', cache_in_ram=False, axis=1,
                 pad_to=None, resize_to=None):
    """
    Processes a scan into per-slice .pt files.
    Steps:
    1. Mask volume.
    2. Optionally crop (from config).
    3. Pad slice to pad_to (if given).
    4. Resize slice to resize_to (if given).
    5. Normalize to [0,1].
    6. Save with metadata: original_size, padding, affine.
    
    Just min max normalization+tractable padding
    """
    pad_to = config.get('slicify', {}).get('pad_to', None)
    resize_to = config.get('slicify', {}).get('resize_to', None)
    
    scan_name = Path(scan_path).stem.replace(".nii", "")
    scan_out_dir_all = Path(output_dir) / 'all' / scan_name
    scan_out_dir_non_empty = Path(output_dir) / 'non_empty' / scan_name
    
    scan_out_dir_all.mkdir(parents=True, exist_ok=True)
    scan_out_dir_non_empty.mkdir(parents=True, exist_ok=True)

    # Load scan and mask
    scan_img = nib.load(scan_path)
    mask_img = nib.load(mask_path)
    scan_np = scan_img.get_fdata()
    mask_np = mask_img.get_fdata()
    affine = scan_img.affine

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

    # Apply mask
    scan_tensor = scan_tensor * mask_tensor
    # if scan_tensor.any():
    #     upper_limit = torch.quantile(scan_tensor[scan_tensor > 0], 0.995)
    #     scan_tensor = torch.clamp(scan_tensor, min=0.0, max=upper_limit)
    
    # Per scan normalization
    scan_max = scan_tensor.max()
    scan_min = scan_tensor.min()
    scan_tensor = (scan_tensor-scan_min)/(scan_max-scan_min)
    # scan_tensor = (scan_tensor - 0.5)*2 #[-1,1]
    num_slices = scan_tensor.shape[axis + 1]
    slice_iter = tqdm(range(num_slices), desc=f"Slicing {scan_name}", leave=False)
    

    for i in slice_iter:
        slice_tensor = extract_slice(scan_tensor, axis, i)  # [C, H, W]
        _, H, W = slice_tensor.shape
        original_size = torch.tensor([H, W], dtype=torch.int16, device=device)

        # Optional cropping from config 
        # One sided crop
        # if config['training']['change_slice_size']:
        #     k = config['training']['slice_size']
        #     if H < k or W < k:
        #         print(f"Skipping slice {i} of {scan_name}: too small ({H}x{W}) vs {k}x{k})")
        #         continue
        #     start_y = (H - k) // 2
        #     start_x = (W - k) // 2
        #     slice_tensor = slice_tensor[:, start_y:start_y + k, start_x:start_x + k]
        #     _, H, W = slice_tensor.shape  # update size after crop

        # 1️⃣ Pad first
        padding = (0, 0, 0, 0)  # default (left, right, top, bottom)
        if isinstance(pad_to, int):
            pad_to = (pad_to, pad_to)
        if pad_to[0]>resize_to:

            pad_h = pad_to[0] - H
            pad_w = pad_to[1] - W
            if pad_h < 0 or pad_w < 0:
                raise ValueError(f"Slice {i} larger than pad_to size {pad_to}")
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            slice_tensor = F.pad(slice_tensor, (pad_left, pad_right, pad_top, pad_bottom),
                                 mode='constant', value=0)
            padding = (pad_left, pad_right, pad_top, pad_bottom)
            _, H, W = slice_tensor.shape  # update size after pad
            

        # 2️⃣ Resize second
        if resize_to:
            if isinstance(resize_to, int):
                slice_tensor = F.interpolate(slice_tensor.unsqueeze(0),
                                         size= (resize_to, resize_to),
                                         mode='bilinear',
                                         align_corners=False).squeeze(0)
            else:
                raise(ValueError("resize_to needs to be a int."))

        # # Per slice normalization
        # min_val, max_val = slice_tensor.min(), slice_tensor.max()
        # if (max_val - min_val) > 0:
        #     slice_tensor = (slice_tensor - min_val) / (max_val - min_val)
        

        # Save slice
        slice_dict = {
            "data": slice_tensor.to(dtype=torch.float16, device=device),
            "affine": torch.from_numpy(affine).to(dtype=torch.float32, device=device),
            "original_size": original_size,  # before pad & resize
            "padding": torch.tensor(padding, dtype=torch.int16, device=device),  # (left, right, top, bottom)
            "path_scan": scan_path,  # Store original path for reference
            "path_mask": mask_path  # Store original mask path
        }

        torch.save(slice_dict, scan_out_dir_all / f"slice_{i:04d}.pt", _use_new_zipfile_serialization=True)

        if slice_dict["data"].any():
            torch.save(slice_dict, scan_out_dir_non_empty / f"slice_{i:04d}.pt", _use_new_zipfile_serialization=True)

        slice_iter.set_postfix(slice=i, axis=axis, shape=f"{slice_tensor.shape[1]}x{slice_tensor.shape[2]}")



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
    
    axix = 2  # Default slicing axis
    

    process_all_scans_with_masks(
        scan_dir=scan_dir,
        mask_dir=mask_dir,
        output_dir=output_dir,
        device=device,
        cache_in_ram=cache_in_ram,
        axis=axix
    )