import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np

def pad_batch_to_shape(x: torch.Tensor, target_shape: tuple):
    """
    Pads a (B, H, W) tensor with zeros to match the target shape (H_target, W_target).
    
    Args:
        x (torch.Tensor): Input tensor of shape (B, H, W)
        target_shape (tuple): Target (H_target, W_target)

    Returns:
        torch.Tensor: Zero-padded tensor of shape (B, H_target, W_target)
    """
    if x.ndim == 4:
        # If input is (B, C, H, W), squeeze to (B, H, W)
        x = x.squeeze(1)
    elif x.ndim != 3:
        raise ValueError(f"Expected input tensor to be 3D or 4D, but got {x.ndim}D tensor.")
    b, h_in, w_in = x.shape
    h_target, w_target = target_shape

    pad_h = max(h_target - h_in, 0)
    pad_w = max(w_target - w_in, 0)

    pad_top    = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left   = pad_w // 2
    pad_right  = pad_w - pad_left

    # Pad expects (left, right, top, bottom)
    # So we must reshape (B, H, W) → (B, 1, H, W) for padding, then back
    x = x.unsqueeze(1)  # (B, 1, H, W)
    x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    return x_padded.squeeze(1)  # Back to (B, H_target, W_target)


def restore_original_batch(slice_dicts, pad_to, resize_to):
    """
    Restores a batch of slices from stored dicts to original unpadded, unresized shapes.
    Assumes all slices in batch share same padding & resize settings.
    
    slice_dicts: list of dicts (each with "data", "padding", "original_size")
    pad_to: int or (H, W)
    resize_to: int or (H, W)
    """
    # Stack data into [B, C, H, W]
    batch_data = torch.stack([sd["data"].to(torch.float32) for sd in slice_dicts], dim=0)

    # Assume padding same for all
    padding = slice_dicts[0]["padding"].tolist()
    pad_left, pad_right, pad_top, pad_bottom = padding

    # Step 1: Undo resize → back to padded size
    if resize_to:
        if isinstance(pad_to, int):
            pad_to = (pad_to, pad_to)
        batch_data = F.interpolate(batch_data, size=pad_to, mode='bilinear', align_corners=False)

    # Step 2: Remove padding
    H_padded, W_padded = batch_data.shape[2], batch_data.shape[3]
    batch_data = batch_data[:, :, pad_top:H_padded-pad_bottom, pad_left:W_padded-pad_right]

    # Step 3: Check each slice against stored original_size
    for idx, sd in enumerate(slice_dicts):
        orig_h, orig_w = sd["original_size"].tolist()
        if batch_data[idx].shape[1] != orig_h or batch_data[idx].shape[2] != orig_w:
            raise ValueError(f"Slice {idx} restored to wrong size {batch_data[idx].shape[1:]} (expected {orig_h, orig_w})")

    return batch_data  # [B, C, H_orig, W_orig]


def load_and_normalize_slices(experiment, path, dtype=torch.float32):
    img = nib.load(path).get_fdata(dtype=np.float32)
    
    # # Move slice axis to front → shape: (num_slices, H, W)
    # img_swapped = np.moveaxis(img, axis, 0)

    # # Per-slice min/max → shape: (num_slices, 1, 1)
    # min_vals = img_swapped.min(axis=(1, 2), keepdims=True)
    # max_vals = img_swapped.max(axis=(1, 2), keepdims=True)

    # # Normalize with broadcasting, avoid divide-by-zero
    # normed = (img_swapped - min_vals) / np.clip(max_vals - min_vals, 1e-8, None)

    # # Move axis back to original position
    # img_norm = np.moveaxis(normed, 0, axis)
    # changed from per slice back to per scan
    TOP = experiment["intensity_augmentations"]["mri_intensity_percentile"][1]
    BOT = experiment["intensity_augmentations"]["mri_intensity_percentile"][0]
    upper_limit = np.percentile(img, TOP)
    img = np.clip(img, a_min=BOT, a_max=upper_limit)
    img_norm = (img-img.min())/(img.max()-img.min())
    
    return torch.from_numpy(img_norm).to(dtype)

import torch.nn.functional as F

def restore_original_batch(data_tensor, padding, original_size, pad_to, resize_to):
    """
    Reverse the preprocessing: resize back to padded size, then remove padding.
    
    data_tensor: [B, C, H, W] model output
    padding: torch.Tensor [B, 4] with (left, right, top, bottom) per sample
    original_size: torch.Tensor [B, 2] with (H_orig, W_orig) per sample
    pad_to: tuple (H_pad, W_pad) used during preprocessing ...... To conserve isotropic spacing, pad to this.
    resize_to: tuple (H_resize, W_resize) used during preprocessing...... Basically TrainDims, the dimension on which model will be trained.
    """
    if data_tensor.ndim == 4:
        B, C, _, _ = data_tensor.shape
    else:
        raise ValueError(f"Expected data_tensor to be 4D, but got {data_tensor.ndim}D tensor.")

    # Case 1: No Padding Done during pre-processing. 
    # ==> Only Resizing to original dims is required during evaluation.
    if resize_to==pad_to:
        restored = []
        for i in range(B):
            un_resized = F.interpolate(
                data_tensor[i:i+1], size=original_size[i].tolist(), mode='bilinear', align_corners=False
            )

            # Step 3: Safety check: match stored original size. Only reason why interpolling slice by slice.
            H_orig, W_orig = original_size[i].tolist()
            if un_resized.shape[2] != H_orig or un_resized.shape[3] != W_orig:
                raise ValueError(
                    f"Restoration mismatch for sample {i}: got {unpadded.shape[2:]}, expected {(H_orig, W_orig)}"
                )
            restored.append(un_resized)

        return torch.cat(restored, dim=0)
      

    # Case 2: Padding then resizing done during pre-processing.
    # ==> First resize back to pad_to dims. Then remove padding,
    # Step 1: Undo resize → back to padded dimensions
    if resize_to != pad_to:
        data_tensor = F.interpolate(
            data_tensor, size=pad_to, mode='bilinear', align_corners=False
        )
    else:
        raise RuntimeWarning("pad_to/ resize_to misbehaving.")

    # Check if padding is all zeros
    if isinstance(pad_to, int):
        pad_to = (pad_to, pad_to)
    if isinstance(padding, torch.Tensor):
        padding = padding.tolist()  
    # pad_flag = (torch.tensor(pad_to) != 0).any().item() or (padding != 0).any().item()
    
    # if not pad_flag:
    #     return data_tensor

    restored = []
    
    for i in range(B):
        pad_left, pad_right, pad_top, pad_bottom = padding[i]
        H_padded, W_padded = data_tensor.shape[2], data_tensor.shape[3]

        # Step 2: Remove padding
        unpadded = data_tensor[i:i+1, :, 
                               pad_top:H_padded-pad_bottom,
                               pad_left:W_padded-pad_right]

        # Step 3: Safety check: match stored original size
        H_orig, W_orig = original_size[i].tolist()
        if unpadded.shape[2] != H_orig or unpadded.shape[3] != W_orig:
            raise ValueError(
                f"Restoration mismatch for sample {i}: got {unpadded.shape[2:]}, expected {(H_orig, W_orig)}"
            )

        restored.append(unpadded)

    return torch.cat(restored, dim=0)