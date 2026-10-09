import os
import sys
import argparse
import glob
import numpy as np
import torch
import torchio as tio
import matplotlib.pyplot as plt

def generate_comparison_figure(src_2d, syn_2d, tgt_2d, save_path, title="3T -> Synthetic 7T vs Actual 7T"):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    
    # Min-max normalization for visualization
    def norm(img):
        img_min, img_max = img.min(), img.max()
        if img_max - img_min < 1e-8:
            return np.zeros_like(img)
        return (img - img_min) / (img_max - img_min)

    src_n = norm(src_2d)
    syn_n = norm(syn_2d)
    tgt_n = norm(tgt_2d)
    diff = np.abs(syn_n - tgt_n)

    axes[0].imshow(src_n, cmap='gray')
    axes[0].set_title("Input 3T MRI", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(syn_n, cmap='gray')
    axes[1].set_title("Synthetic 7T (Predicted)", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(tgt_n, cmap='gray')
    axes[2].set_title("Actual 7T (Ground Truth)", fontsize=12, fontweight='bold')
    axes[2].axis('off')

    im = axes[3].imshow(diff, cmap='inferno')
    axes[3].set_title("Absolute Error Map", fontsize=12, fontweight='bold')
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved visual comparison figure to: {save_path}")

def compare_nifti_volumes(src_nii, syn_nii, tgt_nii, output_dir, slice_indices=None):
    os.makedirs(output_dir, exist_ok=True)
    
    src_vol = tio.ScalarImage(src_nii).data.squeeze().numpy()
    syn_vol = tio.ScalarImage(syn_nii).data.squeeze().numpy()
    tgt_vol = tio.ScalarImage(tgt_nii).data.squeeze().numpy()

    depth = src_vol.shape[0] if src_vol.ndim == 3 else src_vol.shape[-1]
    
    if slice_indices is None:
        slice_indices = [int(depth * 0.35), int(depth * 0.50), int(depth * 0.65)]

    sample_name = os.path.basename(syn_nii).replace(".nii.gz", "").replace(".nii", "")

    for s_idx in slice_indices:
        if src_vol.ndim == 3:
            s_src = src_vol[s_idx, :, :]
            s_syn = syn_vol[s_idx, :, :]
            s_tgt = tgt_vol[s_idx, :, :]
        else:
            s_src = src_vol[:, :, s_idx]
            s_syn = syn_vol[:, :, s_idx]
            s_tgt = tgt_vol[:, :, s_idx]

        save_path = os.path.join(output_dir, f"{sample_name}_slice_{s_idx}.png")
        generate_comparison_figure(s_src, s_syn, s_tgt, save_path, title=f"Sample: {sample_name} | Slice {s_idx}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate 3T vs Synthetic 7T vs Actual 7T visual comparisons")
    parser.add_argument("--src", type=str, required=True, help="Path to input 3T NIfTI file")
    parser.add_argument("--syn", type=str, required=True, help="Path to predicted Synthetic 7T NIfTI file")
    parser.add_argument("--tgt", type=str, required=True, help="Path to ground truth 7T NIfTI file")
    parser.add_argument("--out", type=str, required=True, help="Output directory to save PNG images")
    args = parser.parse_args()

    compare_nifti_volumes(args.src, args.syn, args.tgt, args.out)
