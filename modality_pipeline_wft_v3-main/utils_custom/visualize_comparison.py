import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchio as tio

def generate_comparison_plot(source_slice, synth_slice, target_slice, save_path, title="3T -> Synthetic 7T vs Ground Truth 7T"):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Normalize for display
    src = (source_slice - source_slice.min()) / (source_slice.max() - source_slice.min() + 1e-8)
    syn = (synth_slice - synth_slice.min()) / (synth_slice.max() - synth_slice.min() + 1e-8)
    tgt = (target_slice - target_slice.min()) / (target_slice.max() - target_slice.min() + 1e-8)
    diff = np.abs(syn - tgt)

    axes[0].imshow(src, cmap='gray')
    axes[0].set_title("Input 3T MRI")
    axes[0].axis('off')

    axes[1].imshow(syn, cmap='gray')
    axes[1].set_title("Synthetic 7T (Model)")
    axes[1].axis('off')

    axes[2].imshow(tgt, cmap='gray')
    axes[2].set_title("Actual 7T (Ground Truth)")
    axes[2].axis('off')

    im = axes[3].imshow(diff, cmap='hot')
    axes[3].set_title("Absolute Difference")
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Comparison image saved to: {save_path}")

if __name__ == '__main__':
    print("Visual comparison helper script ready!")
