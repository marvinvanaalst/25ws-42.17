import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from multiprocessing import Pool, cpu_count
from functools import partial

def save_single_slice(i, img_in, img_gt, img_syn, error_vol, axis, output_dir, global_metrics):
    """Worker function to process and save a single slice."""
    # Extract slices
    if axis == 0:
        s_in, s_gt, s_syn, s_err = img_in[i,:,:], img_gt[i,:,:], img_syn[i,:,:], error_vol[i,:,:]
    elif axis == 1:
        s_in, s_gt, s_syn, s_err = img_in[:,i,:], img_gt[:,i,:], img_syn[:,i,:], error_vol[:,i,:]
    else:
        s_in, s_gt, s_syn, s_err = img_in[:,:,i], img_gt[:,:,i], img_syn[:,:,i], error_vol[:,:,i]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    titles = ['INPUT', 'GROUND TRUTH', 'SYNTHETIC', 'ERROR (GT - SYN)']
    data = [s_in, s_gt, s_syn, s_err]
    cmaps = ['gray', 'gray', 'gray', 'hot']

    for ax, slice_data, title, cmap in zip(axes, data, titles, cmaps):
        ax.imshow(np.rot90(slice_data), cmap=cmap)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

    # Add Global Metrics to the image
    fig.suptitle(f"Global Volume Metrics: PSNR: {global_metrics['psnr']:.2f} dB | SSIM (3D): {global_metrics['ssim']:.4f}", 
                 fontsize=18, y=1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"comparison_slice_{i:03d}.png"), bbox_inches='tight', dpi=120)
    plt.close(fig)

def generate_comparison_multicore(input_path, gt_path, syn_path, output_dir, axis=2):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load volumes
    print("Loading NIfTI volumes...")
    img_in = nib.load(input_path).get_fdata()
    img_gt = nib.load(gt_path).get_fdata()
    img_syn = nib.load(syn_path).get_fdata()

    # Calculate Global 3D Metrics
    print("Calculating 3D Metrics (this may take a moment)...")
    # Note: data_range is the difference between max and min possible pixel values
    data_range = img_gt.max() - img_gt.min()
    v_psnr = psnr(img_gt, img_syn, data_range=data_range)
    v_ssim = ssim(img_gt, img_syn, data_range=data_range)
    
    global_metrics = {'psnr': v_psnr, 'ssim': v_ssim}

    # Prepare Error Volume
    error_vol = np.abs(img_gt - img_syn)
    error_vol = error_vol / (np.max(error_vol) if np.max(error_vol) > 0 else 1)

    num_slices = img_in.shape[axis]
    
    # Multiprocessing setup
    num_workers = min(8,cpu_count()-150)
    print(f"Starting multicore processing with {num_workers} cores for {num_slices} slices...")
    
    # Use partial to pass constant arguments to the worker
    worker_func = partial(save_single_slice, 
                          img_in=img_in, img_gt=img_gt, img_syn=img_syn, 
                          error_vol=error_vol, axis=axis, 
                          output_dir=output_dir, global_metrics=global_metrics)

    with Pool(processes=num_workers) as pool:
        pool.map(worker_func, range(num_slices))

    print(f"Success! All slices saved to: {output_dir}")


gen_samples = "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/30_ESAU_3D_L1_32_trilinear/generations"

input_nifti = "/storage/an_inam/MR2MR/Data/10_Pat_t1/3T_7TReg/sub-01_T1w.nii.gz"
gt_nifti = f"{gen_samples}/0_sub-01_T1w_orig.nii.gz"
syn_nifti = f"{gen_samples}/0_sub-01_T1w_saveEvery_epoch_epoch=300.pt_fake.nii.gz"
output_folder = "DoctorResults"


if __name__ == "__main__":
    # Ensure you are inside __main__ block for multiprocessing to work on Windows/macOS
    generate_comparison_multicore(input_nifti, gt_nifti, syn_nifti, "Doctor_2")

# --- Execution ---
# Update these paths to your actual files

generate_comparison_slices(input_nifti, gt_nifti, syn_nifti, output_folder, axis=2)