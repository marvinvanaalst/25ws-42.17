#!/usr/bin/env python3
"""
train_loss_ablation.py
======================
Training driver for the 2.5D UNet Loss Ablation Study (15 Experiments).
Allows plug-and-play selection of any loss function from losses_library.py
while keeping all other settings (2.5D UNet, bilinear upsampling, Adam, learning rate, folds)
strictly identical to Experiment 6.

Key Features:
- Plug-and-play loss functions via losses_library.py
- Automatically converts outputs to PNG images instead of .nii.gz files:
    1. Side-by-side comparison PNG (Synthesized 7T | Ground Truth 7T | Error Heatmap)
    2. Multi-slice brain overview montage PNG (16 axial slices across depth)
    3. Individual 2D slice PNGs in logs/<exp>/generations/slices/<subject>/
- Optional --save_nii flag if .nii.gz files are also desired
- Disables blocking interactive prompts for headless/background runs
- Standardized logging of mean_psnr and mean_ssim
"""

import argparse
import csv
import json
import os
import runpy
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib


# Global cache for ground-truth reference volumes during evaluation
_GROUND_TRUTH_CACHE = {}


def setup_image_saving_hook(save_nii=False):
    """
    Hooks nibabel.save to generate PNG images instead of .nii.gz files.
    """
    real_nib_save = nib.save

    def hooked_save(img_obj, path):
        path_str = str(path)
        if path_str.endswith('.nii.gz') or path_str.endswith('.nii'):
            arr = img_obj.get_fdata(dtype=np.float32)
            gen_dir = os.path.dirname(path_str)
            filename = os.path.basename(path_str).replace('.nii.gz', '').replace('.nii', '')
            base_png_path = os.path.join(gen_dir, filename)

            # Robust intensity percentile normalization to [0, 1] for display
            p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99.5)
            if p99 > p1:
                arr_norm = np.clip((arr - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)
            else:
                arr_norm = np.clip(arr, 0.0, 1.0)

            depth = arr.shape[2]

            # 1. Save 16-slice brain overview montage (4x4 grid)
            try:
                slice_indices = [int(depth * frac) for frac in np.linspace(0.20, 0.80, 16)]
                fig, axes = plt.subplots(4, 4, figsize=(14, 14))
                for idx, ax in zip(slice_indices, axes.flatten()):
                    slice_img = np.rot90(arr_norm[:, :, idx])
                    ax.imshow(slice_img, cmap='gray')
                    ax.set_title(f"z={idx}", fontsize=10)
                    ax.axis('off')
                fig.suptitle(f"{filename}", fontsize=14, y=0.92)
                montage_file = f"{base_png_path}_montage.png"
                fig.savefig(montage_file, bbox_inches='tight', dpi=120)
                plt.close(fig)
                print(f"  [Image Saved] Montage: {montage_file}")
            except Exception as e:
                print(f"  [Warning] Failed saving montage: {e}")

            # 2. Save individual slice PNGs for brain volume
            try:
                slices_dir = os.path.join(gen_dir, "slices", filename)
                os.makedirs(slices_dir, exist_ok=True)
                start_z = int(depth * 0.15)
                end_z = int(depth * 0.85)
                for z in range(start_z, end_z, 2):
                    sl = np.rot90(arr_norm[:, :, z])
                    sl_uint8 = (sl * 255.0).astype(np.uint8)
                    Image.fromarray(sl_uint8).save(os.path.join(slices_dir, f"slice_{z:03d}.png"))
                print(f"  [Images Saved] Individual slices in: {slices_dir}/")
            except Exception as e:
                print(f"  [Warning] Failed saving slice PNGs: {e}")

            # 3. Cache original reference volume or create comparison figure if fake
            sub_id = None
            for token in filename.split('_'):
                if token.startswith('sub-'):
                    sub_id = token
                    break

            if "_original" in filename and sub_id:
                _GROUND_TRUTH_CACHE[sub_id] = arr_norm

            if "_fake" in filename and sub_id and sub_id in _GROUND_TRUTH_CACHE:
                try:
                    gt_arr = _GROUND_TRUTH_CACHE[sub_id]
                    comp_slices = [int(depth * f) for f in [0.35, 0.45, 0.55, 0.65]]
                    fig, axes = plt.subplots(len(comp_slices), 3, figsize=(15, 4 * len(comp_slices)))
                    cols = ['Synthesized 7T', 'Ground Truth 7T', 'Absolute Error |Fake - Real|']
                    for ax, col in zip(axes[0], cols):
                        ax.set_title(col, fontsize=14, fontweight='bold', pad=10)

                    for r_idx, z in enumerate(comp_slices):
                        fake_sl = np.rot90(arr_norm[:, :, z])
                        real_sl = np.rot90(gt_arr[:, :, z])
                        err_sl = np.abs(fake_sl - real_sl)

                        axes[r_idx, 0].imshow(fake_sl, cmap='gray')
                        axes[r_idx, 0].set_ylabel(f"Slice z={z}", fontsize=12)
                        axes[r_idx, 1].imshow(real_sl, cmap='gray')
                        im_err = axes[r_idx, 2].imshow(err_sl, cmap='inferno', vmin=0.0, vmax=0.30)
                        plt.colorbar(im_err, ax=axes[r_idx, 2], fraction=0.046, pad=0.04)

                        for c in range(3):
                            axes[r_idx, c].set_xticks([])
                            axes[r_idx, c].set_yticks([])

                    comp_file = f"{base_png_path}_comparison.png"
                    fig.savefig(comp_file, bbox_inches='tight', dpi=150)
                    plt.close(fig)
                    print(f"  [Image Saved] Comparison Figure: {comp_file}")
                except Exception as e:
                    print(f"  [Warning] Failed saving comparison figure: {e}")

            if save_nii:
                real_nib_save(img_obj, path)
            else:
                # Omit saving the large 21MB .nii.gz file to save disk space and prioritize PNG images
                pass
        else:
            real_nib_save(img_obj, path)

    nib.save = hooked_save


def main():
    parser = argparse.ArgumentParser(description="2.5D UNet Loss Ablation Training")
    parser.add_argument("--config", required=True, help="Path to base config JSON")
    parser.add_argument("--loss", required=True, help="Loss name from losses_library")
    parser.add_argument("--out_tag", default=None, help="Output tag for logs and checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Optional override for training epochs")
    parser.add_argument("--folds", type=int, default=10, help="Number of cross-validation folds (default: 10)")
    parser.add_argument("--save_nii", action="store_true", default=False, help="Also save .nii.gz files in addition to PNGs")
    args = parser.parse_args()

    loss_name = args.loss
    out_tag = args.out_tag if args.out_tag else loss_name

    print("============================================================")
    print(f"  Starting Loss Ablation Experiment: {loss_name}")
    print(f"  Output tag:  {out_tag}")
    print(f"  Base config: {args.config}")
    print(f"  CV Folds:    {args.folds}-Fold Cross-Validation")
    print(f"  Image Output: PNG images instead of .nii.gz (save_nii={args.save_nii})")
    print("============================================================")

    # 1. Load base configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    # 2. Apply ablation overrides
    config['experiment_name'] = out_tag
    if 'prompts' not in config:
        config['prompts'] = {}
    config['prompts']['confirm_overwrite'] = False

    # Set 10-Fold CV (1 test sample per fold)
    num_subjects = len(config['data']['training']['subjects'])
    test_per_fold = max(1, num_subjects // args.folds)
    config['training']['splits']['num_test_samples'] = test_per_fold
    actual_folds = num_subjects // test_per_fold
    print(f"  Configured: {actual_folds}-Fold CV ({num_subjects - test_per_fold} train : {test_per_fold} test per fold)")

    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
        print(f"  Overriding epochs to: {args.epochs}")

    # Ensure 2.5D UNet dataset loader is specified
    if config['model_info'].get('modelType') == 'pre_computed_slice2slice':
        if config['model_info'].get('is_25d', True):
            config['model_info']['modelType'] = 'pre_computed_slice2slice_25d'

    # 3. Instantiate custom loss from losses_library
    from losses_library import get_loss, _LOSS_REGISTRY
    if loss_name not in _LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{loss_name}'. Available:\n" +
                         "\n".join(f"  {k}" for k in _LOSS_REGISTRY))

    loss_fn_instance = get_loss(loss_name)
    print(f"  Successfully loaded loss module: {loss_fn_instance.__class__.__name__}")

    # 4. Patch compile_loss_fn in utils_custom.composite_loss so the pipeline uses our loss
    try:
        import utils_custom.composite_loss
        utils_custom.composite_loss.compile_loss_fn = lambda exp: loss_fn_instance
        print("  Patched utils_custom.composite_loss.compile_loss_fn successfully.")
    except Exception as e:
        print(f"  Warning: Could not patch utils_custom.composite_loss: {e}")

    # 5. Hook image saving: save PNGs instead of .nii.gz
    setup_image_saving_hook(save_nii=args.save_nii)
    print("  Hooked nibabel.save: Image outputs will be saved as PNGs instead of .nii.gz.")

    # 6. Write modified runtime config to configs/ablation_runtime/
    runtime_config_dir = os.path.join("configs", "ablation_runtime")
    os.makedirs(runtime_config_dir, exist_ok=True)
    runtime_config_path = os.path.join(runtime_config_dir, f"{out_tag}.json")
    with open(runtime_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Saved runtime config: {runtime_config_path}")

    # 7. Prepare sys.argv and invoke the master training script
    master_script_path = "master_half_slice_re.py"
    if not os.path.exists(master_script_path):
        hpc_path = "/home/ss_students/mtp/modality_pipeline_wft_v3-main/master_half_slice_re.py"
        if os.path.exists(hpc_path):
            master_script_path = hpc_path

    sys.argv = [master_script_path, runtime_config_path]
    print(f"  Executing master script: {master_script_path} with {runtime_config_path}...")

    # Run the training script in the current process space so patches stay active
    globs = runpy.run_path(master_script_path, run_name="__main__")

    # 8. Post-run metric summary extraction
    print("\n============================================================")
    print("  Training finished. Extracting metrics...")

    csv_path = f"logs/{out_tag}/metrics/average_testing_.csv"
    psnr_list, ssim_list = [], []

    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'test_psnr' in row and row['test_psnr']:
                        psnr_list.append(float(row['test_psnr']))
                    if 'test_ssim_3D' in row and row['test_ssim_3D']:
                        ssim_list.append(float(row['test_ssim_3D']))
        except Exception as e:
            print(f"  Error reading CSV {csv_path}: {e}")

    if not psnr_list and 'final_results' in globs:
        for r in globs['final_results']:
            if 'test_psnr' in r:
                psnr_list.append(float(r['test_psnr']))
            if 'test_ssim_3D' in r:
                ssim_list.append(float(r['test_ssim_3D']))

    if psnr_list:
        mean_psnr = float(np.mean(psnr_list))
        std_psnr = float(np.std(psnr_list))
        mean_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
        std_ssim = float(np.std(ssim_list)) if ssim_list else 0.0

        print("\n============================================================")
        print(f"  ABLATION RESULT SUMMARY: {loss_name}")
        print(f"  mean_psnr={mean_psnr:.4f}")
        print(f"  std_psnr={std_psnr:.4f}")
        print(f"  mean_ssim={mean_ssim:.4f}")
        print(f"  std_ssim={std_ssim:.4f}")
        print("============================================================\n")
    else:
        print("  No metrics could be automatically extracted from CSV or memory.")
        print(f"  Check logs/{out_tag} for details.")


if __name__ == "__main__":
    main()
