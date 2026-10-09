#!/usr/bin/env python3
"""
2.5D Slice Adversarial Training Pipeline supporting LSGAN and WGAN.
- Generator: 2.5D U-Net (5 input channels -> 1 target channel)
- Discriminator: NLayerDiscriminator PatchGAN (input_nc=1, ndf=64, n_layers=3)
- Loss: L1 + Perceptual/Wavelet2D + Adversarial (LSGAN or WGAN)
- Optimizers:
    * LSGAN: Adam(beta1=0.5, beta2=0.999) for G and D
    * WGAN: RMSprop with weight clipping for G and D
"""
import sys
import os
import shutil
import warnings
import random
import json
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import nibabel as nib

warnings.simplefilter("default")

from utils_custom.utils_opus import model_fetcher
from utils_custom.eval_metrics import VolumeIQAMetrics
from utils_custom.composite_loss import compile_loss_fn
from utils_custom.csv_logger import CSVLogger
from utils_custom.LightningStyleCheckpoint import LightningStyleCheckpoint
from utils_custom.tensor_operations import restore_original_batch, load_and_normalize_slices
from pl_models.models.pixGAN import NLayerDiscriminator as PatchDiscriminator

argv = sys.argv
if len(argv) < 2:
    raise ValueError("Expected config file argument: python master_half_slice_adv_25d.py <config.json>")
config_path = argv[1]

with open(config_path, 'r') as f:
    experiment = json.load(f)

if 'intensity_augmentations' not in experiment:
    experiment['intensity_augmentations'] = {'mri_intensity_percentile': [0, 99.5]}

exp_name = experiment["experiment_name"]
DEVICE = experiment['device']['devices'][0]
print(f"=== Starting Adversarial 2.5D Training: {exp_name} on {DEVICE} ===")

gan_type = experiment.get('gan', {}).get('type', experiment['loss'].get('gan_type', 'lsgan')).lower()
adv_weight = experiment.get('gan', {}).get('adversarial_weight', experiment['loss'].get('adversarial_weight', 0.01))
disc_lr = experiment.get('gan', {}).get('disc_lr', experiment['loss'].get('disc_lr', 1e-4))
gen_lr = experiment['training']['learning_rate']
weight_decay = experiment['training'].get('weight_decay', 1e-5)

print(f"GAN Type: {gan_type.upper()}")
print(f"Generator LR: {gen_lr} | Discriminator LR: {disc_lr} | Adv Weight: {adv_weight}")

os.makedirs(f'logs/{exp_name}', exist_ok=True)
os.makedirs(f'logs/{exp_name}/generations', exist_ok=True)
os.makedirs(f'logs/{exp_name}/checkpoints', exist_ok=True)
os.makedirs(f'logs/{exp_name}/metrics', exist_ok=True)
shutil.copy(config_path, f"logs/{exp_name}/config.json")

iqa_metrics = VolumeIQAMetrics(data_range=1.0, device=DEVICE)
csv_logger_average = CSVLogger(log_dir=f'logs/{exp_name}/metrics', filename='average_testing_.csv', resume=False)

# Fetch generator architecture
model_initializer, model_loader_params = model_fetcher(experiment).fetch_model()
print(f"Generator Model Loader Params: {model_loader_params}")

num_subjects = len(experiment['data']['training']['subjects'])
num_test = experiment['training']['splits']['num_test_samples']
num_splits = num_subjects // num_test

final_results = []

for sdx in range(num_splits):
    sample_index = sdx
    if experiment.get("resume_from_sample", {}).get("do", False) and sample_index < experiment["resume_from_sample"]["index"]:
        print(f"Skipping split {sample_index} per resume config.")
        continue

    print(f"\n==========================================")
    print(f"     STARTING SPLIT {sample_index + 1} / {num_splits}")
    print(f"==========================================")

    csv_logger_train = CSVLogger(log_dir=f'logs/{exp_name}/metrics', filename=f'{sample_index}_metircs_train_.csv')
    csv_logger_testing = CSVLogger(log_dir=f'logs/{exp_name}/metrics', filename=f'{sample_index}_metircs_testing_.csv')

    # Data loading: 2.5D Caching Slice Dataloader
    from patchify.slicify.slice_checker import create_slicify_dataset
    create_slicify_dataset(experiment)

    if experiment['model_info']['modelType'] == 'pre_computed_slice2slice_25d':
        from data_modules.opus_CachingSliceDataset_2_5D import cache_slice_25d_dataloader
        cache_dl = cache_slice_25d_dataloader(experiment, DEVICE, idx=sample_index)
    else:
        from data_modules.opus_CachingSliceDataset import cache_slice_dataloader
        cache_dl = cache_slice_dataloader(experiment, DEVICE, idx=sample_index, cache_mode='same_gpu')

    temp = cache_dl.fetch_TrainValLoaders(build_train=experiment['training']['epochs'] > 0)
    train_dataloader = temp['train']
    val_dataloader = temp['validation']
    number_test_samples = temp['number_test_samples']

    # Instantiate Generator
    if len(model_loader_params.keys()) == 0:
        generator_model = model_initializer().to(DEVICE)
    else:
        generator_model = model_initializer(**model_loader_params).to(DEVICE)

    # Instantiate PatchGAN Discriminator (2D single-channel input: target or fake 7T slice)
    discimnator_model = PatchDiscriminator(
        input_nc=1,
        ndf=64,
        n_layers=3
    ).to(DEVICE)

    # Optimizers tuned per GAN formulation
    if gan_type == 'wgan':
        opt_g = torch.optim.RMSprop(generator_model.parameters(), lr=gen_lr, weight_decay=weight_decay)
        opt_d = torch.optim.RMSprop(discimnator_model.parameters(), lr=disc_lr)
    else:  # lsgan
        criterion_gan = nn.MSELoss()
        opt_g = torch.optim.Adam(generator_model.parameters(), lr=gen_lr, betas=(0.5, 0.999), weight_decay=weight_decay)
        opt_d = torch.optim.Adam(discimnator_model.parameters(), lr=disc_lr, betas=(0.5, 0.999))

    scalerG = GradScaler()
    scalerD = GradScaler()
    loss_fn = compile_loss_fn(experiment)

    checkpoint_callback = LightningStyleCheckpoint(
        save_dir=f'logs/{exp_name}/checkpoints/{sample_index}',
        monitor='val_loss',
        mode='min',
        top_k=experiment['checkpoint'].get('top_k', 1),
        save_every_n_epochs=experiment['checkpoint'].get('save_every_n_epochs', 50),
        save_weights_only=True,
        filename_template='{epoch:02d}-{monitor}={score:.4f}.pt'
    )

    epochs = experiment['training']['epochs']
    ckpt_dir = f"logs/{exp_name}/checkpoints/{sample_index}"
    final_epoch_ckpt = os.path.join(ckpt_dir, f"saveEvery_epoch_epoch={epochs}.pt")
    skip_training = os.path.exists(final_epoch_ckpt)

    if skip_training:
        print(f"Split {sample_index} already has final epoch {epochs} checkpoint. Proceeding directly to evaluation...")
    else:
        for epoch_idx in range(epochs):
            generator_model.train()
            discimnator_model.train()
            train_g_loss = 0.0
            train_d_loss = 0.0
            num_train_samples = 0

            train_bar = tqdm(train_dataloader, desc=f"Split {sample_index} | Ep {epoch_idx + 1}/{epochs}", leave=False)
            for batch in train_bar:
                src = batch['source']
                tgt = batch['target']
                mask = batch['mask']
                b_size = src.shape[0]

                # ------------------------------------------------------------------
                # 1. Train Discriminator
                # ------------------------------------------------------------------
                opt_d.zero_grad()
                with autocast():
                    # Real target
                    real_score = discimnator_model(tgt * mask)
                    # Fake generated (detached to prevent backprop to G)
                    with torch.no_grad():
                        fake = generator_model(src) * mask
                    fake_score = discimnator_model(fake.detach())

                    if gan_type == 'wgan':
                        d_loss = -torch.mean(real_score) + torch.mean(fake_score)
                    else:  # lsgan
                        d_real = criterion_gan(real_score, torch.ones_like(real_score))
                        d_fake = criterion_gan(fake_score, torch.zeros_like(fake_score))
                        d_loss = 0.5 * (d_real + d_fake)

                scalerD.scale(d_loss).backward()
                scalerD.step(opt_d)
                scalerD.update()

                if gan_type == 'wgan':
                    # Weight clipping for Lipschitz continuity
                    clip_value = 0.015
                    for p in discimnator_model.parameters():
                        p.data.clamp_(-clip_value, clip_value)

                # ------------------------------------------------------------------
                # 2. Train Generator
                # ------------------------------------------------------------------
                opt_g.zero_grad()
                with autocast():
                    fake = generator_model(src) * mask
                    fake_score_g = discimnator_model(fake)

                    if gan_type == 'wgan':
                        adv_loss = -torch.mean(fake_score_g)
                    else:  # lsgan
                        adv_loss = criterion_gan(fake_score_g, torch.ones_like(fake_score_g))

                    # Fidelity / Perceptual loss
                    fidelity_loss = loss_fn(fake.float(), tgt.float() * mask.float())
                    total_g_loss = fidelity_loss + adv_weight * adv_loss

                scalerG.scale(total_g_loss).backward()
                scalerG.step(opt_g)
                scalerG.update()

                train_g_loss += total_g_loss.item() * b_size
                train_d_loss += d_loss.item() * b_size
                num_train_samples += b_size
                train_bar.set_postfix({'g_loss': f"{total_g_loss.item():.4f}", 'd_loss': f"{d_loss.item():.4f}"})

            avg_g_loss = train_g_loss / max(1, num_train_samples)
            avg_d_loss = train_d_loss / max(1, num_train_samples)

            # ------------------------------------------------------------------
            # Validation Step
            # ------------------------------------------------------------------
            generator_model.eval()
            val_loss = 0.0
            num_val_samples = 0
            if len(val_dataloader) > 0:
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        with autocast():
                            v_src = val_batch['source']
                            v_tgt = val_batch['target']
                            v_mask = val_batch['mask']
                            v_fake = generator_model(v_src) * v_mask
                            v_loss = loss_fn(v_fake.float(), v_tgt.float() * v_mask.float())
                        val_loss += v_loss.item() * v_src.shape[0]
                        num_val_samples += v_src.shape[0]
                val_loss /= max(1, num_val_samples)
            else:
                val_loss = avg_g_loss

            epoch_metrics = {
                'epoch': epoch_idx + 1,
                'train_g_loss': avg_g_loss,
                'train_d_loss': avg_d_loss,
                'val_loss': val_loss
            }
            csv_logger_train.log(epoch_metrics)
            checkpoint_callback.save(
                model=generator_model,
                optimizer=opt_g,
                scheduler=None,
                scaler=scalerG,
                step=epoch_idx,
                epoch=epoch_idx + 1,
                split=sample_index,
                logs={'val_loss': val_loss}
            )

    # ------------------------------------------------------------------
    # Test Evaluation Across Checkpoint Weights
    # ------------------------------------------------------------------
    print(f"\nEvaluating Saved Checkpoints for Split {sample_index}...")
    ckpt_files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.pt')]

    for weight_path in sorted(ckpt_files):
        weight_prefix = os.path.basename(weight_path).replace('.pt', '')
        ckpt = torch.load(weight_path, weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        generator_model.load_state_dict(state_dict)
        generator_model.eval()

        pad_k = experiment['slicify']['pad_to']
        resize_k = experiment['slicify']['resize_to']

        for test_idx in range(number_test_samples):
            test_bundle = cache_dl.fetch_testLoader(idx=test_idx)
            test_name = test_bundle['test_name']
            test_loader = test_bundle['test']

            for first_batch in test_loader:
                original_path_scan = first_batch['path_scan'][0]
                original_path_mask = first_batch['path_mask'][0]
                break

            target_path_scan = original_path_scan.replace(
                experiment['data']['direction']['source'],
                experiment['data']['direction']['target']
            )
            scan_orig_full = nib.load(target_path_scan)
            mask_orig = torch.from_numpy(nib.load(original_path_mask).get_fdata(dtype=np.float32)).float().to(DEVICE)
            scan_orig = load_and_normalize_slices(experiment, target_path_scan, dtype=torch.float32).to(DEVICE)
            scan_orig_masked = scan_orig * mask_orig

            # Save ground truth NIfTI once per split
            orig_nii_path = f"logs/{exp_name}/generations/Split{sdx}_{test_idx}_{test_name}_original.nii.gz"
            if not os.path.exists(orig_nii_path):
                orig_img = nib.Nifti1Image(scan_orig_masked.squeeze(0).cpu().numpy(), affine=scan_orig_full.affine)
                nib.save(orig_img, orig_nii_path)

            # Generate synthetic 7T
            with torch.no_grad():
                pred_slices = []
                for b in test_loader:
                    with autocast():
                        p = generator_model(b['source'])
                    p_restored = restore_original_batch(p, b['padding'], b['original_size'], pad_k, resize_k)
                    pred_slices.append(p_restored.squeeze(1))

                pred_vol = torch.cat(pred_slices, dim=0).permute(1, 2, 0).contiguous()
                pred_vol = torch.clip(pred_vol * mask_orig, min=0.0, max=1.0)

                iqa_metrics.compute(pred_vol, scan_orig_masked)
                affine = first_batch['affine'][0].cpu().numpy().astype(np.float32)
                pred_img = nib.Nifti1Image(pred_vol.squeeze(0).cpu().numpy().astype(np.float32), affine)
                nib.save(pred_img, f"logs/{exp_name}/generations/Split{sdx}_{test_idx}_{test_name}_{weight_prefix}_fake.nii.gz")

        split_metrics = iqa_metrics.value()
        iqa_metrics.reset()
        print(f"Split {sample_index} [{weight_prefix}] -> PSNR: {split_metrics['PSNR']:.2f}, SSIM 3D: {split_metrics['SSIM_3D']:.4f}, SSIM 2D: {split_metrics['SSIM_2D']:.4f}, MAE: {split_metrics['MAE']:.4f}")

        row = {
            'sample_index': sample_index,
            'weight_prefix': weight_prefix,
            'test_psnr': split_metrics['PSNR'],
            'test_ssim_3D': split_metrics['SSIM_3D'],
            'test_ssim_2D': split_metrics['SSIM_2D'],
            'test_mae': split_metrics['MAE'],
            'test_mse': split_metrics['MSE']
        }
        final_results.append(row)
        csv_logger_testing.log(row)

    if experiment['training']['splits'].get('single_pass', False):
        print("Single pass mode completed.")
        break

print(f"\nAll training and evaluations completed for {exp_name}!")
