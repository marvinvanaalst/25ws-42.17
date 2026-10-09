"""Volume reconstruction + metric logic for the slice (2D) pipeline.

Mirrors `components/testing_logic_patch.py`, but instead of a torchio
GridSampler/GridAggregator round-trip it walks the per-subject slice loader
produced by `data_modules.opus_CachingSliceDataset.cache_slice_dataloader`,
undoes the slicify pad/resize per slice and re-stacks the volume along the
original slice axis.

Both the periodic in-training sampling and the final per-weight evaluation in
`master_half_slice_new.py` go through `SliceVolumePredictor`, so the two can
never drift apart.
"""

import os

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from augmentations.pre_post_norm import normalize_per_sample_masked, denormalize
from augmentations.tta import tta_forward_2d
from utils_custom.csv_logger import CSVLogger
from utils_custom.eval_metrics import VolumeIQAMetrics
from utils_custom.tensor_operations import restore_original_batch, load_and_normalize_slices


# torch.quantile refuses inputs above 2**24 elements; fall back to numpy there.
_QUANTILE_LIMIT = 2 ** 24


def safe_quantile(x, q):
    """torch.quantile that survives whole-brain volumes."""
    if x.numel() == 0:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    if x.numel() <= _QUANTILE_LIMIT:
        return torch.quantile(x, q)
    value = np.quantile(x.detach().float().cpu().numpy(), q)
    return torch.tensor(float(value), device=x.device, dtype=x.dtype)


def slicify_normalize(volume, mask, percentile_clip=None):
    """Reproduce `patchify/slicify/slicify_nii.py::process_scan` normalisation.

    As `process_scan` currently stands: mask the volume, then a single global
    min-max to [0, 1]. The percentile clamp in that file is commented out, so it
    is off here too unless `evaluation.slicify_percentile_clip` turns it back on
    (set it to 0.995 if you re-enable the clamp in slicify_nii.py).

    This has to track slicify exactly. The cached target slices the model was
    trained against were produced this way; normalising the reference volume any
    other way scores the model against a distribution it never saw.
    """
    volume = volume * mask
    if percentile_clip is not None and bool(volume.any()):
        upper = safe_quantile(volume[volume > 0], float(percentile_clip))
        volume = torch.clamp(volume, min=0.0, max=float(upper))
    v_min, v_max = volume.min(), volume.max()
    return (volume - v_min) / torch.clamp(v_max - v_min, min=1e-8)


class SliceVolumePredictor:
    """Runs a 2D model over one subject's slices and rebuilds the volume."""

    def __init__(self, config):
        self.config = config
        self.DEVICE = config['device']['devices'][0]
        self.single_channel_model = config['model_info'].get('single_channel_model', False)
        # See master_half_slice_re.py: [0, 1] everywhere except inside the
        # forward, where a [-1, 1] model gets its input scaled up and its
        # prediction scaled back down.
        self.output_range_01 = config['model_info'].get('output_range_01', True)
        self.slice_axis = config['data'].get('slice_axis', 2)

        self.tta_flip_enabled = config.get("tta_flips", {}).get("enable", False)
        self.pre_post_norm = config.get("pretrain_augments", {}).get("pre_post_norm", {}).get("apply", False)
        self.pre_post_norm_type = config.get("pretrain_augments", {}).get("pre_post_norm", {}).get("type", "minmax")
        self.pre_post_denormalize = config.get("pretrain_augments", {}).get("pre_post_norm", {}).get("denormalize_output", False)

        self.pad_to = config['slicify']['pad_to']
        self.resize_to = config['slicify']['resize_to']
        self.reference_norm = config.get("evaluation", {}).get("ground_truth_norm", "slicify")
        self.slicify_percentile_clip = config.get("evaluation", {}).get("slicify_percentile_clip", None)

    # -- forward helpers ---------------------------------------------------
    def _raw_forward(self, model, source):
        if not self.output_range_01:
            source = source * 2.0 - 1.0
        if self.single_channel_model:
            out = model(source.squeeze(1)).unsqueeze(1)
        else:
            out = model(source)
        if not self.output_range_01:
            out = (out + 1.0) / 2.0
        return out

    def forward_batch(self, model, source, mask=None):
        """One slice batch through the model, honouring pre/post norm + TTA."""
        norm_params = None
        if self.pre_post_norm and mask is not None:
            source, norm_params = normalize_per_sample_masked(
                x=source, mask=mask, method=self.pre_post_norm_type
            )

        if self.tta_flip_enabled:
            out = tta_forward_2d(model, source, forward=lambda t: self._raw_forward(model, t))
        else:
            out = self._raw_forward(model, source)

        out = out.float()
        if norm_params is not None and self.pre_post_denormalize:
            out = denormalize(out, norm_params, mask=mask)
        return out

    # -- volume assembly ---------------------------------------------------
    def _stack(self, slices):
        """[N, H, W] slices -> volume with the slices back on `slice_axis`."""
        volume = torch.cat(slices, dim=0)          # [N, H, W]
        return volume.movedim(0, self.slice_axis)  # e.g. axis=2 -> [H, W, N]

    def predict(self, model, slice_loader):
        """Reconstruct the fake volume plus its reference/mask/affine metadata.

        `slice_loader` must be a `type_load='all'` loader for a single subject
        (i.e. `cache_slice_dataloader.fetch_testLoader(...)['test']`), otherwise
        the slice order that the stacking relies on is meaningless.
        """
        meta = None
        for batch in slice_loader:
            meta = batch
            break
        if meta is None:
            raise ValueError("Empty slice loader — nothing to reconstruct.")

        path_scan = meta['path_scan'][0]
        path_mask = meta['path_mask'][0]
        # slices are cached per modality directory; swap source -> target to
        # find the ground-truth scan for this subject.
        path_target = path_scan.replace(
            self.config['data']['direction']['source'],
            self.config['data']['direction']['target'],
        )

        target_img = nib.load(path_target)
        affine = target_img.affine
        mask_volume = torch.from_numpy(
            nib.load(path_mask).get_fdata(dtype=np.float32)
        ).float().to(self.DEVICE)

        target_volume = torch.from_numpy(
            target_img.get_fdata(dtype=np.float32)
        ).float().to(self.DEVICE)

        if self.reference_norm == "slicify":
            reference = slicify_normalize(target_volume, mask_volume,
                                          percentile_clip=self.slicify_percentile_clip)
        elif self.reference_norm == "percentile":
            reference = load_and_normalize_slices(
                self.config, path_target, dtype=torch.float32
            ).to(self.DEVICE) * mask_volume
        else:
            raise ValueError(
                f"Unknown evaluation.ground_truth_norm={self.reference_norm!r}; "
                f"expected 'slicify' or 'percentile'."
            )

        predicted = []
        with torch.no_grad():
            for batch in slice_loader:
                source = batch['source'].to(self.DEVICE)
                mask = batch['mask'].to(self.DEVICE)
                with torch.autocast(device_type="cuda", enabled=str(self.DEVICE).startswith("cuda")):
                    fake = self.forward_batch(model, source, mask)
                fake = restore_original_batch(
                    fake.float(), batch['padding'], batch['original_size'],
                    self.pad_to, self.resize_to,
                )
                predicted.append(fake.squeeze(1))   # [B, H_orig, W_orig]

        fake_volume = self._stack(predicted).to(self.DEVICE)
        if fake_volume.shape != mask_volume.shape:
            raise ValueError(
                f"Reconstructed volume {tuple(fake_volume.shape)} does not match "
                f"mask {tuple(mask_volume.shape)}. Check data.slice_axis "
                f"(currently {self.slice_axis}) against how the slices were cached."
            )
        fake_volume = torch.clip(fake_volume * mask_volume, min=0.0, max=1.0)

        return {
            'fake': fake_volume,
            'reference': reference,
            'mask': mask_volume,
            'affine': affine,
            'path_target': path_target,
        }


class test_on_sample:
    """Periodic full-volume sampling during training (slice pipeline).

    API-compatible with `components/testing_logic_patch.py::test_on_sample`.
    """

    def __init__(self, config):
        self.config = config
        self.DEVICE = config['device']['devices'][0]
        self.exp_name = f'{config["experiment_name"]}'
        self.predictor = SliceVolumePredictor(config)

    def test_sample(self, model, sample_idx, epoch, test_dataloader, dir_name="training_eval"):
        was_training = model.training
        model.eval()

        base_dir = f'logs/{self.exp_name}/{dir_name}'
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(f'{base_dir}/{sample_idx}', exist_ok=True)

        csv_logger_testing = CSVLogger(
            log_dir=f'{base_dir}/', filename=f"{sample_idx}_metrics.csv", resume=True
        )
        iqa_metrics = VolumeIQAMetrics(data_range=1.0, device=self.DEVICE)

        loaders = test_dataloader if isinstance(test_dataloader, (list, tuple)) else [test_dataloader]
        pbar = tqdm(loaders, desc=f"Sampling volumes @ epoch {epoch}", leave=False)

        for entry in pbar:
            if entry is None:
                continue
            name = entry.get('test_name', 'sample')
            pbar.set_description(f"Evaluating {name} for epoch {epoch}")

            result = self.predictor.predict(model, entry['test'])
            iqa_metrics.compute(result['fake'], result['reference'], result['mask'])

            nib.save(
                nib.Nifti1Image(result['fake'].cpu().numpy().astype(np.float32), result['affine']),
                f"{base_dir}/{sample_idx}/{epoch}_{name}_fake.nii.gz",
            )
            original_path = f"{base_dir}/{sample_idx}/000_{name}_orig.nii.gz"
            if not os.path.exists(original_path):
                nib.save(
                    nib.Nifti1Image(result['reference'].cpu().numpy().astype(np.float32), result['affine']),
                    original_path,
                )

            metrics = iqa_metrics.value()
            iqa_metrics.reset()

            print(f"Validation Metrics @ epoch {epoch} ({name}): PSNR: {metrics['PSNR']}, "
                  f"SSIM_3D: {metrics['SSIM_3D']}, SSIM_2D: {metrics['SSIM_2D']}, "
                  f"MAE: {metrics['MAE']}, MSE: {metrics['MSE']}")

            csv_logger_testing.log({
                'epoch': epoch,
                'sample_index': sample_idx,
                'subject': name,
                'test_psnr': metrics['PSNR'],
                'test_ssim_3D': metrics['SSIM_3D'],
                'test_ssim_2D': metrics['SSIM_2D'],
                'test_mae': metrics['MAE'],
                'test_mse': metrics['MSE'],
            })

        if was_training:
            model.train()
