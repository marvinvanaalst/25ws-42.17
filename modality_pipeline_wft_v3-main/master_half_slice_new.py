"""Slice (2D) training master — feature parity with `master_half_patch.py`.

`master_half_patch.py` drives the 3D patch pipeline; this is the same driver for
the pre-computed slice pipeline (`pre_computed_slice2slice`). Everything the
patch master gained — tensorboard logging, resumable training state, warmup
scheduling, GIN/downsample/mask block augments, masked pre/post normalisation,
flip TTA, periodic full-volume sampling, weight-filtered evaluation — is carried
over here, with the 3D pieces swapped for their 2D counterparts:

    GIN3D / augment            -> GIN2D / augment_2d          (block_augments_2d)
    tta_forward  (dims 2,3,4)  -> tta_forward_2d (dims 2,3)
    tio GridSampler/Aggregator -> slicify pad/resize round-trip (testing_logic_slice)
    wavelet3D                  -> wavelet2D

The slice-only features of `master_half_slice_train.py` are all kept: kornia
spatial/intensity augmentation with static or decaying probability,
`single_channel_model`, the `fine_tunning.pretrain_*` evaluate-only mode, and
per-test-subject volume reconstruction.

Usage:
    python master_half_slice_new.py configs/New_slice/<config>.json
"""

import json
import os
import shutil
import sys
import warnings

import nibabel as nib
import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

warnings.simplefilter("default")  # or "always"

import random

import kornia.augmentation as K

from utils_custom.csv_logger import CSVLogger
from utils_custom.eval_metrics import VolumeIQAMetrics
from utils_custom.LightningStyleCheckpoint import LightningStyleCheckpoint
from utils_custom.load_weights import load_model_state, load_training_state

wand_boolean = False
print("WAND DB BOOLEAN", wand_boolean)

argv = sys.argv


def set_seed(seed: int = 42):
    random.seed(seed)                      # Python random module
    np.random.seed(seed)                   # Numpy
    torch.manual_seed(seed)                # CPU tensors
    torch.cuda.manual_seed(seed)           # Current GPU
    torch.cuda.manual_seed_all(seed)       # All GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures deterministic convs


if len(argv) < 2:
    raise ValueError("Expected argv config file. Using default configuration.")
else:
    experimnet_config = argv[1]

avg_weighting = False
if avg_weighting:
    from merge_model_weights import average_model_weights

with open(experimnet_config, 'r') as f:
    experiment = json.load(f)

if experiment.get("seed", {}).get("apply", False):
    set_seed(experiment["seed"].get("value", 42))
    print(f"Deterministic mode ON, seed={experiment['seed'].get('value', 42)}")

single_channel_model = experiment['model_info'].get('single_channel_model', False)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
if experiment['model_info']['modelType'] == 'vol2vol':
    raise ValueError("vol2vol is not handled by the slice master.")
elif experiment['model_info']['modelType'] in ('patch2patch', 'pre_computed_patch2patch'):
    raise ValueError(
        f"{experiment['model_info']['modelType']} is a 3D pipeline; use master_half_patch.py."
    )
elif experiment['model_info']['modelType'] == 'pre_computed_slice2slice':
    from utils_custom.utils_opus import model_fetcher
    model_initializer, model_loader_params = model_fetcher(experiment).fetch_model()
else:
    raise ValueError(
        f"Unsupported model type: {experiment['model_info']['modelType']}. "
        f"Expected one of {experiment['model_info'].get('possibleTypes')}."
    )

exp_name = f'{experiment["experiment_name"]}'

final_results = []

writer = SummaryWriter(log_dir=f"runs/{exp_name}")

if os.path.isdir(f'logs/{experiment["experiment_name"]}'):
    print(f"\n\n\033[1;94mDirectory .logs/{experiment['experiment_name']} already exists. Do you want to continue?\033[0m\n")
else:
    print("\n\nUnique experiment name. Proceeding with training.")

if os.path.exists(f"logs/{exp_name}") and experiment.get("prompts", {}).get("confirm_overwrite", True):
    input(f"logs/{exp_name} Already exists. Do you want to continue?")

os.makedirs(f'logs/{exp_name}', exist_ok=True)
os.makedirs(f'logs/{exp_name}/generations', exist_ok=True)
os.makedirs(f'logs/{exp_name}/checkpoints', exist_ok=True)
os.makedirs(f'logs/{exp_name}/metrics', exist_ok=True)

if wand_boolean:
    import wandb
    wandb.init(
        project=experiment["Project_Name"],
        name=experiment["experiment_name"],
        config=experiment,
    )

from utils_custom.composite_loss import compile_loss_fn

DEVICE = experiment['device']['devices'][0]
print(f"Using device: {DEVICE}")
cache_mode = experiment["device"].get("cache_stratergy", "same_gpu")

# ---------------------------------------------------------------------------
# Augmentation / normalisation switches (same keys as master_half_patch.py)
# ---------------------------------------------------------------------------
block_augments = experiment.get("pretrain_augments", {}).get("block_augments", {}).get("apply", False)
block_config = experiment.get("pretrain_augments", {}).get("block_augments", {}).get("config", {})

pre_post_norm = experiment.get("pretrain_augments", {}).get("pre_post_norm", {}).get("apply", False)
pre_post_norm_type = experiment.get("pretrain_augments", {}).get("pre_post_norm", {}).get("type", "minmax")
pre_post_denormalize = experiment.get("pretrain_augments", {}).get("pre_post_norm", {}).get("denormalize_output", False)

tta_flip_enabled = experiment.get("tta_flips", {}).get("enable", False)

from augmentations.block_augments_2d import GIN2D, augment_2d
from components.testing_logic_slice import SliceVolumePredictor, test_on_sample

val_test_sample = test_on_sample(experiment)
volume_predictor = SliceVolumePredictor(experiment)

if block_augments and block_config.get('gin_apply', False):
    print("\n\n\n GIN+MASK Augments enabled.")
    gin_net = GIN2D(in_channels=experiment['model'].get('in_channels', 1)).to(DEVICE)
else:
    gin_net = None

# GIN alone leaves corrupt_mask empty. Masking the loss by an all-zero map makes
# the loss identically 0, so only gate the loss on the corruption map when a
# *spatial* corruption actually ran.
spatial_corruption = bool(block_config.get('downsample_apply', False) or
                          block_config.get('mask_apply', False))
if block_augments and not spatial_corruption:
    print("[Augments] GIN-only corruption: loss stays on the full brain mask.")

if pre_post_norm:
    from augmentations.pre_post_norm import normalize_per_sample_masked, denormalize

# slicify only pads when pad_to > resize_to, and restore_original_batch only
# knows how to undo that ordering. Fail loudly instead of at reconstruction.
_pad_to = experiment['slicify']['pad_to']
_resize_to = experiment['slicify']['resize_to']
_pad_scalar = _pad_to[0] if isinstance(_pad_to, (list, tuple)) else _pad_to
if _pad_scalar < _resize_to:
    raise ValueError(
        f"slicify.pad_to ({_pad_scalar}) < slicify.resize_to ({_resize_to}). "
        f"Padding is skipped at slicify time but the restore path cannot undo "
        f"that combination — use pad_to >= resize_to."
    )

if os.path.exists(f"logs/{exp_name}/config.json"):
    print("Skipping config file copying,  source and destination are the same file.")
else:
    shutil.copy(experimnet_config, f"logs/{exp_name}/config.json")

iqa_metrics = VolumeIQAMetrics(data_range=1.0, device=DEVICE)

if experiment['resume_from_sample']['do']:
    csv_logger_average = CSVLogger(log_dir=f'logs/{exp_name}/metrics', filename=f'average_testing_.csv', resume=True)
else:
    csv_logger_average = CSVLogger(log_dir=f'logs/{exp_name}/metrics', filename=f'average_testing_.csv', resume=False)

if experiment.get('fine_tunning', {}).get('pretrain_evaluate_only', None) is True \
        and experiment.get('fine_tunning', {}).get('pretrain_boolean', None) is True:
    print("Pretrained-evaluation mode: forcing epochs to 0.")
    experiment['training']['epochs'] = 0

print("Total Number of samples", len(experiment['data']['training']['subjects']))
print("Total Splits", (len(experiment['data']['training']['subjects']) // experiment['training']['splits']['num_test_samples']))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def model_forward(model, source):
    """Single forward pass, honouring `model_info.single_channel_model`."""
    if single_channel_model:
        return model(source.squeeze(1)).unsqueeze(1)
    return model(source)


def to_device(tensor):
    """`cache_stratergy == same_gpu` already has the tensors here; this is a no-op then."""
    return tensor.to(DEVICE, non_blocking=True).float()


def build_augmenters(epoch_idx):
    """Kornia augmenters for this epoch, or None when augmentation is off.

    Three separate sequentials on purpose:
      * geometric — one sampled transform shared by source/target/mask, so the
        pair stays registered (applying the same sequential three times draws
        three *different* transforms and silently destroys the pairing).
      * erasing   — corrupts the model input only.
      * intensity — photometric jitter, applied to whichever side the config
        names (`intensity_apply_to`).
    """
    aug_cfg = experiment.get('augmentations', {})
    if not aug_cfg.get('apply', False):
        return None
    if aug_cfg.get('type', 'static') != "static" and epoch_idx >= aug_cfg['till']:
        return None

    aug_prob = aug_cfg['augmentation_prob']
    if aug_cfg.get('type', 'static') == "static":
        prob = aug_prob
    else:
        # linear decay to 0 at `till`
        prob = aug_prob * (1 - (epoch_idx + 1) / (aug_cfg['till'] + 1e-8))
    prob = float(min(max(prob, 0.0), 1.0))

    geometric = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=prob),
        K.RandomRotation(degrees=aug_cfg.get('rotation_degrees', 15.0), p=prob),
        *( [K.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1),
                           shear=10.0, p=prob)] if aug_cfg.get('affine', False) else [] ),
        data_keys=["input", "input", "mask"],
    ).to(DEVICE)

    erasing = None
    if aug_cfg.get('random_erasing', True):
        erasing = K.AugmentationSequential(
            K.RandomErasing(
                scale=tuple(aug_cfg.get('erasing_scale', (0.02, 0.5))),
                ratio=tuple(aug_cfg.get('erasing_ratio', (0.3, 3.3))),
                same_on_batch=False,
                p=prob,
            ),
            data_keys=["input"],
        ).to(DEVICE)

    intensity = K.AugmentationSequential(
        K.RandomBrightness(aug_cfg.get('brightness', 0.2), p=prob),
        K.RandomContrast(aug_cfg.get('contrast', 0.5), p=prob),
        *( [K.RandomGaussianNoise(mean=0.0, std=aug_cfg.get('noise_std', 0.05), p=prob)]
           if aug_cfg.get('gaussian_noise', False) else [] ),
        *( [K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=prob)]
           if aug_cfg.get('gaussian_blur', False) else [] ),
        data_keys=["input"],
    ).to(DEVICE)

    return {
        'prob': prob,
        'geometric': geometric,
        'erasing': erasing,
        'intensity': intensity,
        'intensity_apply_to': aug_cfg.get('intensity_apply_to', 'target'),
    }


def apply_augmenters(augmenters, source, target, mask):
    if augmenters is None:
        return source, target, mask

    source, target, mask = augmenters['geometric'](source, target, mask)
    # rotation interpolates the label map; put it back on {0, 1}
    mask = (mask > 0.5).to(source.dtype)

    if augmenters['erasing'] is not None:
        source = augmenters['erasing'](source)

    where = augmenters['intensity_apply_to']
    if where in ('source', 'both'):
        source = augmenters['intensity'](source)
    if where in ('target', 'both'):
        target = augmenters['intensity'](target)

    return source, target, mask


def get_lr_scheduler(optimizer, num_warmup_steps, steps_per_epoch, decay_cfg):
    """Linear warmup, optionally multiplied by a StepLR-style epoch decay.

    Folding the decay into the same LambdaLR keeps a single scheduler object,
    which is what the checkpoint / `load_training_state` round-trip stores.
    """
    apply_decay = bool(decay_cfg.get('apply', False))
    step_size = max(int(decay_cfg.get('step_size_epochs', 10)), 1)
    gamma = float(decay_cfg.get('gamma', 0.5))
    steps_per_epoch = max(int(steps_per_epoch), 1)

    def lr_lambda(current_step):
        factor = min(1.0, float(current_step) / float(max(1, num_warmup_steps)))
        if apply_decay:
            epoch = current_step // steps_per_epoch
            factor *= gamma ** (epoch // step_size)
        return factor

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Split loop
# ---------------------------------------------------------------------------
num_splits = len(experiment['data']['training']['subjects']) // experiment['training']['splits']['num_test_samples']

for sdx in range(num_splits):
    sample_index = sdx
    if experiment["resume_from_sample"]["do"] and sample_index < experiment["resume_from_sample"]["index"]:
        print(f"Skipping sample {sample_index} as per resume configuration.")
        continue

    # -- data ---------------------------------------------------------------
    from data_modules.opus_CachingSliceDataset import cache_slice_dataloader
    from patchify.slicify.slice_checker import create_slicify_dataset

    create_slicify_dataset(experiment)
    cache_dl = cache_slice_dataloader(experiment, DEVICE, idx=sample_index)
    temp = cache_dl.fetch_TrainValLoaders(build_train=experiment['training']['epochs'] > 0)
    train_dataloader = temp['train']
    val_dataloader = temp['validation']
    number_test_samples = temp['number_test_samples']
    # built lazily — only a run with sample_volume_during_training pays for it
    intermediate_testing_loader = None

    # -- checkpointing ------------------------------------------------------
    checkpoint_callback_2 = LightningStyleCheckpoint(
        save_dir=f'logs/{exp_name}/checkpoints/{sample_index}',
        monitor='val_loss',
        mode='min',
        top_k=experiment['checkpoint']['top_k'],
        save_every_n_epochs=experiment['checkpoint']['save_every_n_epochs'],
        save_weights_only=True,
        filename_template='{epoch:02d}-{monitor}={score:.4f}.pt'
    )
    checkpoint_callback_val_loss = LightningStyleCheckpoint(
        save_dir=f'logs/{exp_name}/checkpoints/{sample_index}',
        monitor='val_loss',
        mode='min',
        top_k=experiment['checkpoint']['top_k'],
        save_every_n_epochs=2000000,
        save_weights_only=False,
        filename_template='{epoch:02d}-{monitor}={score:.4f}.pt'
    )
    if avg_weighting:
        avg_annealing_checkpoint_callback = LightningStyleCheckpoint(
            save_dir=f'logs/{exp_name}/checkpoints_averaging/{sample_index}',
            monitor='val_loss',
            mode='min',
            save_every_n_epochs=max(experiment['checkpoint']['save_every_n_epochs'] // 2, 1),
            save_weights_only=True,
            filename_template='{epoch:02d}-{monitor}={score:.4f}.pt'
        )

    scaler = GradScaler()
    loss_fn = compile_loss_fn(experiment)

    # -- model --------------------------------------------------------------
    print(model_loader_params)
    if len(model_loader_params.keys()) == 0:
        model = model_initializer().to(DEVICE)
    else:
        model = model_initializer(**model_loader_params).to(DEVICE)

    # `fine_tunning.pretrain_boolean` (slice pipeline) and `finetunning.do`
    # (patch pipeline) both mean "start from these weights"; support both.
    initial_weight_path = None
    if experiment.get('fine_tunning', {}).get('pretrain_boolean', None) is True:
        pretrained = experiment['fine_tunning'].get('pretrained_model_weight', None)
        if pretrained:
            ckpt = torch.load(pretrained, weights_only=False, map_location=DEVICE)
            if "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"])
            elif "model" in ckpt:
                model.load_state_dict(ckpt["model"])
            else:
                model.load_state_dict(ckpt)
            initial_weight_path = pretrained
            print("Initialized model to path ", pretrained)

    if experiment.get("finetunning", {}).get('do', False):
        ckpt = torch.load(experiment['finetunning']['weight_path'], weights_only=False, map_location=DEVICE)
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        elif "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
        initial_weight_path = experiment['finetunning']['weight_path']
        print("Initialized model to path ", experiment['finetunning']['weight_path'])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=experiment['training']['learning_rate'],
        weight_decay=experiment['training']['weight_decay'],
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    steps_per_epoch = len(train_dataloader) if experiment['training']['epochs'] > 0 else 1
    num_warmup_steps = int(steps_per_epoch * experiment['training']['epochs']
                           * experiment['training'].get('warmup_percentage_steps', 0.1))
    scheduler = get_lr_scheduler(
        optimizer,
        num_warmup_steps,
        steps_per_epoch,
        experiment['training'].get('lr_schedule', {}).get('step_decay', {}),
    )

    # -- resume -------------------------------------------------------------
    if experiment.get("resume_training", {}).get('do', False):
        epoch, split_index = load_training_state(
            experiment['resume_training']['weight_path'],
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        )
        print("Initialized model to path ", experiment['resume_training']['weight_path'])
        if sdx < split_index:
            print("Skipping to match Split to weight. Skipping: ", sdx)
            del model
            continue
        csv_logger_train = CSVLogger(log_dir=f'logs/{exp_name}/metrics', filename=f'{sample_index}_metircs_train_.csv', resume=True)
        csv_logger_testing = CSVLogger(log_dir=f'logs/{exp_name}/metrics', filename=f'{sample_index}_metircs_testing_.csv', resume=True)
    else:
        epoch = 0
        csv_logger_train = CSVLogger(log_dir=f'logs/{exp_name}/metrics', filename=f'{sample_index}_metircs_train_.csv', resume=False)
        csv_logger_testing = CSVLogger(log_dir=f'logs/{exp_name}/metrics', filename=f'{sample_index}_metircs_testing_.csv', resume=False)

    start_epoch = epoch
    total_epochs = experiment['training']['epochs']

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    pbar = tqdm(range(start_epoch, total_epochs), desc=f"Epoch {start_epoch}")

    for epoch_idx in pbar:
        pbar.set_description(f"Epoch {epoch_idx}")
        loss_value = 0.0
        num_samples = 0
        model = model.float()
        model.train()

        augmenters = build_augmenters(epoch_idx)
        if augmenters is None and experiment.get('augmentations', {}).get('apply', False) \
                and experiment['augmentations'].get('type', 'static') != 'static' \
                and epoch_idx == experiment['augmentations']['till']:
            print("Stopping augmentations for next epochs")

        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training", leave=False)):
            optimizer.zero_grad(set_to_none=True)
            if gin_net is not None:
                gin_net.reinit()

            source, target, mask = to_device(batch['source']), to_device(batch['target']), to_device(batch['mask'])
            source, target, mask = apply_augmenters(augmenters, source, target, mask)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                if pre_post_norm:
                    source, _ = normalize_per_sample_masked(x=source, mask=mask, method=pre_post_norm_type)
                    # Fixed target optimiation
                    # target, _ = normalize_per_sample_masked(x=target, mask=mask, method=pre_post_norm_type)

                if block_augments:
                    source, aug_mask, weighted_loss_dict = augment_2d(x=source, config=block_config, gin=gin_net)
                    output = model_forward(model, source)
                    if not spatial_corruption:
                        loss = loss_fn(output.float() * mask, target * mask)
                    elif weighted_loss_dict['weighted_loss']:
                        w_full, w_corrupt = weighted_loss_dict['weighted_loss_weight']
                        loss = (w_full * loss_fn(output.float() * mask, target * mask)
                                + w_corrupt * loss_fn(output.float() * aug_mask, target * aug_mask))
                    else:
                        loss = loss_fn((output.float() * mask) * aug_mask, (target * mask) * aug_mask)
                else:
                    output = model_forward(model, source)
                    loss = loss_fn(output.float() * mask, target * mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss_value += loss.item() * source.shape[0]
            num_samples += source.shape[0]
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

        loss_value /= max(num_samples, 1)
        writer.add_scalar(f"loss/{sdx}_train_loss", loss_value, epoch_idx + 1)
        writer.add_scalar(f"lr/{sdx}", optimizer.param_groups[0]['lr'], epoch_idx + 1)

        # -- validation -----------------------------------------------------
        model.eval()
        val_error, num_samples = 0.0, 0

        if len(val_dataloader):
            val_tqdm_loop = tqdm(val_dataloader, leave=False)
            with torch.no_grad():
                for val_batch_idx, val_batch in enumerate(val_tqdm_loop):
                    source = to_device(val_batch['source'])
                    target = to_device(val_batch['target'])
                    mask = to_device(val_batch['mask'])

                    with torch.autocast(device_type="cuda"):
                        if pre_post_norm:
                            source, _ = normalize_per_sample_masked(x=source, mask=mask, method=pre_post_norm_type)
                        # validation measures the clean objective; no block augments here
                        val_output = model_forward(model, source)
                        batch_val_loss = loss_fn(val_output.float() * mask, target * mask)

                    val_error += batch_val_loss.item() * source.shape[0]
                    num_samples += source.shape[0]
                    val_tqdm_loop.set_description(f"Val Loss={val_error / max(num_samples, 1):.4f}")

            val_loss = val_error / max(num_samples, 1)
            writer.add_scalar(f"loss/{sdx}_val_loss", val_loss, epoch_idx + 1)

            metrics = {
                'epoch': epoch_idx + 1,
                'train_loss': loss_value,
                'val_loss': val_loss,
            }
            if wand_boolean:
                wandb.log({
                    "epoch": epoch_idx + 1,
                    "split": sdx,
                    f"train/loss_{sample_index}": loss_value,
                    f"val/loss_{sample_index}": val_loss,
                })
        else:
            metrics = {
                'epoch': epoch_idx + 1,
                'train_loss': loss_value,
            }
            if wand_boolean:
                wandb.log({
                    "split": sdx,
                    "epoch": epoch_idx + 1,
                    f"train/loss_{sample_index}": loss_value,
                })

        csv_logger_train.log(metrics)
        checkpoint_callback_2.save(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                                   step=epoch_idx, epoch=epoch_idx + 1, split=sdx, logs=metrics)
        if avg_weighting:
            avg_annealing_checkpoint_callback.save(model=model, optimizer=optimizer, scheduler=scheduler,
                                                   scaler=scaler, step=epoch_idx, epoch=epoch_idx + 1,
                                                   split=sdx, logs=metrics)

        # -- periodic full-volume sampling ----------------------------------
        if experiment.get("sample_volume_during_training", {}).get("do", False):
            every = max(int(experiment["sample_volume_during_training"]["every"]), 1)
            if (epoch_idx + 1) % every == 0:
                if intermediate_testing_loader is None:
                    intermediate_testing_loader = cache_dl.fetch_intermediateLoader()
                assert intermediate_testing_loader is not None, \
                    "Make sure a validation set exists (training.splits.validation_samples > 0) to sample during training"
                val_test_sample.test_sample(
                    model=model, sample_idx=sdx, epoch=epoch_idx + 1,
                    test_dataloader=intermediate_testing_loader,
                )

    if avg_weighting:
        average_model_weights(
            f'logs/{exp_name}/checkpoints_averaging/{sample_index}',
            f'logs/{exp_name}/checkpoints/{sample_index}',
            model_loader_params, model_initializer,
            f"{experiment['training']['epochs']}_{experiment['checkpoint']['save_every_n_epochs']}",
        )

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------
    # `evaluation.weights_dir` lets an evaluation-only run score another
    # experiment's checkpoints; "{split}" is substituted with the split index.
    weights_dir_override = experiment.get('evaluation', {}).get('weights_dir', None)
    if weights_dir_override:
        checkpoint_dir = weights_dir_override.replace("{split}", str(sample_index))
    else:
        checkpoint_dir = f"logs/{exp_name}/checkpoints/{sample_index}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    all_weights = [os.path.join(checkpoint_dir, p) for p in sorted(os.listdir(checkpoint_dir))
                   if os.path.isfile(os.path.join(checkpoint_dir, p))]

    if experiment.get('only_evaluate_weigths_with', None):
        all_weights = [ele for ele in all_weights if experiment['only_evaluate_weigths_with'] in ele]

    if not all_weights and initial_weight_path:
        # evaluate-only runs (epochs == 0) train nothing, so there is no
        # checkpoint directory to walk — score the weights we were handed.
        print(f"No checkpoints in {checkpoint_dir}; evaluating the initial weights instead.")
        all_weights = [initial_weight_path]

    if not all_weights:
        print(f"No checkpoints found in {checkpoint_dir}; nothing to evaluate for split {sdx}.")

    weight_tqdm_loop = tqdm(all_weights, desc="Evaluating Weights", leave=False)
    split_results = []

    for wdx, weight_path in enumerate(weight_tqdm_loop):
        epoch = -1
        for ele in os.path.basename(weight_path).split('/'):
            if 'epoch' in ele:
                try:
                    epoch = int(ele.split('=')[1].split('.')[0])
                except (IndexError, ValueError):
                    pass
        weight_prefix = f"{os.path.basename(weight_path).split('-')[0]}"
        weight_tqdm_loop.set_description(f"Evaluating Weight: {weight_prefix} Epoch: {epoch}")

        load_model_state(model=model, path=weight_path, device=DEVICE)
        model.eval()

        testing_loop_per_weight = tqdm(range(number_test_samples), desc=f"Testing {weight_prefix}", leave=False)
        for i in testing_loop_per_weight:
            sample_index_w = i
            test_entry = cache_dl.fetch_testLoader(idx=i)
            test_subject_name = test_entry['test_name']
            testing_loop_per_weight.set_description(f"Testing {weight_prefix} Scan: {test_subject_name}")

            result = volume_predictor.predict(model, test_entry['test'])

            iqa_metrics.compute(result['fake'], result['reference'], result['mask'])

            original_path = f"logs/{exp_name}/generations/Split{sdx}_{sample_index_w}_{test_subject_name}_original.nii.gz"
            if not os.path.exists(original_path):
                nib.save(
                    nib.Nifti1Image(result['reference'].cpu().numpy().astype(np.float32), result['affine']),
                    original_path,
                )
            nib.save(
                nib.Nifti1Image(result['fake'].cpu().numpy().astype(np.float32), result['affine']),
                f"logs/{exp_name}/generations/Split{sdx}_{sample_index_w}_{test_subject_name}_{weight_prefix}_fake.nii.gz",
            )

        metrics = iqa_metrics.value()
        iqa_metrics.reset()

        print(f"Test Metrics for split {sdx} weight {weight_prefix}: PSNR: {metrics['PSNR']}, "
              f"SSIM_3D: {metrics['SSIM_3D']},  SSIM_2D: {metrics['SSIM_2D']}, "
              f"MAE: {metrics['MAE']}, MSE: {metrics['MSE']}")

        sample_test_metics = {
            'sample_index': sample_index,
            'weight_prefix': weight_prefix,
            'test_psnr': metrics['PSNR'],
            'test_ssim_3D': metrics['SSIM_3D'],
            'test_ssim_2D': metrics['SSIM_2D'],
            'test_mae': metrics['MAE'],
            'test_mse': metrics['MSE'],
        }
        if wand_boolean:
            wandb.log(sample_test_metics)
        final_results.append(sample_test_metics)
        split_results.append(sample_test_metics)
        csv_logger_testing.log(sample_test_metics)

    if split_results:
        keys = ['test_psnr', 'test_ssim_3D', 'test_ssim_2D', 'test_mae', 'test_mse']
        csv_logger_average.log({
            'sample_index': sample_index,
            'num_weights': len(split_results),
            **{k: sum(r[k] for r in split_results) / len(split_results) for k in keys},
        })

    del model
    print("Training and evaluation completed for sample index:", sample_index)
    if experiment['training']['splits']['single_pass']:
        writer.close()
        exit()

writer.close()
print("All splits done.")
