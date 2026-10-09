"""Slice (2D) training master — `master_half_slice_train.py` with the
`master_half_patch.py` machinery folded in.

Ported over from the patch master:
  * weight saving — full training state (model/optimizer/scheduler/scaler/RNG/
    split) through `LightningStyleCheckpoint.save(...)`, restored with
    `load_training_state` / `load_model_state`, so `resume_training` works.
  * affine matrix taken from the *target* NIfTI (not the float16 affine cached
    alongside the source slices), for both the saved original and the fake.
  * augmentations — GIN / block-downsample / block-mask pretext corruptions
    (2D variants) and flip TTA at inference. (The patch master's masked
    pre/post normalisation is deliberately NOT carried over: this pipeline is
    [0, 1] end to end via slicify, so a second per-sample normalisation only
    added a train/inference asymmetry — it normalised the source but not the
    target. `pretrain_augments.pre_post_norm` is ignored here.)
    The kornia spatial/intensity augmentation of the slice master is kept.
  * validation-volume sampling during training — `sample_volume_during_training`
    reconstructs a full validation volume every N epochs into
    `logs/<exp>/training_eval`.

Kept from the slice master: the training-time restoration of target dimensions,
i.e. every predicted slice goes back through `restore_original_batch`
(undo resize -> undo pad) and is re-stacked on the original slice axis before
any metric or NIfTI is written.

Usage:
    python master_half_slice_re.py configs/New_slice/<config>.json
"""

from utils_custom.utils_opus import model_fetcher

import torch
import json
import sys
import csv
import torchio as tio
import warnings
warnings.simplefilter("default")  # or "always"
import shutil
from torch.cuda.amp import autocast, GradScaler
import os
from tqdm import tqdm
wand_boolean = False
print("WAND DB BOOLEAN", wand_boolean)
from utils_custom.csv_logger import CSVLogger
from utils_custom.LightningStyleCheckpoint import LightningStyleCheckpoint
from utils_custom.load_weights import load_model_state, load_training_state
import nibabel as nib
from utils_custom.tensor_operations import restore_original_batch, load_and_normalize_slices
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

argv = sys.argv

import random
import numpy as np

from augmentations.tta import tta_forward_2d


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

try:
    import kornia.augmentation as K
except ImportError:
    K = None
    print("[WARNING] kornia not found - Kornia augmentations will be disabled")

with open(experimnet_config, 'r') as f:
    experiment = json.load(f)

if experiment.get("seed", {}).get("apply", False):
    set_seed(experiment["seed"].get("value", 42))
    print(f"Deterministic mode ON, seed={experiment['seed'].get('value', 42)}")

single_channel_model = experiment['model_info'].get('single_channel_model', False)

# The pipeline works in [0, 1] end to end: slicify min-max normalises source and
# target, the loss is taken against the [0, 1] target and VolumeIQAMetrics scores
# with data_range=1.0. `model_info.output_range_01` says whether the *model* also
# speaks [0, 1] (true, e.g. a Hardtanh/Sigmoid tail) or [-1, 1] (false, e.g. a
# tanh tail). When it is false the conversion is confined to `model_forward`
# below — the input is scaled on the way in and the prediction scaled back on the
# way out — so augmentation, the loss, the metrics and the NIfTIs never leave
# [0, 1] and the two settings stay directly comparable.
OUTPUT_RANGE_01 = experiment['model_info'].get('output_range_01', True)
print(f"Model output range: {'[0, 1]' if OUTPUT_RANGE_01 else '[-1, 1]'} "
      f"(model_info.output_range_01={OUTPUT_RANGE_01})")

from utils_custom.eval_metrics import VolumeIQAMetrics

if experiment['model_info']['modelType'] == 'vol2vol':
    pass
elif experiment['model_info']['modelType'] == 'patch2patch' or experiment['model_info']['modelType'] == "pre_computed_patch2patch":
    from utils_custom.utils_opus import model_fetcher
    model_initializer, model_loader_params = model_fetcher(experiment).fetch_model()
elif experiment['model_info']['modelType'] in ('pre_computed_slice2slice', 'pre_computed_slice2slice_25d'):
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
# Canonicalised here so an unknown/typo'd `cache_stratergy` is reported once,
# up front, rather than silently doing something else deep in the data module.
from data_modules.opus_CachingSliceDataset import resolve_cache_mode
cache_mode = resolve_cache_mode(experiment)
print(f"Cache strategy: {cache_mode} "
      f"('same_gpu' = slices cached in VRAM, 'ram' = cached in host memory, "
      f"'none' = loaded from disk on demand)")

# ---------------------------------------------------------------------------
# Augmentation / normalisation switches (same config keys as master_half_patch)
# ---------------------------------------------------------------------------
block_augments = experiment.get("pretrain_augments", {}).get("block_augments", {}).get("apply", False)
block_config = experiment.get("pretrain_augments", {}).get("block_augments", {}).get("config", {})

tta_flip_enabled = experiment.get("tta_flips", {}).get("enable", False)

# 2D counterparts of the patch master's GIN3D / augment.
from augmentations.block_augments_2d import GIN2D, augment_2d
from components.testing_logic_slice import test_on_sample, slicify_normalize

val_test_sample = test_on_sample(experiment)

if block_augments and block_config.get('gin_apply', False):
    print("\n\n\n GIN+MASK Augments enabled.")
    gin_net = GIN2D(in_channels=experiment['model'].get('in_channels', 1)).to(DEVICE)
else:
    gin_net = None

# GIN on its own leaves `corrupt_mask` all zeros. Gating the loss on that map
# would make the loss identically 0, so only use it when a *spatial* corruption
# (downsample / mask) actually ran.
spatial_corruption = bool(block_config.get('downsample_apply', False) or
                          block_config.get('mask_apply', False))
if block_augments and not spatial_corruption:
    print("[Augments] GIN-only corruption: loss stays on the full brain mask.")

# slicify only pads when pad_to > resize_to, and restore_original_batch only
# knows how to undo that ordering. Fail here rather than at reconstruction.
_pad_to = experiment['slicify']['pad_to']
_resize_to = experiment['slicify']['resize_to']
_pad_scalar = _pad_to[0] if isinstance(_pad_to, (list, tuple)) else _pad_to
if _pad_scalar < _resize_to:
    raise ValueError(
        f"slicify.pad_to ({_pad_scalar}) < slicify.resize_to ({_resize_to}). "
        f"Padding is skipped at slicify time but the restore path cannot undo "
        f"that combination — use pad_to >= resize_to."
    )

SLICE_AXIS = experiment['data'].get('slice_axis', 2)
REFERENCE_NORM = experiment.get("evaluation", {}).get("ground_truth_norm", "slicify")
SLICIFY_PCTL_CLIP = experiment.get("evaluation", {}).get("slicify_percentile_clip", None)

if os.path.exists(f"logs/{exp_name}/config.json"):
    print("Skipping config file copying,  source and destination are the same file.")
else:
    shutil.copy(experimnet_config, f"logs/{exp_name}/config.json")

iqa_metrics = VolumeIQAMetrics(data_range=1.0, device=DEVICE)

if experiment['resume_from_sample']['do']:
    csv_logger_average = CSVLogger(log_dir=f'logs/{exp_name}/metrics', filename=f'average_testing_.csv', resume=True)
else:
    csv_logger_average = CSVLogger(log_dir=f'logs/{exp_name}/metrics', filename=f'average_testing_.csv', resume=False)

if experiment['fine_tunning'].get('pretrain_evaluate_only', None) == True and experiment['fine_tunning'].get('pretrain_boolean', None) == True:
    print("Pretrained-evaluation mode: forcing epochs to 0.")
    experiment['training']['epochs'] = 0

print("Total Number of samples", len(experiment['data']['training']['subjects']))
print("Total Splits", (len(experiment['data']['training']['subjects']) // experiment['training']['splits']['num_test_samples']))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_model_range(x, OUTPUT_RANGE_01=OUTPUT_RANGE_01):
    """[0, 1] -> the range the model expects."""
    return x if OUTPUT_RANGE_01 else x * 2.0 - 1.0


def from_model_range(x, OUTPUT_RANGE_01=OUTPUT_RANGE_01):
    """The model's output range -> [0, 1]."""
    return x if OUTPUT_RANGE_01 else (x + 1.0) / 2.0


def model_forward(model, source):
    """One forward pass, honouring `model_info.single_channel_model`.

    This is the only place that knows about `model_info.output_range_01`.
    Everything on either side of it — augmentation, block corruptions, the
    loss, evaluation, the saved volumes — stays in [0, 1], so switching the
    flag does not silently rescale the loss or the metrics.
    """
    source = to_model_range(source)
    if single_channel_model:
        out = model(source.squeeze(1)).unsqueeze(1)
    else:
        out = model(source)
    return from_model_range(out)


def inference_forward(model, source, mask=None):
    """Inference-time forward: optional flip TTA.

    `mask` is accepted for call-site compatibility but is not used — masking
    happens once on the reconstructed volume, not per slice.
    """
    if tta_flip_enabled:
        out = tta_forward_2d(model, source, forward=lambda t: model_forward(model, t))
    else:
        out = model_forward(model, source)
    return out.float()


def to_device(tensor):
    """Move a batch tensor to DEVICE, whatever the cache strategy was.

    Under `same_gpu` the tensor is already there and this is a no-op; under
    `ram` / `none` it is the (pinned, hence async) host->device copy.
    """
    return tensor.to(DEVICE, non_blocking=True).float()


def target_path_from_source(path_scan):
    """Swap the *modality directory* of a cached source path for the target one.

    `str.replace` on the whole path would also hit any other occurrence of the
    modality name (e.g. a patient id containing 'ct'), so only the path
    component that exactly matches the source directory is rewritten.
    """
    src_name = experiment['data']['direction']['source']
    tgt_name = experiment['data']['direction']['target']
    parts = path_scan.split(os.sep)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == src_name:
            parts[i] = tgt_name
            return os.sep.join(parts)
    # nothing matched a whole component — fall back to the old behaviour
    return path_scan.replace(src_name, tgt_name)


def build_augmenters(epoch_idx):
    """Kornia augmenters for this epoch, or None when augmentation is off.

    Three separate sequentials on purpose:
      * geometric — ONE sampled transform shared by source/target/mask, so the
        pair stays registered. (Calling a single sequential three times draws
        three *different* transforms and silently destroys the pairing.)
      * erasing   — corrupts the model input only.
      * intensity — photometric jitter on whichever side `intensity_apply_to`
        names.
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
        *([K.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1),
                          shear=10.0, p=prob)] if aug_cfg.get('affine', False) else []),
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
        *([K.RandomGaussianNoise(mean=0.0, std=aug_cfg.get('noise_std', 0.05), p=prob)]
          if aug_cfg.get('gaussian_noise', False) else []),
        *([K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=prob)]
          if aug_cfg.get('gaussian_blur', False) else []),
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
    """Linear warmup (per optimizer step), optionally times a StepLR-style decay.

    Folded into one LambdaLR so a single scheduler object round-trips through
    the checkpoint / `load_training_state` pair.
    """
    apply_decay = bool(decay_cfg.get('apply', False))
    step_size = max(int(decay_cfg.get('step_size_epochs', 10)), 1)
    gamma = float(decay_cfg.get('gamma', 0.5))
    steps_per_epoch = max(int(steps_per_epoch), 1)

    def lr_lambda(current_step):
        # num_warmup_steps == 0 means "no warmup", so the factor has to be 1.0
        # from step 0. The old `current_step / max(1, num_warmup_steps)` returned
        # 0.0 at step 0 in that case — and because LambdaLR's constructor calls
        # step() once, that silently set lr=0 before training even started and
        # threw away the first optimizer step of every split.
        if num_warmup_steps > 0:
            factor = min(1.0, float(current_step) / float(num_warmup_steps))
        else:
            factor = 1.0
        if apply_decay:
            epoch = current_step // steps_per_epoch
            factor *= gamma ** (epoch // step_size)
        return factor

    # ---- Polynomial decay (for SGD, matching TransUNet official repo) ----
    sched_type = experiment['training'].get('lr_schedule', {}).get('type', 'linear_warmup')
    if sched_type == 'poly_decay':
        total_steps = max(1, int(experiment['training']['epochs']) * max(1, int(steps_per_epoch)))
        power = float(experiment['training'].get('lr_schedule', {}).get('poly_power', 0.9))
        print(f"[Scheduler] poly_decay power={power} total_steps={total_steps}")
        def _poly_lr(step):
            return max(0.0, (1.0 - step / total_steps) ** power)
        return LambdaLR(optimizer, _poly_lr)

        return LambdaLR(optimizer, lr_lambda)


def build_optimizer(model):  # TRANSUNET_PATCH_APPLIED
    name = str(experiment['training'].get('optimizer', 'adamw')).lower()
    lr  = experiment['training']['learning_rate']
    wd  = experiment['training']['weight_decay']
    if name == 'sgd':
        momentum = float(experiment['training'].get('momentum', 0.9))
        print(f"[Optimizer] SGD lr={lr} momentum={momentum} wd={wd} nesterov=True")
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd,
                               momentum=momentum, nesterov=True)
    if name == 'adam':
        print(f"[Optimizer] Adam lr={lr} wd={wd}")
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    # default: AdamW
    print(f"[Optimizer] AdamW lr={lr} wd={wd}")
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def load_weights_into(model, path):
    """Accept both the new full-state checkpoints and the legacy bare state dicts."""
    ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)



full_percision = experiment.get("percision",{}).get("full",False)

# Gate for the LR scheduler; see the construction site inside the split loop.
use_lr_schedule = bool(experiment['training'].get('lr_schedule', {}).get('apply', True))
print(f"LR schedule: {'on' if use_lr_schedule else 'off (constant lr)'} "
      f"(training.lr_schedule.apply={use_lr_schedule})")


# ---------------------------------------------------------------------------
# Split loop
# ---------------------------------------------------------------------------
for sdx in range(len(experiment['data']['training']['subjects']) // experiment['training']['splits']['num_test_samples']):
    sample_index = sdx
    if experiment["resume_from_sample"]["do"] and sample_index < experiment["resume_from_sample"]["index"]:
        print(f"Skipping sample {sample_index} as per resume configuration.")
        continue

    # -- data ---------------------------------------------------------------
    if experiment['model_info']['modelType'] == 'pre_computed_slice2slice':
        from data_modules.opus_CachingSliceDataset import cache_slice_dataloader
        from patchify.slicify.slice_checker import create_slicify_dataset
        create_slicify_dataset(experiment)
        cache_dl = cache_slice_dataloader(experiment, DEVICE, idx=sample_index)
        temp = cache_dl.fetch_TrainValLoaders(build_train=experiment['training']['epochs'] > 0)
        train_dataloader, val_dataloader, number_test_samples = temp['train'], temp['validation'], temp['number_test_samples']
    elif experiment['model_info']['modelType'] == 'pre_computed_slice2slice_25d':
        from data_modules.opus_CachingSliceDataset_2_5D import cache_slice_25d_dataloader
        from patchify.slicify.slice_checker import create_slicify_dataset
        create_slicify_dataset(experiment)
        cache_dl = cache_slice_25d_dataloader(experiment, DEVICE, idx=sample_index)
        temp = cache_dl.fetch_TrainValLoaders(build_train=experiment['training']['epochs'] > 0)
        train_dataloader, val_dataloader, number_test_samples = temp['train'], temp['validation'], temp['number_test_samples']
    else:
        raise ValueError(
            f"Unsupported model type: {experiment['model_info']['modelType']}. "
            f"master_half_slice_re.py only drives 'pre_computed_slice2slice' or 'pre_computed_slice2slice_25d'."
        )

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
    checkpoint_callback_3 = LightningStyleCheckpoint(
        save_dir=f'logs/{exp_name}/checkpoints/{sample_index}',
        monitor='train_loss',
        mode='min',
        top_k=experiment['checkpoint']['top_k'],
        save_every_n_epochs=10000000,
        save_weights_only=True,
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
    if experiment.get('fine_tunning', {}).get('pretrain_boolean', None) == True:
        pretrained = experiment['fine_tunning'].get('pretrained_model_weight', None)
        if pretrained:
            load_weights_into(model, pretrained)
            initial_weight_path = pretrained
            print("Initialized model to path ", pretrained)

    if experiment.get("finetunning", {}).get('do', False):
        load_weights_into(model, experiment['finetunning']['weight_path'])
        initial_weight_path = experiment['finetunning']['weight_path']
        print("Initialized model to path ", experiment['finetunning']['weight_path'])

    optimizer = build_optimizer(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    steps_per_epoch = len(train_dataloader) if experiment['training']['epochs'] > 0 else 1
    num_warmup_steps = int(steps_per_epoch * experiment['training']['epochs']
                           * experiment['training'].get('warmup_percentage_steps', 0.1))
    
    # `training.lr_schedule.apply` gates the scheduler. It has to gate
    # CONSTRUCTION, not just the step() calls: LambdaLR's constructor performs an
    # initial step(), so a built-but-never-stepped scheduler pins the lr to
    # base_lr * lr_lambda(0) for the whole run. None == plain constant lr.
    if use_lr_schedule:
        scheduler = get_lr_scheduler(
            optimizer,
            num_warmup_steps,
            steps_per_epoch,
            experiment['training'].get('lr_schedule', {}).get('step_decay', {}),
        )
    else:
        scheduler = None
        print(f"[LR] lr_schedule.apply=false — no scheduler, constant lr="
              f"{experiment['training']['learning_rate']}")

    # -- resume -------------------------------------------------------------
    if experiment.get("resume_training", {}).get('do', False):
        # map_location stays 'cpu': the checkpoint carries CPU RNG ByteTensors
        # that torch.random.set_rng_state refuses to accept off-device.
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
            if experiment.get('augmentations', {}).get('apply', False):
                source, target, mask = apply_augmenters(augmenters, source, target, mask)
            if not full_percision:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
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
                if scheduler is not None:
                    scheduler.step()
            else:
                if block_augments:
                    source, aug_mask, weighted_loss_dict = augment_2d(x=source, config=block_config, gin=gin_net)
                    output = model_forward(model, source)
                    if not spatial_corruption:
                        loss = loss_fn(output * mask, target * mask)
                    elif weighted_loss_dict['weighted_loss']:
                        w_full, w_corrupt = weighted_loss_dict['weighted_loss_weight']
                        loss = (w_full * loss_fn(output * mask, target * mask)
                                + w_corrupt * loss_fn(output * aug_mask, target * aug_mask))
                    else:
                        loss = loss_fn((output * mask) * aug_mask, (target * mask) * aug_mask)
                else:
                    output = model_forward(model, source)
                    loss = loss_fn(output * mask, target * mask)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                loss_value += loss.item() * source.shape[0]
                num_samples += source.shape[0]

                optimizer.step()
                if scheduler is not None:
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
        checkpoint_callback_3.save(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
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

    pad_k = experiment['slicify']['pad_to']
    resize_k = experiment['slicify']['resize_to']
    split_results = []

    for wdx, weight_path in enumerate(weight_tqdm_loop):
        epoch = -1
        for ele in os.path.basename(weight_path).split('_'):
            if 'epoch' in ele:
                try:
                    epoch = int(ele.split('=')[1].split('.')[0])
                except (IndexError, ValueError):
                    pass
        # strip the extension so generated filenames stay readable
        weight_prefix = f"{os.path.splitext(os.path.basename(weight_path))[0].split('-')[0]}"
        weight_tqdm_loop.set_description(f"Evaluating Weight: {weight_prefix} Epoch: {epoch}")

        load_model_state(model=model, path=weight_path, device=DEVICE)

        # Evaluate the model on the test set
        model.eval()
        testing_loop_pwe_weight = tqdm(range(number_test_samples), desc=f"Testing {weight_prefix}", leave=False)
        for idxx, i in enumerate(testing_loop_pwe_weight):
            sample_index_w = i
            test_entry = cache_dl.fetch_testLoader(idx=i)
            test_subject_name = test_entry['test_name']

            testing_loop_pwe_weight.set_description(f"Testing {weight_prefix} Scan: {test_subject_name}")

            test_dataloader = test_entry['test']

            for train_batch_idx, batch in enumerate(test_dataloader):
                original_path_scan = batch['path_scan'][0]
                original_path_mask = batch['path_mask'][0]
                break

            # The ground truth lives in the target modality directory; its NIfTI
            # affine (float64) is the one we stamp on every volume we write —
            # the cached `target_affine` is a float16 copy.
            original_path_target = target_path_from_source(original_path_scan)
            target_img = nib.load(original_path_target)
            affine = target_img.affine

            mask_orig = torch.from_numpy(
                nib.load(original_path_mask).get_fdata(dtype=np.float32)
            ).float().to(DEVICE)

            if REFERENCE_NORM == "slicify":
                # rebuild the reference exactly the way slicify built the cached
                # target slices the model was trained against
                target_volume = torch.from_numpy(
                    target_img.get_fdata(dtype=np.float32)
                ).float().to(DEVICE)
                scan_orig_div = slicify_normalize(target_volume, mask_orig,
                                                  percentile_clip=SLICIFY_PCTL_CLIP)
            elif REFERENCE_NORM == "percentile":
                scan_orig = load_and_normalize_slices(experiment, original_path_target,
                                                      dtype=torch.float32).to(DEVICE)
                scan_orig_div = scan_orig * mask_orig
            else:
                raise ValueError(
                    f"Unknown evaluation.ground_truth_norm={REFERENCE_NORM!r}; "
                    f"expected 'slicify' or 'percentile'."
                )

            original_out_path = f"logs/{exp_name}/generations/Split{sdx}_{sample_index_w}_{test_subject_name}_original.nii.gz"
            if not os.path.exists(original_out_path):
                nib.save(
                    nib.Nifti1Image(scan_orig_div.cpu().numpy().astype(np.float32), affine),
                    original_out_path,
                )

            with torch.no_grad():
                train_target_fake = []
                for train_batch_idx, batch in enumerate(test_dataloader):
                    source = batch['source'].to(DEVICE).float()
                    slice_mask = batch['mask'].to(DEVICE).float()
                    with torch.autocast(device_type="cuda"):
                        fake_output = inference_forward(model, source, slice_mask)

                    # training-time restoration of target dimensions:
                    # undo the slicify resize, then the slicify padding
                    fake_output = restore_original_batch(fake_output.float(), batch['padding'],
                                                         batch['original_size'], pad_k, resize_k)
                    train_target_fake.append(fake_output.squeeze(1))  # [B, H_orig, W_orig]

                # put the slices back on the axis they were cut from
                train_target_fake = torch.cat(train_target_fake, dim=0).movedim(0, SLICE_AXIS).contiguous()

                if train_target_fake.shape != mask_orig.shape:
                    raise ValueError(
                        f"Reconstructed volume {tuple(train_target_fake.shape)} does not match "
                        f"mask {tuple(mask_orig.shape)}. Check data.slice_axis (currently "
                        f"{SLICE_AXIS}) against how the slices were cached."
                    )

                # The pipeline trains and scores in [0, 1] — slicify min-max
                # normalises both source and target, the loss is computed
                # directly against the [0, 1] target, and the reference volume
                # below is rebuilt the same way. The old `(x+1)/2` rescale here
                # assumed a tanh-ranged model and would squash a [0, 1]
                # prediction into [0.5, 1]. Mask, then clamp — which is also
                # exactly what components/testing_logic_slice.py does, so the
                # in-training sampler and this final evaluation now agree.
                train_target_fake = train_target_fake * mask_orig
                train_target_fake = torch.clip(train_target_fake, min=0.0, max=1.0)

                iqa_metrics.compute(train_target_fake, scan_orig_div, mask_orig)

                output_img = nib.Nifti1Image(
                    train_target_fake.cpu().numpy().astype(np.float32), affine
                )
                nib.save(output_img,
                         f"logs/{exp_name}/generations/Split{sdx}_{sample_index_w}_{test_subject_name}_{weight_prefix}_fake.nii.gz")

        if not number_test_samples:
            continue

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
            'test_mse': metrics['MSE']
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
