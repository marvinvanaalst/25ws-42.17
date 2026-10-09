"""2D counterparts of `augmentations/block_augments.py`.

Same pretext-task corruptions (GIN intensity mixing, block-wise resolution
reduction, block masking) but operating on (B, C, H, W) slices instead of
(B, C, D, H, W) volumes. Kept in a separate module so the 3D patch pipeline is
untouched.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GIN2D(nn.Module):
    """
    Global Intensity Non-linear augmentation (2D).
    Shallow randomly-weighted 2D CNN, re-initialized on every call
    (call once per epoch or once per batch -- your choice).
    """

    def __init__(self, in_channels=1, hidden_channels=8,
                 n_layers=4, kernel_size=3):
        super().__init__()
        layers = []
        ch = in_channels
        for _ in range(n_layers - 1):
            layers += [
                nn.Conv2d(ch, hidden_channels, kernel_size,
                          padding=kernel_size // 2, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = hidden_channels
        # final 1x1 projection back to input channels
        layers.append(nn.Conv2d(ch, in_channels, 1, bias=False))
        self.net = nn.Sequential(*layers)

        for p in self.parameters():
            p.requires_grad_(False)  # augmentation only, no training

    def reinit(self):
        """Randomize weights -- call every epoch/batch."""
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2)

    @torch.no_grad()
    def forward(self, x):
        """x: (B, C, H, W). Mixing/energy-preservation is done in `augment_2d`."""
        return self.net(x)


def _as_block_2d(block):
    """Accept 8, (8, 8) or a 3D block spec [8, 8, 8] and return (bh, bw)."""
    if isinstance(block, int):
        return (block, block)
    block = list(block)
    if len(block) == 1:
        return (block[0], block[0])
    if len(block) == 2:
        return (block[0], block[1])
    # tolerate a 3D block config being reused for the slice pipeline
    return (block[-2], block[-1])


def _grid_shape(H, W, block):
    """Coarse grid size, never zero (a block larger than the image => 1 cell)."""
    bh, bw = block
    return max(H // bh, 1), max(W // bw, 1)


@torch.no_grad()
def _blockwise_downsample_2d(x, factors=(2, 4), ds_ratio=0.5, block=(8, 8),
                             up_mode="bilinear"):
    """Degrade random coarse blocks by down-then-up sampling.

    Each coarse block is, with probability `ds_ratio`, replaced by a version of
    the slice that was anti-aliased-downsampled by a factor drawn uniformly
    from `factors` and upsampled back. Cheap: one global down/up per factor,
    then a blockwise select -- no per-block loops.

    Returns:
        out:      (B, C, H, W) degraded slice
        ds_mask:  (B, 1, H, W) float, 1 where resolution was reduced
    """
    B, C, H, W = x.shape
    block = _as_block_2d(block)
    hg, wg = _grid_shape(H, W, block)

    # per-block choice: 0 = untouched, k = factors[k-1]
    choice = torch.randint(1, len(factors) + 1, (B, 1, hg, wg), device=x.device)
    choice[torch.rand(B, 1, hg, wg, device=x.device) >= ds_ratio] = 0
    choice = F.interpolate(choice.float(), size=(H, W), mode="nearest").long()

    out = x.clone()
    for k, f in enumerate(factors, start=1):
        # anti-aliased downsample, then upsample back to full resolution
        lo = F.avg_pool2d(x, kernel_size=f, stride=f)
        lo = F.interpolate(
            lo, size=(H, W), mode=up_mode,
            align_corners=False if up_mode in ("bilinear", "bicubic") else None,
        )
        out = torch.where(choice == k, lo, out)

    return out, (choice > 0).to(x.dtype)


@torch.no_grad()
def _blockwise_downsample_2d_aniso(x, factors=(2, 4), ds_ratio=0.5, block=(8, 8)):
    """Optional variant: downsample only along H (simulates anisotropic sampling)."""
    B, C, H, W = x.shape
    block = _as_block_2d(block)
    hg, wg = _grid_shape(H, W, block)

    choice = torch.randint(1, len(factors) + 1, (B, 1, hg, wg), device=x.device)
    choice[torch.rand(B, 1, hg, wg, device=x.device) >= ds_ratio] = 0
    choice = F.interpolate(choice.float(), size=(H, W), mode="nearest").long()

    out = x.clone()
    for k, f in enumerate(factors, start=1):
        lo = F.avg_pool2d(x, kernel_size=(f, 1), stride=(f, 1))
        lo = F.interpolate(lo, size=(H, W), mode="bilinear", align_corners=False)
        out = torch.where(choice == k, lo, out)

    return out, (choice > 0).to(x.dtype)


@torch.no_grad()
def augment_2d(x, config, gin=None):
    """Corrupt a slice batch for a reconstruction pretext task.

    Args:
        x: (B, C, H, W) clean slices.
        config: dict with keys
            block               -- coarse block size, e.g. [8, 8]
            gin_apply           -- bool, apply GIN style mixing
            gin_lam_max         -- lambda upper bound, keeps input != target
            gin_conserve_enegry -- bool, rescale to preserve per-sample L2 energy
            downsample_apply    -- bool, apply block-wise resolution reduction
            downsample_factors  -- candidate factors, e.g. [2, 4]
            downsample_ratio    -- P(a block is downsampled)
            downsample_aniso    -- bool, degrade along H only (optional)
            mask_apply          -- bool, zero out random blocks
            mask_ratio          -- P(a block is masked)
            weighted_loss       -- bool, ask the caller for a two-term loss
            weighted_loss_weight-- [w_full, w_corrupted]
        gin: GIN2D module, required when config['gin_apply'] is True.

    Returns:
        mixed:        (B, C, H, W) corrupted input for the model
        corrupt_mask: (B, 1, H, W) 1 where the input was corrupted
        loss_spec:    dict describing the weighted-loss request

    Compute the reconstruction loss on corrupt_mask (or upweight those pixels).
    Untouched regions admit a trivial identity solution -- especially under
    "downsample", where the input already matches the target at low frequency.
    """
    block = _as_block_2d(config['block'])
    gin_apply = config['gin_apply']
    gin_lam_max = config['gin_lam_max']
    gin_conserve_energy = config.get('gin_conserve_enegry', True)
    downsample_apply = config['downsample_apply']
    downsample_factors = config['downsample_factors']
    downsample_ratio = config['downsample_ratio']
    downsample_aniso = config.get('downsample_aniso', False)
    mask_apply = config['mask_apply']
    mask_ratio = config['mask_ratio']
    weighted_loss = config.get('weighted_loss', False)
    weighted_loss_weight = config.get('weighted_loss_weight', [1.0, 1.0])

    if gin_apply and gin is None:
        raise ValueError("config['gin_apply'] is True but no gin module was passed")

    B, _, H, W = x.shape

    if gin_apply:
        mixed = gin(x)
        # lambda bounded to [0, lam_max] so input != target
        lam = gin_lam_max * torch.rand(B, 1, 1, 1,
                                       device=x.device, dtype=x.dtype)
        mixed = lam * x + (1.0 - lam) * mixed
        # energy preservation (per slice, per batch sample)
        if gin_conserve_energy:
            dims = (1, 2, 3)
            e_in = x.norm(p=2, dim=dims, keepdim=True)
            e_mix = mixed.norm(p=2, dim=dims, keepdim=True).clamp_min(1e-8)
            mixed = mixed * (e_in / e_mix)
    else:
        # clone so later in-place ops can never touch the target
        mixed = x.clone()

    corrupt_mask = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)

    # ORDER MATTERS: downsample first, then mask. A blurred block may then be
    # zeroed (clamp_ collapses the overlap to 1). Blurring *after* masking would
    # smear zeros across block boundaries and leak the mask pattern outward.
    if downsample_apply:
        ds_fn = _blockwise_downsample_2d_aniso if downsample_aniso else _blockwise_downsample_2d
        mixed, ds_mask = ds_fn(mixed, downsample_factors, downsample_ratio, block)
        corrupt_mask = corrupt_mask + ds_mask

    if mask_apply:
        hg, wg = _grid_shape(H, W, block)
        # m == 1 keeps the block, m == 0 masks it; P(mask) = mask_ratio
        m = (torch.rand(B, 1, hg, wg, device=x.device, dtype=x.dtype)
             > mask_ratio).to(x.dtype)
        m = F.interpolate(m, size=(H, W), mode="nearest")
        mixed = mixed * m
        corrupt_mask = corrupt_mask + (1.0 - m)

    corrupt_mask = corrupt_mask.clamp_(0, 1)
    return mixed, corrupt_mask, {'weighted_loss': weighted_loss,
                                 'weighted_loss_weight': weighted_loss_weight}
