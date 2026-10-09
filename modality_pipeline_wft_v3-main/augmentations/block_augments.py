import torch
import torch.nn as nn
import torch.nn.functional as F


class GIN3D(nn.Module):
    """
    Global Intensity Non-linear augmentation (3D).
    Shallow randomly-weighted 3D CNN, re-initialized on every call
    (call once per epoch or once per batch — your choice).
    """
    def __init__(self, in_channels=1, hidden_channels=8,
                 n_layers=4, kernel_size=3):
        super().__init__()
        layers = []
        ch = in_channels
        for i in range(n_layers - 1):
            layers += [
                nn.Conv3d(ch, hidden_channels, kernel_size,
                          padding=kernel_size // 2, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = hidden_channels
        # final 1x1x1 projection back to input channels
        layers.append(nn.Conv3d(ch, in_channels, 1, bias=False))
        self.net = nn.Sequential(*layers)

        for p in self.parameters():
            p.requires_grad_(False)  # augmentation only, no training

    def reinit(self):
        """Randomize weights — call every epoch."""
        for m in self.net.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, a=0.2)

    @torch.no_grad()
    def forward(self, x):
        """
        x: (B, C, D, H, W) — each element of the batch is a patch.
        A separate lambda is sampled per patch.
        """
        # y = self.net(x)
        return self.net(x)
        # per-patch lambda in [0, 1], broadcast over C,D,H,W
        lam = torch.rand(x.shape[0], 1, 1, 1, 1,
                         device=x.device, dtype=x.dtype)
        mixed = lam * x + (1.0 - lam) * y

        # energy preservation: rescale so ||x_aug|| == ||x|| per patch
        dims = (1, 2, 3, 4)
        e_in  = x.norm(p=2, dim=dims, keepdim=True)
        e_mix = mixed.norm(p=2, dim=dims, keepdim=True).clamp_min(1e-8)
        return mixed * (e_in / e_mix)
    
# @torch.no_grad()
# def gin_augment(gin, x, gin_mask, gin_downsample, ds_factors, ds_ratio,  lam_max=0.7, mask_ratio=0.7, block=(8, 8, 8), ):
#     """
#     x: (B, C, D, H, W). Returns (augmented_input, target, mask).
#     - lambda bounded to [0, lam_max] so input != target
#     - random 3D block masking to forbid identity/copy
#     """
#     y = gin(x)

#     lam = lam_max * torch.rand(x.shape[0], 1, 1, 1, 1,
#                                device=x.device, dtype=x.dtype)
#     mixed = lam * x + (1.0 - lam) * y
#     if gin_apply:
#     # energy preservation (per-patch)
#         dims = (1, 2, 3, 4)
#         e_in  = x.norm(p=2, dim=dims, keepdim=True)
#         e_mix = mixed.norm(p=2, dim=dims, keepdim=True).clamp_min(1e-8)
#         mixed = mixed * (e_in / e_mix)

    
#     if downsample_apply:
#         mixed, ds_mask = _blockwise_downsample(mixed, ds_factors, ds_ratio, block)
#         corrupt = corrupt + ds_mask
        
#     # coarse block mask (downsampled Bernoulli, then upsampled)
#     B, C, D, H, W = x.shape
#     dg, hg, wg = D // block[0], H // block[1], W // block[2]
#     m = (torch.rand(B, 1, dg, hg, wg, device=x.device) > mask_ratio).float()
#     m = F.interpolate(m, size=(D, H, W), mode="nearest")

#     return mixed * m  # feed mixed*m to model, reconstruct x on masked region



@torch.no_grad()
def _blockwise_downsample(x, factors=(2, 4), ds_ratio=0.5, block=(8, 8, 8),
                          up_mode="trilinear"): # manually checked
    """Degrade random coarse blocks by down-then-up sampling.
 
    Each coarse block is, with probability `ds_ratio`, replaced by a version of
    the volume that was anti-aliased-downsampled by a factor drawn uniformly
    from `factors` and upsampled back. Cheap: one global down/up per factor,
    then a blockwise select — no per-block loops.
 
    Returns:
        out:      (B, C, D, H, W) degraded volume
        ds_mask:  (B, 1, D, H, W) float, 1 where resolution was reduced
    """
    B, C, D, H, W = x.shape
    dg, hg, wg = D // block[0], H // block[1], W // block[2]
 
    # per-block choice: 0 = untouched, k = factors[k-1]
    choice = torch.randint(1, len(factors) + 1, (B, 1, dg, hg, wg), device=x.device) # assign downsample ratio to blocks
    choice[torch.rand(B, 1, dg, hg, wg, device=x.device) >= ds_ratio] = 0 # maintaion downsampling ratios to apply
    ### ^ logic
    # choice tensor is filled with nums [1,2,3...] the downsampling factors for each block.
    # Randomly sample for each block a number. If number greater than ds_ratio, make it 0. ie, 
    # dont downsample it, as per the code below. 
    # If ds_ratio = 1, the entire downsampling tensor remains unchanged, and the following code downsamples it 
    # sector by sector.
    # var[torch.rand(B, 1, dg, hg, wg, device=x.device) >= ds_ratio] controls if var maintained or rejected.
    # For everything greater than ds_ratio, set 0, i.e, maintain. i.e, do not mask.
    # i,e, higher the ds_ratio, 
    
    choice = F.interpolate(choice.float(), size=(D, H, W), mode="nearest").long()
 
    out = x.clone()
    for k, f in enumerate(factors, start=1):
        # anti-aliased downsample, then upsample back to full resolution
        lo = F.avg_pool3d(x, kernel_size=f, stride=f)
        lo = F.interpolate(
            lo, size=(D, H, W), mode=up_mode,
            align_corners=False if up_mode == "trilinear" else None,
        )
        out = torch.where(choice == k, lo, out)
 
    return out, (choice > 0).float() # choice > 1 if downsampled else 0.
 
 
@torch.no_grad()
def augment(x, config, gin=None): # manually checked
    """
    x: (B, C, D, H, W). Returns (augmented_input, target, corruption_mask).
 
    mode:
        "none"       -> GIN mix only
        "mask"       -> zero out random blocks (inpainting pretext)
        "downsample" -> reduce resolution of random blocks (super-res pretext)
        "both"       -> apply both corruptions
 
    corruption_mask is 1 where the input was corrupted; compute the
    reconstruction loss there (or weight those voxels up), otherwise the
    untouched regions provide a trivial identity shortcut — especially for
    "downsample", where the input already matches the target at low freq.
    """
    block= config['block']
    gin_apply = config['gin_apply']
    gin_lam_max = config['gin_lam_max']
    downsample_apply = config['downsample_apply']
    downsample_factor = config['downsample_factors']
    downsample_ratio = config['downsample_ratio']
    
    mask_apply = config['mask_apply']
    mask_ratio = config['mask_ratio']
    
    mixed = gin(x) if gin_apply else x.clone()
    if gin_apply:
    
        # lambda bounded to [0, lam_max] so input != target
        lam = gin_lam_max * torch.rand(x.shape[0], 1, 1, 1, 1,
                                device=x.device, dtype=x.dtype)
        mixed = lam * x + (1.0 - lam) * mixed
    
        # energy preservation (per-patch)
        dims = (1, 2, 3, 4)
        e_mix = mixed.norm(p=2, dim=dims, keepdim=True).clamp_min(1e-8)
        e_in = x.norm(p=2, dim=dims, keepdim=True)
        mixed = mixed * (e_in / e_mix)
    
    B, C, D, H, W = x.shape
    corrupt_mask = torch.zeros(B, 1, D, H, W, device=x.device, dtype=x.dtype)
    
    if downsample_apply:
        mixed, ds_mask = _blockwise_downsample(mixed, downsample_factor, downsample_ratio, block)
        corrupt_mask = corrupt_mask + ds_mask
 
    if mask_apply:
        dg, hg, wg = D // block[0], H // block[1], W // block[2]
        m = (torch.rand(B, 1, dg, hg, wg, device=x.device) > mask_ratio).float()
        m = F.interpolate(m, size=(D, H, W), mode="nearest")
        mixed = mixed * m
        corrupt_mask = corrupt_mask + (1.0 - m) #returns what was corrupted
 
    corrupt_mask = corrupt_mask.clamp_(0, 1)
    return mixed, corrupt_mask
 
  
@torch.no_grad()
def augment(x, config, gin=None):
    """Corrupt a volume for a reconstruction pretext task.
 
    Args:
        x: (B, C, D, H, W) clean volume. This is also the reconstruction target.
        config: dict with keys
            block               -- coarse block size, e.g. (8, 8, 8)
            gin_apply           -- bool, apply GIN style mixing
            gin_lam_max         -- lambda upper bound, keeps input != target
            downsample_apply    -- bool, apply block-wise resolution reduction
            downsample_factors  -- candidate factors, e.g. (2, 4)
            downsample_ratio    -- P(a block is downsampled)
            mask_apply          -- bool, zero out random blocks
            mask_ratio          -- P(a block is masked)
        gin: GIN module, required when config['gin_apply'] is True.
 
    Returns:
        mixed:        (B, C, D, H, W) corrupted input for the model
        target:       (B, C, D, H, W) the original x, unmodified
        corrupt_mask: (B, 1, D, H, W) 1 where the input was corrupted, or None
                      if no corruption was applied.
 
    Compute the reconstruction loss on corrupt_mask (or upweight those voxels).
    Untouched regions admit a trivial identity solution -- especially under
    "downsample", where the input already matches the target at low frequency.
    """
    block = config['block']
    gin_apply = config['gin_apply']
    gin_lam_max = config['gin_lam_max']
    gin_conserve_energy = config.get('gin_conserve_enegry', True)
    downsample_apply = config['downsample_apply']
    downsample_factors = config['downsample_factors']
    
    downsample_ratio = config['downsample_ratio']
    mask_apply = config['mask_apply']
    mask_ratio = config['mask_ratio']
    weighted_loss = config.get('weighted_loss', False)
    weighted_loss_weight = config.get('weighted_loss_weight', 1)
    if gin_apply and gin is None:
        raise ValueError("config['gin_apply'] is True but no gin module was passed")
 
    B, _, D, H, W = x.shape
 
    if gin_apply:
        mixed = gin(x)
        # lambda bounded to [0, lam_max] so input != target
        lam = gin_lam_max * torch.rand(B, 1, 1, 1, 1,
                                       device=x.device, dtype=x.dtype)
        mixed = lam * x + (1.0 - lam) * mixed
        # energy preservation (per volume, per batch sample)
        if gin_conserve_energy:
            dims = (1, 2, 3, 4)
            e_in = x.norm(p=2, dim=dims, keepdim=True)
            e_mix = mixed.norm(p=2, dim=dims, keepdim=True).clamp_min(1e-8)
            mixed = mixed * (e_in / e_mix)
    else:
        # clone so later in-place ops can never touch the target
        mixed = x.clone()
 
    corrupt_mask = torch.zeros(B, 1, D, H, W, device=x.device, dtype=x.dtype)
 
    # ORDER MATTERS: downsample first, then mask. A blurred block may then be
    # zeroed (clamp_ collapses the overlap to 1). Blurring *after* masking would
    # smear zeros across block boundaries and leak the mask pattern outward.
    if downsample_apply:
        mixed, ds_mask = _blockwise_downsample(
            mixed, downsample_factors, downsample_ratio, block
        )
        corrupt_mask = corrupt_mask + ds_mask
 
    if mask_apply:
        dg, hg, wg = D // block[0], H // block[1], W // block[2]
        # m == 1 keeps the block, m == 0 masks it; P(mask) = mask_ratio
        m = (torch.rand(B, 1, dg, hg, wg, device=x.device, dtype=x.dtype)
             > mask_ratio).to(x.dtype)
        m = F.interpolate(m, size=(D, H, W), mode="nearest")
        mixed = mixed * m
        corrupt_mask = corrupt_mask + (1.0 - m)
 
 
    corrupt_mask = corrupt_mask.clamp_(0, 1)
    return mixed, corrupt_mask, {'weighted_loss': weighted_loss, 'weighted_loss_weight': weighted_loss_weight}

 
@torch.no_grad()
def _blockwise_downsample_aniso(x, factors=(2, 4), ds_ratio=0.5, block=(8, 8, 8)):
    """Optional variant: downsample only along D (simulates thick-slice scans)."""
    B, C, D, H, W = x.shape
    dg, hg, wg = D // block[0], H // block[1], W // block[2]
 
    choice = torch.randint(1, len(factors) + 1, (B, 1, dg, hg, wg), device=x.device)
    choice[torch.rand(B, 1, dg, hg, wg, device=x.device) >= ds_ratio] = 0
    choice = F.interpolate(choice.float(), size=(D, H, W), mode="nearest").long()
 
    out = x.clone()
    for k, f in enumerate(factors, start=1):
        lo = F.avg_pool3d(x, kernel_size=(f, 1, 1), stride=(f, 1, 1))
        lo = F.interpolate(lo, size=(D, H, W), mode="trilinear", align_corners=False)
        out = torch.where(choice == k, lo, out)
 
    return out, (choice > 0).float()