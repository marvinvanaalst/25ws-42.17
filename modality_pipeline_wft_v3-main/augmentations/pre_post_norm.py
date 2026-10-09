import torch

EPS = 1e-8

# ---------------------------------------------------------------------------
# Masked per-sample statistics (all reduce over every dim except batch)
# ---------------------------------------------------------------------------

def _reduce_dims(x):
    return tuple(range(1, x.dim()))

def _masked_mean_std(x, mask):
    dims = _reduce_dims(x)
    n = mask.sum(dim=dims, keepdim=True).clamp(min=1)
    mean = (x * mask).sum(dim=dims, keepdim=True) / n
    var = ((x - mean) ** 2 * mask).sum(dim=dims, keepdim=True) / n
    return mean, var.sqrt()

def _masked_median_mad(x, mask, scaled=True):
    B = x.shape[0]
    stat_shape = (B,) + (1,) * (x.dim() - 1)

    x_nan = torch.where(mask.bool(), x, torch.nan)   # nanmedian skips NaNs
    flat = x_nan.reshape(B, -1)

    median = flat.nanmedian(dim=1).values.reshape(stat_shape)
    mad = (x_nan - median).abs().reshape(B, -1).nanmedian(dim=1).values.reshape(stat_shape)
    if scaled:
        mad = mad * 1.4826                            # MAD -> sigma for Gaussian data

    # a fully-masked sample yields NaN stats; let it pass through as zeros
    return torch.nan_to_num(median), torch.nan_to_num(mad)

def _masked_min_max(x, mask):
    dims = _reduce_dims(x)
    valid = mask.bool()
    lo = x.masked_fill(~valid, float("inf")).amin(dim=dims, keepdim=True)
    hi = x.masked_fill(~valid, float("-inf")).amax(dim=dims, keepdim=True)

    # fully-masked sample -> inf/-inf; fall back to 0/1 so it passes through
    bad = ~torch.isfinite(lo)
    lo = torch.where(bad, torch.zeros_like(lo), lo)
    hi = torch.where(bad, torch.ones_like(hi), hi)
    return lo, hi

# ---------------------------------------------------------------------------
# Pointwise bijections (no per-sample stats needed)
# ---------------------------------------------------------------------------

def _signed_log1p(x):
    return x.sign() * torch.log1p(x.abs())

def _signed_expm1(y):
    return y.sign() * torch.expm1(y.abs())

def _yeo_johnson(x, lam):
    pos, neg = x >= 0, x < 0
    out = torch.empty_like(x)
    if lam != 0:
        out[pos] = (torch.pow(x[pos] + 1, lam) - 1) / lam
    else:
        out[pos] = torch.log1p(x[pos])
    if lam != 2:
        out[neg] = -(torch.pow(1 - x[neg], 2 - lam) - 1) / (2 - lam)
    else:
        out[neg] = -torch.log1p(-x[neg])
    return out

def _yeo_johnson_inverse(y, lam):
    pos, neg = y >= 0, y < 0
    out = torch.empty_like(y)
    if lam != 0:
        out[pos] = torch.pow(y[pos] * lam + 1, 1 / lam) - 1
    else:
        out[pos] = torch.expm1(y[pos])
    if lam != 2:
        out[neg] = 1 - torch.pow(1 - (2 - lam) * y[neg], 1 / (2 - lam))
    else:
        out[neg] = -torch.expm1(-y[neg])
    return out

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_per_sample_masked(x, mask, method="mean", lam=1.0):
    """
    Normalize each sample in the batch using only positions where mask == 1.

    method:
        "mean"    -> (x - mean) / std
        "median"  -> (x - median) / MAD
        "minmax"  -> (x - min) / (max - min)         [maps valid values to 0..1]
        "log"     -> sign(x) * log1p(|x|)            [pointwise, no stats]
        "power"   -> Yeo-Johnson with exponent lam   [pointwise, no stats]

    Returns (x_norm, params); pass params to `denormalize` to invert exactly.
    """
    mask = mask.float().expand_as(x)
    if method == "mean":
        center, scale = _masked_mean_std(x, mask)
    elif method == "median":
        center, scale = _masked_median_mad(x, mask)
    elif method == "minmax":
        lo, hi = _masked_min_max(x, mask)
        center, scale = lo, hi - lo
    elif method == "log":
        x_norm = _signed_log1p(x) * mask
        return x_norm, {"method": "log"}
    elif method == "power":
        x_norm = _yeo_johnson(x, lam) * mask
        return x_norm, {"method": "power", "lam": lam}
    else:
        raise ValueError(f"unknown method: {method!r}")

    # shared affine path for mean / median / minmax
    x_norm = (x - center) / (scale + EPS) * mask
    return x_norm, {"method": method, "center": center, "scale": scale}


def denormalize(y, params, mask=None):
    """Invert `normalize_per_sample_masked` given its returned params."""
    method = params["method"]

    if method in ("mean", "median", "minmax"):
        y = y * (params["scale"] + EPS) + params["center"]
    elif method == "log":
        y = _signed_expm1(y)
    elif method == "power":
        y = _yeo_johnson_inverse(y, params["lam"])
    else:
        raise ValueError(f"unknown method: {method!r}")

    if mask is not None:
        y = y * mask.float()
    return y