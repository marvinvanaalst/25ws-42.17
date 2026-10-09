import itertools
import torch

SPATIAL = (2, 3, 4) # Dealing with batches here, B,C,H,W,D
def flip_combos(axes=SPATIAL):
    return [c for r in range(len(axes) + 1) for c in itertools.combinations(axes, r)]
@torch.no_grad()
def tta_forward(model, x, axes=SPATIAL):
    combos = flip_combos(axes)
    acc = None
    for dims in combos:
        xa = x.flip(dims) if dims else x
        y = model(xa)
        y = (y.flip(dims) if dims else y).float()   # fp32 accumulation
        acc = y if acc is None else acc + y
    return acc / len(combos)

SPATIAL_2D = (2, 3)  # Dealing with slice batches here, B,C,H,W


def flip_combos_2d(axes=SPATIAL_2D):
    return flip_combos(axes)


@torch.no_grad()
def tta_forward_2d(model, x, axes=SPATIAL_2D, forward=None):
    """Flip-averaged inference for 2D slice models.

    `forward` lets a caller inject the single-channel squeeze/unsqueeze wrapper
    used by `single_channel_model` configs; it defaults to calling the model.
    """
    if forward is None:
        forward = model
    combos = flip_combos(axes)
    acc = None
    for dims in combos:
        xa = x.flip(dims) if dims else x
        y = forward(xa)
        y = (y.flip(dims) if dims else y).float()   # fp32 accumulation
        acc = y if acc is None else acc + y
    return acc / len(combos)
