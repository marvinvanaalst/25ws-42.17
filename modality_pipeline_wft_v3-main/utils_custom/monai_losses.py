# pip install "monai>=1.3" torch
import torch
import torch.nn as nn
from monai.losses import PerceptualLoss


def zscore(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-volume z-score over spatial dims. Differentiable.
    Input: (B, 1, D, H, W) in [-1, 1]."""
    mean = x.mean(dim=(2, 3, 4), keepdim=True)
    std = x.std(dim=(2, 3, 4), keepdim=True)
    return (x - mean) / (std + eps)


class MRIPerceptualLoss(nn.Module):
    def __init__(self, network_type="medicalnet_resnet50_23datasets"):
        super().__init__()
        self.perc = PerceptualLoss(
            spatial_dims=3,
            network_type=network_type,
            is_fake_3d=False,
        )

    def forward(self, pred, target):
        # pred, target: (B, 1, 64, 64, 64) in [-1, 1]
        return self.perc(zscore(pred), zscore(target))


if __name__ == '__main__':
    device = torch.device("cuda")
    perc_loss = MRIPerceptualLoss().to(device)
    l1 = nn.L1Loss()
    lambda_perc = 0.05

    def loss_fn(pred, target):
        return l1(pred, target) + lambda_perc * perc_loss(pred, target)
