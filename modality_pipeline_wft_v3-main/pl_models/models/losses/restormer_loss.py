import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision.models.feature_extraction import create_feature_extractor

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.epsilon ** 2))

import lpips

class LPIPSLoss3D(nn.Module):
    def __init__(self, net='alex', axis=2):  # axis = 2 for axial slices
        super().__init__()
        self.lpips = lpips.LPIPS(net=net)
        self.axis = axis

    def forward(self, pred, target):
        # pred/target: [B, C, H, W, D] (3D volumes)
        assert pred.shape == target.shape
        B, C, H, W, D = pred.shape

        if C == 1:
            pred = pred.repeat(1, 3, 1, 1, 1)
            target = target.repeat(1, 3, 1, 1, 1)

        # LPIPS expects [B, 3, H, W] images in [-1, 1]
        pred = pred * 2 - 1
        target = target * 2 - 1

        lpips_total = 0.0
        count = 0

        for i in range(D):
            pred_slice = pred[:, :, :, :, i]  # [B, 3, H, W]
            target_slice = target[:, :, :, :, i]
            lpips_total += self.lpips(pred_slice, target_slice).mean()
            count += 1

        return lpips_total / count


class UnifiedAdaptiveLoss(nn.Module):
    def __init__(self, use_perceptual=False, perceptual_net='alex'):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.use_perceptual = use_perceptual
        if use_perceptual:
            self.perceptual = LPIPSLoss3D(net=perceptual_net)

        self.log_sigma_char = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_l1 = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_l2 = nn.Parameter(torch.tensor(0.0))
        if use_perceptual:
            self.log_sigma_perc = nn.Parameter(torch.tensor(0.0))

    def forward(self, pred, target):
        loss = 0.0

        charbonnier = self.charbonnier(pred, target)
        loss += torch.exp(-self.log_sigma_char) * charbonnier + self.log_sigma_char

        l1 = self.l1(pred, target)
        loss += torch.exp(-self.log_sigma_l1) * l1 + self.log_sigma_l1

        l2 = self.l2(pred, target)
        loss += torch.exp(-self.log_sigma_l2) * l2 + self.log_sigma_l2

        if self.use_perceptual:
            perc = self.perceptual(pred, target)
            loss += torch.exp(-self.log_sigma_perc) * perc + self.log_sigma_perc

        return loss