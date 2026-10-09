"""
losses_library.py
=================
Complete loss function library for the 2.5D UNet loss ablation study.
ALL implementations are pure PyTorch — zero external dependencies.
Place this file alongside your training script on the HPC.

Usage:
    from losses_library import get_loss
    criterion = get_loss("exp10_msssim_l1")
    loss = criterion(pred, target)

Author: MTP Project — 3T->7T MRI Synthesis
Date: September 2026
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# BUILDING BLOCK 1: SSIM (Structural Similarity)
# Paper: Wang et al., IEEE TIP, 2004
# =============================================================================

def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    """Creates a 1D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _create_window(window_size: int, n_channels: int) -> torch.Tensor:
    """Creates a 2D Gaussian window for SSIM computation."""
    kernel_1d = _gaussian_kernel_1d(window_size, sigma=1.5)
    kernel_2d = kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)
    window = kernel_2d.unsqueeze(0).unsqueeze(0)
    window = window.expand(n_channels, 1, window_size, window_size).contiguous()
    return window


def _ssim_per_channel(pred, target, window, window_size,
                       C1=0.01**2, C2=0.03**2):
    n_ch = pred.size(1)
    pad  = window_size // 2
    win  = window.to(pred.device)

    mu1 = F.conv2d(pred,   win, padding=pad, groups=n_ch)
    mu2 = F.conv2d(target, win, padding=pad, groups=n_ch)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred,     win, padding=pad, groups=n_ch) - mu1_sq
    sigma2_sq = F.conv2d(target * target, win, padding=pad, groups=n_ch) - mu2_sq
    sigma12   = F.conv2d(pred * target,   win, padding=pad, groups=n_ch) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


class SSIMLoss(nn.Module):
    """
    Experiment 2 (exp09): SSIM loss only.
    L = 1 - SSIM(pred, target)
    Captures luminance, contrast, and structure jointly.
    Paper: Wang et al., IEEE TIP 2004.
    """
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self._cache = {}

    def _get_window(self, n_ch, device):
        key = (n_ch, str(device))
        if key not in self._cache:
            self._cache[key] = _create_window(self.window_size, n_ch).to(device)
        return self._cache[key]

    def forward(self, pred, target):
        w = self._get_window(pred.size(1), pred.device)
        return 1.0 - _ssim_per_channel(pred, target, w, self.window_size)


# =============================================================================
# BUILDING BLOCK 2: MS-SSIM + L1
# Papers: Wang et al. Asilomar 2003; Zhao et al. IEEE T-CI 2017
# =============================================================================

class MSSSIMLoss(nn.Module):
    """Multi-Scale SSIM at 5 scales. Scale weights from Wang et al. 2003."""
    _SCALE_WEIGHTS = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self._cache = {}

    def _get_window(self, n_ch, device):
        key = (n_ch, str(device))
        if key not in self._cache:
            self._cache[key] = _create_window(self.window_size, n_ch).to(device)
        return self._cache[key]

    def forward(self, pred, target):
        w = self._get_window(pred.size(1), pred.device)
        ms_ssim = 1.0
        p, t = pred, target
        for i, wt in enumerate(self._SCALE_WEIGHTS):
            ms_ssim = ms_ssim * (_ssim_per_channel(p, t, w, self.window_size) ** wt)
            if i < len(self._SCALE_WEIGHTS) - 1:
                p = F.avg_pool2d(p, 2, 2)
                t = F.avg_pool2d(t, 2, 2)
        return 1.0 - ms_ssim


class MSSSIMPlusL1Loss(nn.Module):
    """
    Experiment 3 (exp10): MS-SSIM + L1.
    L = alpha*L1 + (1-alpha)*(1-MS-SSIM)   alpha=0.84 per Zhao et al. 2017.
    Paper: Zhao et al., IEEE Transactions on Computational Imaging, 2017.
    """
    def __init__(self, alpha=0.84):
        super().__init__()
        self.alpha   = alpha
        self.ms_ssim = MSSSIMLoss()

    def forward(self, pred, target):
        return self.alpha * F.l1_loss(pred, target) + (1.0 - self.alpha) * self.ms_ssim(pred, target)


# =============================================================================
# BUILDING BLOCK 3: SOBEL GRADIENT LOSS
# Paper: Mathieu, Couprie, LeCun — ICLR 2016
# =============================================================================

class SobelGradientLoss(nn.Module):
    """
    Experiment 4 (exp11): L1 + Sobel gradient sharpness loss.
    L = L1(pred,target) + lambda_g*(L1(Gx_p,Gx_t) + L1(Gy_p,Gy_t))
    Fixed Sobel kernels (not learned). Fully offline.
    Paper: Mathieu et al., ICLR 2016 (Gradient Difference Loss).
    """
    def __init__(self, lambda_g=0.5):
        super().__init__()
        self.lambda_g = lambda_g
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sy = torch.tensor([[-1.,-2.,-1.], [ 0., 0., 0.], [ 1., 2., 1.]])
        self.register_buffer('sobel_x', sx.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sy.view(1, 1, 3, 3))

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        sx = self.sobel_x.to(dtype=pred.dtype, device=pred.device)
        sy = self.sobel_y.to(dtype=pred.dtype, device=pred.device)
        gx_p = F.conv2d(pred,   sx, padding=1)
        gy_p = F.conv2d(pred,   sy, padding=1)
        gx_t = F.conv2d(target, sx, padding=1)
        gy_t = F.conv2d(target, sy, padding=1)
        grad = F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)
        return l1 + self.lambda_g * grad


# =============================================================================
# BUILDING BLOCK 4: HAAR WAVELET LOSS (our existing loss, cleaned up)
# Papers: Haar 1910; Liu et al. CVPR-W 2018
# =============================================================================

def _haar_dwt_2d(x):
    """2D Haar DWT. Input [B,C,H,W]. Returns LL,LH,HL,HH each [B,C,H/2,W/2]."""
    L = (x[:,:,:,0::2] + x[:,:,:,1::2]) / 2.0
    H = (x[:,:,:,0::2] - x[:,:,:,1::2]) / 2.0
    LL = (L[:,:,0::2,:] + L[:,:,1::2,:]) / 2.0
    LH = (L[:,:,0::2,:] - L[:,:,1::2,:]) / 2.0
    HL = (H[:,:,0::2,:] + H[:,:,1::2,:]) / 2.0
    HH = (H[:,:,0::2,:] - H[:,:,1::2,:]) / 2.0
    return LL, LH, HL, HH


class HaarWaveletLoss(nn.Module):
    """
    Experiment 5 (exp07 - already run ~30.8dB): Haar wavelet L1.
    Detail sub-bands (LH,HL,HH) weighted 2x vs approximation (LL).
    Papers: Haar 1910; Liu et al. CVPR-W 2018.
    """
    def __init__(self, detail_weight=2.0):
        super().__init__()
        self.dw = detail_weight

    def forward(self, pred, target):
        LL_p, LH_p, HL_p, HH_p = _haar_dwt_2d(pred)
        LL_t, LH_t, HL_t, HH_t = _haar_dwt_2d(target)
        ll = F.l1_loss(LL_p, LL_t)
        det = F.l1_loss(LH_p,LH_t) + F.l1_loss(HL_p,HL_t) + F.l1_loss(HH_p,HH_t)
        return ll + self.dw * det


# =============================================================================
# BUILDING BLOCK 5: DAUBECHIES db2 WAVELET LOSS
# Papers: Daubechies 1988; Huang et al. ICCV 2017
# =============================================================================

def _db2_dwt_1d(x, lo, hi):
    """
    1D db2 DWT along LAST dimension of x via circular-padded conv1d.
    x:  [N, L] where N=B*C*H and L=W (or transposed)
    lo, hi: [1,1,4] filter buffers
    Returns (approx, detail) each [N, L//2]
    """
    lo = lo.to(dtype=x.dtype, device=x.device)
    hi = hi.to(dtype=x.dtype, device=x.device)
    pad = lo.size(-1) - 1
    x_pad = F.pad(x.unsqueeze(1), (pad, 0), mode='circular')  # [N,1,L+pad]
    approx = F.conv1d(x_pad, lo, stride=2).squeeze(1)
    detail = F.conv1d(x_pad, hi, stride=2).squeeze(1)
    return approx, detail


def _db2_dwt_2d(x, lo, hi):
    """
    2D db2 DWT. Separable: apply 1D along W then along H.
    x: [B,C,H,W] -> LL,LH,HL,HH each [B,C,H//2,W//2]
    """
    B, C, H, W = x.shape
    # Along width: treat [B,C,H] as batch, signals of length W
    x_w = x.reshape(B * C * H, W)
    L_w, H_w = _db2_dwt_1d(x_w, lo, hi)                # [B*C*H, W//2]
    L_w = L_w.reshape(B, C, H, W // 2)
    H_w = H_w.reshape(B, C, H, W // 2)

    # Along height: treat [B,C,W//2] as batch, signals of length H
    L_h = L_w.permute(0, 1, 3, 2).reshape(B * C * (W // 2), H)
    H_h = H_w.permute(0, 1, 3, 2).reshape(B * C * (W // 2), H)

    LL_h, LH_h = _db2_dwt_1d(L_h, lo, hi)              # [B*C*W//2, H//2]
    HL_h, HH_h = _db2_dwt_1d(H_h, lo, hi)

    LL = LL_h.reshape(B, C, W // 2, H // 2).permute(0, 1, 3, 2)
    LH = LH_h.reshape(B, C, W // 2, H // 2).permute(0, 1, 3, 2)
    HL = HL_h.reshape(B, C, W // 2, H // 2).permute(0, 1, 3, 2)
    HH = HH_h.reshape(B, C, W // 2, H // 2).permute(0, 1, 3, 2)
    return LL, LH, HL, HH


class DaubechiesWaveletLoss(nn.Module):
    """
    Experiment 6 (exp12): L1 + Daubechies db2 wavelet loss.
    L = L1(pred,target) + lambda_w * DbWaveletL1(pred,target)

    WHY db2 OVER HAAR:
      Haar = 2-tap step function -> Gibbs ringing on smooth gradients
      db2  = 4-tap smooth filter, 2 vanishing moments -> better for curved
             cortical sulci and smoothly varying gray/white matter contrast.

    db2 low-pass coefficients (analytically derived):
      h = [(1+sqrt3), (3+sqrt3), (3-sqrt3), (1-sqrt3)] / (4*sqrt2)
    High-pass = QMF (Quadrature Mirror Filter) of low-pass.

    Papers:
      Daubechies, I. (1988). Orthonormal bases of compactly supported wavelets.
        Communications on Pure and Applied Mathematics, 41(7), 909-996.
      Huang, H. et al. (2017). Wavelet-SRNet. ICCV.
    """
    def __init__(self, lambda_w=0.5, detail_weight=2.0):
        super().__init__()
        self.lambda_w = lambda_w
        self.dw       = detail_weight

        s3 = math.sqrt(3.0)
        s2 = math.sqrt(2.0)
        h = torch.tensor([
            (1.0 + s3) / (4.0 * s2),
            (3.0 + s3) / (4.0 * s2),
            (3.0 - s3) / (4.0 * s2),
            (1.0 - s3) / (4.0 * s2),
        ])
        g = torch.tensor([
             (1.0 - s3) / (4.0 * s2),
            -(3.0 - s3) / (4.0 * s2),
             (3.0 + s3) / (4.0 * s2),
            -(1.0 + s3) / (4.0 * s2),
        ])
        self.register_buffer('lo', h.view(1, 1, 4))
        self.register_buffer('hi', g.view(1, 1, 4))

    def _wavelet_l1(self, pred, target):
        LL_p, LH_p, HL_p, HH_p = _db2_dwt_2d(pred,   self.lo, self.hi)
        LL_t, LH_t, HL_t, HH_t = _db2_dwt_2d(target, self.lo, self.hi)
        ll  = F.l1_loss(LL_p, LL_t)
        det = F.l1_loss(LH_p,LH_t) + F.l1_loss(HL_p,HL_t) + F.l1_loss(HH_p,HH_t)
        return ll + self.dw * det

    def forward(self, pred, target):
        return F.l1_loss(pred, target) + self.lambda_w * self._wavelet_l1(pred, target)


# =============================================================================
# BUILDING BLOCK 6: LOCAL CONTRAST (Local Std) LOSS
# Novel MRI-motivated component — no single paper; concept from texture synthesis
# =============================================================================

class LocalContrastLoss(nn.Module):
    """
    Experiment 7 (exp13): L1 + local contrast (local std deviation) loss.
    L = L1(pred,target) + lambda_c * L1(LocalStd(pred), LocalStd(target))

    LocalStd(x)[i,j] = std dev of x in 11x11 window centred at (i,j).
    Computed as sqrt(E[x^2] - E[x]^2) using Gaussian sliding window.

    WHY THIS HELPS:
    7T MRI has better TISSUE CONTRAST than 3T. Pure L1 optimizes mean intensity
    but ignores local contrast. This loss DIRECTLY penalizes contrast mismatch.
    Expected to improve gray/white matter and cortex/sulcus distinction.
    """
    def __init__(self, lambda_c=0.5, window_size=11):
        super().__init__()
        self.lambda_c    = lambda_c
        self.window_size = window_size
        self._cache = {}

    def _get_window(self, n_ch, device):
        key = (n_ch, str(device))
        if key not in self._cache:
            self._cache[key] = _create_window(self.window_size, n_ch).to(device)
        return self._cache[key]

    def _local_std(self, x):
        n_ch = x.size(1)
        win  = self._get_window(n_ch, x.device)
        pad  = self.window_size // 2
        mu   = F.conv2d(x,     win, padding=pad, groups=n_ch)
        mu2  = F.conv2d(x * x, win, padding=pad, groups=n_ch)
        return (mu2 - mu * mu).clamp(min=1e-8).sqrt()

    def forward(self, pred, target):
        l1  = F.l1_loss(pred, target)
        con = F.l1_loss(self._local_std(pred), self._local_std(target))
        return l1 + self.lambda_c * con


# =============================================================================
# BUILDING BLOCK 7: FFT FOCAL FREQUENCY LOSS
# Paper: Jiang et al., ICCV 2021
# =============================================================================

class FocalFrequencyLoss(nn.Module):
    """
    Experiment 8 (exp14): L1 + FFT focal frequency loss.
    Weight per frequency = 1 - cos(phase_pred - phase_target).
    Focuses the loss on hard-to-predict frequencies (large phase mismatch).

    Paper: Jiang, L., Dai, B., Wu, W., & Loy, C. C. (2021).
           Focal Frequency Loss for Image Reconstruction and Synthesis.
           ICCV 2021.
    """
    def __init__(self, lambda_f=0.3, alpha=1.0):
        super().__init__()
        self.lambda_f = lambda_f
        self.alpha    = alpha

    def _focal_freq(self, pred, target):
        fft_p = torch.fft.fft2(pred,   norm='ortho')
        fft_t = torch.fft.fft2(target, norm='ortho')
        weight = 1.0 - torch.cos(fft_p.angle() - fft_t.angle())
        return (weight * (fft_p.abs() - fft_t.abs()).abs().pow(self.alpha)).mean()

    def forward(self, pred, target):
        return F.l1_loss(pred, target) + self.lambda_f * self._focal_freq(pred, target)


# =============================================================================
# COMPOUND LOSSES (Experiments 9-15)
# =============================================================================

class CompoundLossV1(nn.Module):
    """
    Experiment 9 (exp15): Haar Wavelet + Sobel Gradient + L1
    L = L1 + 0.5*HaarWaveletL1 + 0.3*SobelGradL1
    Each term supervises a different signal:
      L1    -> global pixel accuracy (PSNR)
      Haar  -> multi-scale frequency content (texture)
      Sobel -> local edge sharpness (sulcal boundaries)
    """
    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sy = torch.tensor([[-1.,-2.,-1.], [ 0., 0., 0.], [ 1., 2., 1.]])
        self.register_buffer('sobel_x', sx.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sy.view(1, 1, 3, 3))

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)

        LL_p,LH_p,HL_p,HH_p = _haar_dwt_2d(pred)
        LL_t,LH_t,HL_t,HH_t = _haar_dwt_2d(target)
        haar = (F.l1_loss(LL_p,LL_t) +
                2.0*(F.l1_loss(LH_p,LH_t)+F.l1_loss(HL_p,HL_t)+F.l1_loss(HH_p,HH_t)))

        sx = self.sobel_x.to(dtype=pred.dtype, device=pred.device)
        sy = self.sobel_y.to(dtype=pred.dtype, device=pred.device)
        gx_p = F.conv2d(pred,   sx, padding=1)
        gy_p = F.conv2d(pred,   sy, padding=1)
        gx_t = F.conv2d(target, sx, padding=1)
        gy_t = F.conv2d(target, sy, padding=1)
        sobel = F.l1_loss(gx_p,gx_t) + F.l1_loss(gy_p,gy_t)

        return l1 + 0.5*haar + 0.3*sobel


class CompoundLossV2(nn.Module):
    """
    Experiment 10 (exp16): Daubechies db2 Wavelet + Contrast (local-std) + L1
    L = L1 + 0.5*Db2WaveletL1 + 0.5*ContrastL1
    Targets 7T-specific improvements: smooth-edge frequency + tissue contrast.
    """
    def __init__(self):
        super().__init__()
        self.db2      = DaubechiesWaveletLoss(lambda_w=0.0)
        self.contrast = LocalContrastLoss(lambda_c=0.0)

    def forward(self, pred, target):
        l1  = F.l1_loss(pred, target)
        db2 = self.db2._wavelet_l1(pred, target)
        con = F.l1_loss(self.contrast._local_std(pred),
                        self.contrast._local_std(target))
        return l1 + 0.5*db2 + 0.5*con


class CompoundLossV3(nn.Module):
    """
    Experiment 11 (exp17): Daubechies db2 Wavelet + Sobel Gradient + L1
    L = L1 + 0.5*Db2WaveletL1 + 0.3*SobelGradL1
    Smooth-curve wavelets (db2) + sharp edge detection (Sobel) are complementary.
    """
    def __init__(self):
        super().__init__()
        self.db2 = DaubechiesWaveletLoss(lambda_w=0.0)
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sy = torch.tensor([[-1.,-2.,-1.], [ 0., 0., 0.], [ 1., 2., 1.]])
        self.register_buffer('sobel_x', sx.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sy.view(1, 1, 3, 3))

    def forward(self, pred, target):
        l1  = F.l1_loss(pred, target)
        db2 = self.db2._wavelet_l1(pred, target)
        sx = self.sobel_x.to(dtype=pred.dtype, device=pred.device)
        sy = self.sobel_y.to(dtype=pred.dtype, device=pred.device)
        gxp = F.conv2d(pred,   sx, padding=1)
        gyp = F.conv2d(pred,   sy, padding=1)
        gxt = F.conv2d(target, sx, padding=1)
        gyt = F.conv2d(target, sy, padding=1)
        sobel = F.l1_loss(gxp,gxt) + F.l1_loss(gyp,gyt)
        return l1 + 0.5*db2 + 0.3*sobel


class CompoundLossV4(nn.Module):
    """
    Experiment 12 (exp18): MS-SSIM + Daubechies Wavelet
    L = 0.84*Db2WaveletL1 + 0.16*(1 - MS-SSIM)
    Replaces L1 in the MS-SSIM formula with the richer db2 wavelet loss.
    """
    def __init__(self):
        super().__init__()
        self.db2    = DaubechiesWaveletLoss(lambda_w=0.0)
        self.msssim = MSSSIMLoss()

    def forward(self, pred, target):
        db2 = self.db2._wavelet_l1(pred, target)
        mss = self.msssim(pred, target)
        return 0.84*db2 + 0.16*mss


class CompoundLossV5(nn.Module):
    """
    Experiment 13 (exp19): MS-SSIM + Contrast Loss
    L = 0.84*L1 + 0.08*(1-MS-SSIM) + 0.08*ContrastL1
    Adds local contrast on top of the standard MS-SSIM+L1 combination.
    """
    def __init__(self):
        super().__init__()
        self.msssim   = MSSSIMLoss()
        self.contrast = LocalContrastLoss(lambda_c=0.0)

    def forward(self, pred, target):
        l1  = F.l1_loss(pred, target)
        mss = self.msssim(pred, target)
        con = F.l1_loss(self.contrast._local_std(pred),
                        self.contrast._local_std(target))
        return 0.84*l1 + 0.08*mss + 0.08*con


class CompoundLossV6(nn.Module):
    """
    Experiment 14 (exp20): FULL KITCHEN SINK
    Db2 Wavelet + Contrast + Sobel Gradient + L1
    L = L1 + 0.5*Db2WaveletL1 + 0.3*ContrastL1 + 0.2*SobelGradL1
    """
    def __init__(self):
        super().__init__()
        self.db2      = DaubechiesWaveletLoss(lambda_w=0.0)
        self.contrast = LocalContrastLoss(lambda_c=0.0)
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sy = torch.tensor([[-1.,-2.,-1.], [ 0., 0., 0.], [ 1., 2., 1.]])
        self.register_buffer('sobel_x', sx.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sy.view(1, 1, 3, 3))

    def forward(self, pred, target):
        l1  = F.l1_loss(pred, target)
        db2 = self.db2._wavelet_l1(pred, target)
        con = F.l1_loss(self.contrast._local_std(pred),
                        self.contrast._local_std(target))
        sx = self.sobel_x.to(dtype=pred.dtype, device=pred.device)
        sy = self.sobel_y.to(dtype=pred.dtype, device=pred.device)
        gxp = F.conv2d(pred,   sx, padding=1)
        gyp = F.conv2d(pred,   sy, padding=1)
        gxt = F.conv2d(target, sx, padding=1)
        gyt = F.conv2d(target, sy, padding=1)
        sobel = F.l1_loss(gxp,gxt) + F.l1_loss(gyp,gyt)
        return l1 + 0.5*db2 + 0.3*con + 0.2*sobel


class CompoundLossV7(nn.Module):
    """
    Experiment 15 (exp21): FFT Focal Frequency + Daubechies Wavelet + L1
    L = L1 + 0.3*FocalFreqLoss + 0.5*Db2WaveletL1
    FFT = global frequency spectrum. Wavelet = spatially local frequency.
    Complementary pair covering both global and local frequency supervision.
    """
    def __init__(self):
        super().__init__()
        self.fft = FocalFrequencyLoss(lambda_f=0.0)
        self.db2 = DaubechiesWaveletLoss(lambda_w=0.0)

    def forward(self, pred, target):
        l1  = F.l1_loss(pred, target)
        fft = self.fft._focal_freq(pred, target)
        db2 = self.db2._wavelet_l1(pred, target)
        return l1 + 0.3*fft + 0.5*db2


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

_LOSS_REGISTRY = {
    # Already done (for reference)
    "exp06_l1_baseline":           lambda: nn.L1Loss(),
    "exp07_haar_wavelet":          lambda: HaarWaveletLoss(detail_weight=2.0),

    # New single losses
    "exp09_ssim_only":             lambda: SSIMLoss(window_size=11),
    "exp10_msssim_l1":             lambda: MSSSIMPlusL1Loss(alpha=0.84),
    "exp11_sobel_l1":              lambda: SobelGradientLoss(lambda_g=0.5),
    "exp12_db2_l1":                lambda: DaubechiesWaveletLoss(lambda_w=0.5, detail_weight=2.0),
    "exp13_contrast_l1":           lambda: LocalContrastLoss(lambda_c=0.5),
    "exp14_fft_l1":                lambda: FocalFrequencyLoss(lambda_f=0.3),

    # Compound losses
    "exp15_haar_sobel_l1":         lambda: CompoundLossV1(),
    "exp16_db2_contrast_l1":       lambda: CompoundLossV2(),
    "exp17_db2_sobel_l1":          lambda: CompoundLossV3(),
    "exp18_msssim_db2":            lambda: CompoundLossV4(),
    "exp19_msssim_contrast":       lambda: CompoundLossV5(),
    "exp20_db2_contrast_sobel_l1": lambda: CompoundLossV6(),
    "exp21_fft_db2_l1":            lambda: CompoundLossV7(),
}


class AMPLossWrapper(nn.Module):
    """
    Wraps any loss module so it always executes in float32 precision.
    Prevents 'Input type (torch.cuda.HalfTensor) and weight type (torch.FloatTensor)
    should be the same' errors when Automatic Mixed Precision (AMP) is enabled,
    and protects against numerical underflow in gradients.
    """
    def __init__(self, loss_fn: nn.Module):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, pred, target):
        return self.loss_fn(pred.float(), target.float())


def get_loss(name: str) -> nn.Module:
    """
    Factory: returns loss module for the given experiment name.
    Example:
        criterion = get_loss("exp10_msssim_l1")
        loss = criterion(pred, target)
    """
    if name not in _LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss '{name}'. Available:\n" +
            "\n".join(f"  {k}" for k in _LOSS_REGISTRY)
        )
    return AMPLossWrapper(_LOSS_REGISTRY[name]())


def list_losses():
    print("Available loss names:")
    for k in _LOSS_REGISTRY:
        print(f"  {k}")


# =============================================================================
# SANITY CHECK — run: python losses_library.py
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("Sanity-checking all 15 loss functions on float32 AND float16...")
    print("=" * 65)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for dtype_name, dt in [("Float32", torch.float32), ("Float16 (AMP)", torch.float16)]:
        print(f"\n--- Testing with {dtype_name} on {device} ---")
        pred   = torch.randn(2, 1, 64, 64, device=device, dtype=dt, requires_grad=True)
        target = torch.randn(2, 1, 64, 64, device=device, dtype=dt)
        all_ok = True
        for name in _LOSS_REGISTRY:
            try:
                loss_fn = get_loss(name)
                if hasattr(loss_fn, 'to'):
                    loss_fn = loss_fn.to(device)
                val = loss_fn(pred, target)
                assert torch.isfinite(val), "Non-finite!"
                val.backward(retain_graph=True)
                print(f"  OK  {name:<40}  loss={val.item():.6f}")
            except Exception as e:
                print(f"  FAIL  {name:<38}  ERROR: {e}")
                all_ok = False
        print("RESULT:", "ALL PASSED" if all_ok else "SOME FAILED")
