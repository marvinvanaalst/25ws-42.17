# 2.5D UNet Loss Ablation Study Results

**Model Architecture:** 2.5D UNet (5 adjacent axial slices input, 1 center target slice output)
**Upsampling:** Bilinear
**Optimizer:** Adam (lr=1e-4, betas=(0.9, 0.999))

| Rank | Experiment | Category | Loss Formula | PSNR (mean ± std) | SSIM | Δ vs Exp 6 (L1) | Status |
|:----:|:-----------|:---------|:-------------|:------------------|:----:|:---------------:|:------:|
| 1 | Exp 14: FFT Focal Frequency + L1 | Frequency Spectrum | `L1 + 0.3*FocalFreq` | **32.68** ± 0.94 | 0.9376 | **+0.07 dB** | COMPLETED |
| 2 | Exp 6: L1 Baseline | Baseline | `L1` | **32.61** | 0.9100 | *Baseline* | COMPLETED (Baseline) |
| 3 | Exp 7: Haar Wavelet (WFT) | Wavelet | `Haar Wavelet L1 (LL + 2*Det)` | **30.80** | 0.9000 | **-1.81 dB** | COMPLETED (Exp 7) |
| 4 | Exp 10: MS-SSIM + L1 | Multi-Scale | `0.84*L1 + 0.16*(1-MS-SSIM)` | **16.06** ± 2.72 | 0.6795 | **-16.55 dB** | COMPLETED |
| 5 | Exp 19: MS-SSIM + Contrast | Compound | `0.84*L1 + 0.08*(1-MSSSIM) + 0.08*Contrast` | **15.81** ± 2.63 | 0.6813 | **-16.80 dB** | COMPLETED |
| 6 | Exp 13: Local Contrast + L1 | Tissue Contrast | `L1 + 0.5*LocalStdL1` | **15.56** ± 2.59 | 0.6754 | **-17.05 dB** | COMPLETED |
| 7 | Exp 9: SSIM Loss | Structural | `1 - SSIM` | **15.38** ± 3.56 | 0.6789 | **-17.23 dB** | COMPLETED |
| - | Exp 11: Sobel Gradient + L1 | Edge Sharpness | `L1 + 0.5*SobelGrad` | TBD | TBD | - | RUNNING / QUEUED |
| - | Exp 12: Daubechies db2 Wavelet + L1 | Wavelet (Novel) | `L1 + 0.5*Db2WaveletL1` | TBD | TBD | - | RUNNING / QUEUED |
| - | Exp 15: Haar Wavelet + Sobel + L1 | Compound | `L1 + 0.5*HaarL1 + 0.3*Sobel` | TBD | TBD | - | RUNNING / QUEUED |
| - | Exp 16: Db2 Wavelet + Contrast + L1 | Compound | `L1 + 0.5*Db2L1 + 0.5*LocalStd` | TBD | TBD | - | RUNNING / QUEUED |
| - | Exp 17: Db2 Wavelet + Sobel + L1 | Compound | `L1 + 0.5*Db2L1 + 0.3*Sobel` | TBD | TBD | - | RUNNING / QUEUED |
| - | Exp 18: MS-SSIM + Db2 Wavelet | Compound | `0.84*Db2L1 + 0.16*(1-MS-SSIM)` | TBD | TBD | - | RUNNING / QUEUED |
| - | Exp 20: Full Kitchen Sink | Compound | `L1 + 0.5*Db2 + 0.3*Contrast + 0.2*Sobel` | TBD | TBD | - | RUNNING / QUEUED |
| - | Exp 21: FFT + Db2 Wavelet + L1 | Compound | `L1 + 0.3*FFT + 0.5*Db2L1` | TBD | TBD | - | RUNNING / QUEUED |
