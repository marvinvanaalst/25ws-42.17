#!/usr/bin/env python3
"""
collect_ablation_results.py
===========================
Parses all loss ablation logs and metrics CSV files, compiles a comprehensive
comparison against the baselines (Exp 6 L1: 32.61 dB and Exp 7 Haar: 30.80 dB),
and generates both a Markdown table and CSV report ready for supervisor presentation.

Usage:
    python3 collect_ablation_results.py
    python3 collect_ablation_results.py --log_dir logs/ablation --out_dir results
"""

import argparse
import csv
import glob
import os
import re
import numpy as np


EXPERIMENT_METADATA = {
    "exp06_l1_baseline": {
        "label": "Exp 6: L1 Baseline",
        "loss_formula": "L1",
        "category": "Baseline",
        "psnr": 32.61,
        "ssim": 0.910
    },
    "exp07_haar_wavelet": {
        "label": "Exp 7: Haar Wavelet (WFT)",
        "loss_formula": "Haar Wavelet L1 (LL + 2*Det)",
        "category": "Wavelet",
        "psnr": 30.80,
        "ssim": 0.900
    },
    "exp09_ssim_only": {
        "label": "Exp 9: SSIM Loss",
        "loss_formula": "1 - SSIM",
        "category": "Structural"
    },
    "exp10_msssim_l1": {
        "label": "Exp 10: MS-SSIM + L1",
        "loss_formula": "0.84*L1 + 0.16*(1-MS-SSIM)",
        "category": "Multi-Scale"
    },
    "exp11_sobel_l1": {
        "label": "Exp 11: Sobel Gradient + L1",
        "loss_formula": "L1 + 0.5*SobelGrad",
        "category": "Edge Sharpness"
    },
    "exp12_db2_l1": {
        "label": "Exp 12: Daubechies db2 Wavelet + L1",
        "loss_formula": "L1 + 0.5*Db2WaveletL1",
        "category": "Wavelet (Novel)"
    },
    "exp13_contrast_l1": {
        "label": "Exp 13: Local Contrast + L1",
        "loss_formula": "L1 + 0.5*LocalStdL1",
        "category": "Tissue Contrast"
    },
    "exp14_fft_l1": {
        "label": "Exp 14: FFT Focal Frequency + L1",
        "loss_formula": "L1 + 0.3*FocalFreq",
        "category": "Frequency Spectrum"
    },
    "exp15_haar_sobel_l1": {
        "label": "Exp 15: Haar Wavelet + Sobel + L1",
        "loss_formula": "L1 + 0.5*HaarL1 + 0.3*Sobel",
        "category": "Compound"
    },
    "exp16_db2_contrast_l1": {
        "label": "Exp 16: Db2 Wavelet + Contrast + L1",
        "loss_formula": "L1 + 0.5*Db2L1 + 0.5*LocalStd",
        "category": "Compound"
    },
    "exp17_db2_sobel_l1": {
        "label": "Exp 17: Db2 Wavelet + Sobel + L1",
        "loss_formula": "L1 + 0.5*Db2L1 + 0.3*Sobel",
        "category": "Compound"
    },
    "exp18_msssim_db2": {
        "label": "Exp 18: MS-SSIM + Db2 Wavelet",
        "loss_formula": "0.84*Db2L1 + 0.16*(1-MS-SSIM)",
        "category": "Compound"
    },
    "exp19_msssim_contrast": {
        "label": "Exp 19: MS-SSIM + Contrast",
        "loss_formula": "0.84*L1 + 0.08*(1-MSSSIM) + 0.08*Contrast",
        "category": "Compound"
    },
    "exp20_db2_contrast_sobel_l1": {
        "label": "Exp 20: Full Kitchen Sink",
        "loss_formula": "L1 + 0.5*Db2 + 0.3*Contrast + 0.2*Sobel",
        "category": "Compound"
    },
    "exp21_fft_db2_l1": {
        "label": "Exp 21: FFT + Db2 Wavelet + L1",
        "loss_formula": "L1 + 0.3*FFT + 0.5*Db2L1",
        "category": "Compound"
    },
}


def parse_log_file(log_path):
    psnr, ssim = None, None
    if not os.path.exists(log_path):
        return None, None
    with open(log_path, 'r', errors='ignore') as f:
        content = f.read()
    psnr_matches = re.findall(r"mean_psnr=([0-9.]+)", content)
    if psnr_matches:
        psnr = float(psnr_matches[-1])
    ssim_matches = re.findall(r"mean_ssim=([0-9.]+)", content)
    if ssim_matches:
        ssim = float(ssim_matches[-1])
    return psnr, ssim


def parse_csv_file(csv_path):
    if not os.path.exists(csv_path):
        return None, None, None, None
    psnrs, ssims = [], []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'test_psnr' in row and row['test_psnr']:
                psnrs.append(float(row['test_psnr']))
            if 'test_ssim_3D' in row and row['test_ssim_3D']:
                ssims.append(float(row['test_ssim_3D']))
    if not psnrs:
        return None, None, None, None
    return float(np.mean(psnrs)), float(np.std(psnrs)), float(np.mean(ssims)), float(np.std(ssims))


def main():
    parser = argparse.ArgumentParser(description="Collect and compare 2.5D UNet loss ablation results")
    parser.add_argument("--log_dir", default="logs/ablation", help="Path to ablation logs directory")
    parser.add_argument("--out_dir", default=".", help="Directory to save summary tables")
    args = parser.parse_args()

    results = []

    # 1. Add baselines
    b_l1 = EXPERIMENT_METADATA["exp06_l1_baseline"]
    results.append({
        "key": "exp06_l1_baseline",
        "label": b_l1["label"],
        "loss_formula": b_l1["loss_formula"],
        "category": b_l1["category"],
        "psnr": b_l1["psnr"],
        "psnr_std": 0.0,
        "ssim": b_l1["ssim"],
        "ssim_std": 0.0,
        "status": "COMPLETED (Baseline)"
    })

    b_haar = EXPERIMENT_METADATA["exp07_haar_wavelet"]
    results.append({
        "key": "exp07_haar_wavelet",
        "label": b_haar["label"],
        "loss_formula": b_haar["loss_formula"],
        "category": b_haar["category"],
        "psnr": b_haar["psnr"],
        "psnr_std": 0.0,
        "ssim": b_haar["ssim"],
        "ssim_std": 0.0,
        "status": "COMPLETED (Exp 7)"
    })

    # 2. Iterate through all remaining experiments
    for key, meta in EXPERIMENT_METADATA.items():
        if key in ("exp06_l1_baseline", "exp07_haar_wavelet"):
            continue

        log_path = os.path.join(args.log_dir, f"{key}.log")
        csv_path = f"logs/{key}/metrics/average_testing_.csv"

        psnr_mean, psnr_std, ssim_mean, ssim_std = parse_csv_file(csv_path)

        if psnr_mean is None:
            # Fall back to log file
            l_psnr, l_ssim = parse_log_file(log_path)
            if l_psnr is not None:
                psnr_mean, psnr_std = l_psnr, 0.0
                ssim_mean, ssim_std = l_ssim if l_ssim else 0.0, 0.0

        status = "COMPLETED" if psnr_mean is not None else ("RUNNING / QUEUED" if os.path.exists(log_path) else "NOT STARTED")

        results.append({
            "key": key,
            "label": meta["label"],
            "loss_formula": meta["loss_formula"],
            "category": meta["category"],
            "psnr": psnr_mean,
            "psnr_std": psnr_std if psnr_std is not None else 0.0,
            "ssim": ssim_mean,
            "ssim_std": ssim_std if ssim_std is not None else 0.0,
            "status": status
        })

    # Sort results: completed by descending PSNR, then uncompleted
    completed = [r for r in results if r["psnr"] is not None]
    uncompleted = [r for r in results if r["psnr"] is None]
    completed.sort(key=lambda x: x["psnr"], reverse=True)
    sorted_results = completed + uncompleted

    baseline_psnr = 32.61

    # Print Table
    print("\n" + "=" * 105)
    print(f"{'2.5D UNet Loss Ablation Results (3T -> 7T MRI Synthesis)':^105}")
    print("=" * 105)
    header = f"{'Rank':<5} | {'Experiment Label':<32} | {'Category':<15} | {'PSNR (dB)':<12} | {'SSIM':<8} | {'Δ vs L1':<10} | {'Status'}"
    print(header)
    print("-" * 105)

    rank = 1
    for r in sorted_results:
        if r["psnr"] is not None:
            delta = r["psnr"] - baseline_psnr
            delta_str = f"{delta:+.2f} dB" if r["key"] != "exp06_l1_baseline" else "BASELINE"
            psnr_str = f"{r['psnr']:.2f} dB"
            ssim_str = f"{r['ssim']:.4f}" if r["ssim"] else "N/A"
            rank_str = f"#{rank}"
            rank += 1
        else:
            delta_str = "-"
            psnr_str = "N/A"
            ssim_str = "N/A"
            rank_str = "-"

        print(f"{rank_str:<5} | {r['label']:<32} | {r['category']:<15} | {psnr_str:<12} | {ssim_str:<8} | {delta_str:<10} | {r['status']}")

    print("=" * 105 + "\n")

    # Save to Markdown
    md_path = os.path.join(args.out_dir, "ablation_results_summary.md")
    with open(md_path, 'w') as f:
        f.write("# 2.5D UNet Loss Ablation Study Results\n\n")
        f.write("**Model Architecture:** 2.5D UNet (5 adjacent axial slices input, 1 center target slice output)\n")
        f.write("**Upsampling:** Bilinear\n")
        f.write("**Optimizer:** Adam (lr=1e-4, betas=(0.9, 0.999))\n\n")
        f.write("| Rank | Experiment | Category | Loss Formula | PSNR (mean ± std) | SSIM | Δ vs Exp 6 (L1) | Status |\n")
        f.write("|:----:|:-----------|:---------|:-------------|:------------------|:----:|:---------------:|:------:|\n")

        rank = 1
        for r in sorted_results:
            if r["psnr"] is not None:
                delta = r["psnr"] - baseline_psnr
                delta_str = f"**{delta:+.2f} dB**" if r["key"] != "exp06_l1_baseline" else "*Baseline*"
                psnr_str = f"**{r['psnr']:.2f}** ± {r['psnr_std']:.2f}" if r['psnr_std'] > 0 else f"**{r['psnr']:.2f}**"
                ssim_str = f"{r['ssim']:.4f}" if r["ssim"] else "N/A"
                rank_str = f"{rank}"
                rank += 1
            else:
                delta_str = "-"
                psnr_str = "TBD"
                ssim_str = "TBD"
                rank_str = "-"
            f.write(f"| {rank_str} | {r['label']} | {r['category']} | `{r['loss_formula']}` | {psnr_str} | {ssim_str} | {delta_str} | {r['status']} |\n")

    print(f" Saved markdown report to: {md_path}")

    # Save to CSV
    csv_path = os.path.join(args.out_dir, "ablation_results_summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Key", "Label", "Category", "LossFormula", "PSNR", "PSNR_Std", "SSIM", "SSIM_Std", "Delta_vs_L1", "Status"])
        rank = 1
        for r in sorted_results:
            if r["psnr"] is not None:
                delta = r["psnr"] - baseline_psnr
                delta_str = f"{delta:+.2f}" if r["key"] != "exp06_l1_baseline" else "0.0"
                rank_str = str(rank)
                rank += 1
            else:
                delta_str = ""
                rank_str = ""
            writer.writerow([rank_str, r["key"], r["label"], r["category"], r["loss_formula"], r["psnr"] or "", r["psnr_std"], r["ssim"] or "", r["ssim_std"], delta_str, r["status"]])
    print(f" Saved CSV report to: {csv_path}")


if __name__ == "__main__":
    main()
