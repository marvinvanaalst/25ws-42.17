#!/bin/bash
# =============================================================================
# run_ablation_all.sh
# =============================================================================
# Runs all 15 loss ablation experiments SEQUENTIALLY on one GPU.
# Each experiment uses the same 2.5D UNet + Bilinear + Adam setup as Exp 6.
# Only the loss function changes.
#
# USAGE:
#   chmod +x run_ablation_all.sh
#   nohup bash run_ablation_all.sh > ablation_master.log 2>&1 &
#
# Each experiment logs to: logs/ablation/<loss_name>.log
# =============================================================================

set -e

# ─── PATHS — EDIT THESE ──────────────────────────────────────────────────────
TRAIN_SCRIPT="/home/ss_students/mtp/modality_pipeline_wft_v3-main/train_loss_ablation.py"
BASE_CONFIG="/home/ss_students/mtp/modality_pipeline_wft_v3-main/configs/work2/exp6_2.5d_unet_l1_paper_params.json"
LOG_DIR="/home/ss_students/mtp/modality_pipeline_wft_v3-main/logs/ablation"
PYTHON="/home/ss_students/miniconda3/envs/mht/bin/python"
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$LOG_DIR"

# Experiment list: (loss_name, human label)
declare -a EXPERIMENTS=(
    "exp09_ssim_only|SSIM only"
    "exp10_msssim_l1|MS-SSIM + L1"
    "exp11_sobel_l1|Sobel Gradient + L1"
    "exp12_db2_l1|Daubechies db2 + L1"
    "exp13_contrast_l1|Local Contrast + L1"
    "exp14_fft_l1|FFT Focal Freq + L1"
    "exp15_haar_sobel_l1|Haar Wavelet + Sobel + L1"
    "exp16_db2_contrast_l1|Db2 + Contrast + L1"
    "exp17_db2_sobel_l1|Db2 + Sobel + L1"
    "exp18_msssim_db2|MS-SSIM + Db2 Wavelet"
    "exp19_msssim_contrast|MS-SSIM + Contrast"
    "exp20_db2_contrast_sobel_l1|Db2 + Contrast + Sobel + L1 (Kitchen Sink)"
    "exp21_fft_db2_l1|FFT + Db2 + L1"
)

TOTAL=${#EXPERIMENTS[@]}
DONE=0

echo "============================================================"
echo "  2.5D UNet Loss Ablation Study — $TOTAL experiments"
echo "  Started: $(date)"
echo "============================================================"

for entry in "${EXPERIMENTS[@]}"; do
    LOSS_NAME="${entry%%|*}"
    LABEL="${entry##*|}"

    DONE=$((DONE + 1))
    LOG_FILE="$LOG_DIR/${LOSS_NAME}.log"

    echo ""
    echo "------------------------------------------------------------"
    echo "  [$DONE/$TOTAL] $LABEL"
    echo "  Loss key : $LOSS_NAME"
    echo "  Log      : $LOG_FILE"
    echo "  Started  : $(date)"
    echo "------------------------------------------------------------"

    $PYTHON "$TRAIN_SCRIPT" \
        --config  "$BASE_CONFIG" \
        --loss    "$LOSS_NAME" \
        --out_tag "$LOSS_NAME" \
        2>&1 | tee "$LOG_FILE"

    echo "  Finished : $(date)"
done

echo ""
echo "============================================================"
echo "  All $TOTAL experiments done. $(date)"
echo "  Now collecting results..."
echo "============================================================"

# Print quick results summary from log files
echo ""
echo "LOSS NAME                              | PSNR (mean) | SSIM (mean)"
echo "--------------------------------------------------------------------"

# Baseline for comparison
echo "exp06_l1_baseline (DONE)              |  32.61 dB   |  0.910"
echo "exp07_haar_wavelet (DONE)             |  30.80 dB   |  0.900"

for entry in "${EXPERIMENTS[@]}"; do
    LOSS_NAME="${entry%%|*}"
    LOG_FILE="$LOG_DIR/${LOSS_NAME}.log"
    if [ -f "$LOG_FILE" ]; then
        # Extract final PSNR and SSIM from log
        PSNR=$(grep -oP "mean_psnr=\K[0-9.]+" "$LOG_FILE" | tail -1)
        SSIM=$(grep -oP "mean_ssim=\K[0-9.]+" "$LOG_FILE" | tail -1)
        printf "%-38s | %s dB | %s\n" "$LOSS_NAME" "${PSNR:-N/A}" "${SSIM:-N/A}"
    else
        printf "%-38s | NO LOG FOUND\n" "$LOSS_NAME"
    fi
done

echo "--------------------------------------------------------------------"
echo "Done. Run collect_ablation_results.py for full per-fold breakdown."
