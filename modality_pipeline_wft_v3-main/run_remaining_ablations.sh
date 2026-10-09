#!/bin/bash
# =============================================================================
# run_remaining_ablations.sh
# =============================================================================
# Runs ONLY the 8 remaining loss ablation experiments that failed due to the
# AMP dtype mismatch (now fully fixed in losses_library.py).
#
# USAGE:
#   cd /home/ss_students/mtp/modality_pipeline_wft_v3-main
#   bash run_remaining_ablations.sh 2>&1 | tee logs/ablation_remaining_master.log
#
# Or inside screen:
#   screen -S ablation_remaining
#   bash run_remaining_ablations.sh 2>&1 | tee logs/ablation_remaining_master.log
#   Ctrl+A, D
# =============================================================================

set -e

TRAIN_SCRIPT="/home/ss_students/mtp/modality_pipeline_wft_v3-main/train_loss_ablation.py"
BASE_CONFIG="/home/ss_students/mtp/modality_pipeline_wft_v3-main/configs/work2/exp6_2.5d_unet_l1_paper_params.json"
LOG_DIR="/home/ss_students/mtp/modality_pipeline_wft_v3-main/logs/ablation"
PYTHON="/home/ss_students/miniconda3/envs/mht/bin/python"

mkdir -p "$LOG_DIR"

declare -a REMAINING_EXPERIMENTS=(
    "exp11_sobel_l1|Sobel Gradient + L1"
    "exp12_db2_l1|Daubechies db2 + L1"
    "exp15_haar_sobel_l1|Haar Wavelet + Sobel + L1"
    "exp16_db2_contrast_l1|Db2 + Contrast + L1"
    "exp17_db2_sobel_l1|Db2 + Sobel + L1"
    "exp18_msssim_db2|MS-SSIM + Db2 Wavelet"
    "exp20_db2_contrast_sobel_l1|Db2 + Contrast + Sobel + L1 (Kitchen Sink)"
    "exp21_fft_db2_l1|FFT + Db2 + L1"
)

TOTAL=${#REMAINING_EXPERIMENTS[@]}
DONE=0

echo "============================================================"
echo "  Remaining 2.5D UNet Loss Ablations ($TOTAL experiments)"
echo "  Started: $(date)"
echo "============================================================"

for entry in "${REMAINING_EXPERIMENTS[@]}"; do
    LOSS_NAME="${entry%%|*}"
    LABEL="${entry##*|}"

    DONE=$((DONE + 1))
    LOG_FILE="$LOG_DIR/${LOSS_NAME}.log"

    echo ""
    echo "------------------------------------------------------------"
    echo "  [$DONE / $TOTAL] Running: $LABEL ($LOSS_NAME)"
    echo "  Log: $LOG_FILE"
    echo "  Started at: $(date)"
    echo "------------------------------------------------------------"

    START_SEC=$SECONDS

    set +e
    $PYTHON "$TRAIN_SCRIPT" \
        --config "$BASE_CONFIG" \
        --loss "$LOSS_NAME" \
        --out_tag "$LOSS_NAME" \
        > "$LOG_FILE" 2>&1
    EXIT_CODE=$?
    set -e

    ELAPSED=$(( SECONDS - START_SEC ))
    MINS=$(( ELAPSED / 60 ))
    SECS=$(( ELAPSED % 60 ))

    if [ $EXIT_CODE -eq 0 ]; then
        echo "  [OK] $LOSS_NAME completed in ${MINS}m ${SECS}s."
    else
        echo "  [FAIL] $LOSS_NAME exited with code $EXIT_CODE after ${MINS}m ${SECS}s."
        echo "  Check log: $LOG_FILE"
        echo "  Last 20 lines:"
        tail -n 20 "$LOG_FILE" | sed 's/^/    /'
    fi
done

echo ""
echo "============================================================"
echo "  All remaining ablation experiments completed at: $(date)"
echo "============================================================"
