#!/bin/bash
# run_transunet.sh — Run Exp 9a (SGD) then 9b (Adam) sequentially.
# Usage:
#   cd /home/ss_students/mtp/modality_pipeline_wft_v3-main
#   bash run_transunet.sh 2>&1 | tee logs/transunet_master.log
#
# Or inside a screen session for persistence:
#   screen -S transunet
#   bash run_transunet.sh 2>&1 | tee logs/transunet_master.log
#   Ctrl+A, D   <- detach

PYTHON=/home/ss_students/miniconda3/envs/mht/bin/python
BASE=/home/ss_students/mtp/modality_pipeline_wft_v3-main
MASTER="$BASE/master_half_slice_re.py"

# Verify Python + GPU
echo "======================================================"
echo "  TransUNet Experiments 9a (SGD) and 9b (Adam)"
echo "  $(date)"
echo "======================================================"
$PYTHON -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# ---- Exp 9a: TransUNet + SGD + Polynomial LR Decay ----
echo ""
echo "======================================================"
echo "  STARTING Exp 9a: TransUNet SGD  ($(date))"
echo "======================================================"
$PYTHON "$MASTER" "$BASE/configs/ablation_runtime/exp9a_transunet_sgd.json"
EC=$?
echo "Exp 9a finished with exit code $EC  ($(date))"

# ---- Exp 9b: TransUNet + Adam + Linear Warmup ----
echo ""
echo "======================================================"
echo "  STARTING Exp 9b: TransUNet Adam  ($(date))"
echo "======================================================"
$PYTHON "$MASTER" "$BASE/configs/ablation_runtime/exp9b_transunet_adam.json"
EC=$?
echo "Exp 9b finished with exit code $EC  ($(date))"

echo ""
echo "======================================================"
echo "  BOTH EXPERIMENTS COMPLETE  ($(date))"
echo "======================================================"
