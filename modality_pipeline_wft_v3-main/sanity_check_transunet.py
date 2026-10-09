#!/usr/bin/env python3
"""
sanity_check_transunet.py
Run this BEFORE launching training to confirm the model works correctly.

Usage:
    cd /home/ss_students/mtp/modality_pipeline_wft_v3-main
    /home/ss_students/miniconda3/envs/mht/bin/python sanity_check_transunet.py
"""
import sys
import torch

print("="*60)
print("TransUNet25D Sanity Check")
print("="*60)

# -- 1. Import ----------------------------------------------------------------
try:
    from pl_models.models.TransUNet.transunet_25d import TransUNet25D
    print("[OK] Import successful")
except Exception as e:
    print(f"[FAIL] Import error: {e}")
    sys.exit(1)

# -- 2. CUDA ------------------------------------------------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[OK] Device: {device}")
if device == "cpu":
    print("[WARN] No CUDA — running on CPU (will be very slow for training!)")

# -- 3. Instantiate (no pretrained) -------------------------------------------
try:
    model = TransUNet25D(n_channels=5, img_size=224, pretrained_path=None)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model instantiated. Params: {total_params:,} ({total_params/1e6:.1f}M)")
except Exception as e:
    print(f"[FAIL] Model instantiation: {e}")
    sys.exit(1)

# -- 4. Forward pass with 256x256 input (real pipeline size) ------------------
try:
    model.eval()
    with torch.no_grad():
        x = torch.randn(2, 5, 256, 256, device=device)
        out = model(x)
    print(f"[OK] Forward pass: input {tuple(x.shape)} -> output {tuple(out.shape)}")
    assert out.shape == (2, 1, 256, 256), f"Wrong shape: {out.shape}"
    assert out.min().item() >= -1.01 and out.max().item() <= 1.01, \
        f"Output range wrong: [{out.min():.4f}, {out.max():.4f}]"
    print(f"[OK] Output range: [{out.min().item():.4f}, {out.max().item():.4f}]  (expected ~[-1,1])")
except Exception as e:
    print(f"[FAIL] Forward pass: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# -- 5. Backward pass ---------------------------------------------------------
try:
    model.train()
    x  = torch.randn(2, 5, 256, 256, device=device, requires_grad=False)
    tgt = torch.randn(2, 1, 256, 256, device=device)
    out = model(x)
    # pipeline normalises target to [0,1]; model outputs [-1,1]
    # master script converts via to_model_range/from_model_range — simulate here
    tgt_conv = tgt * 2.0 - 1.0   # [0,1] -> [-1,1]
    loss = torch.nn.L1Loss()(out, tgt_conv)
    loss.backward()
    print(f"[OK] Backward pass. L1 loss = {loss.item():.4f}")
except Exception as e:
    print(f"[FAIL] Backward pass: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# -- 6. GPU memory ------------------------------------------------------------
if device.startswith("cuda"):
    mem_used = torch.cuda.memory_allocated() / 1e9
    mem_res  = torch.cuda.memory_reserved()  / 1e9
    print(f"[OK] GPU memory: {mem_used:.2f} GB allocated, {mem_res:.2f} GB reserved")
    if mem_res > 20:
        print("[WARN] Memory usage is high — consider reducing batch_size to 8")

print("="*60)
print("ALL CHECKS PASSED — TransUNet25D verified and ready.")
print("="*60)
