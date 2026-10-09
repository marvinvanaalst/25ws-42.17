"""
model_benchmark.py
------------------
Comprehensive efficiency benchmarking for PyTorch models in FP16.

Metrics:
    Parameters    : total, trainable, non-trainable, param size (MB)
    Efficiency    : params/GFlop, params/GMAC, bits per param
    Compute       : GFLOPs, GMACs per single sample (fvcore)
    Latency FP16  : first-inference, mean ± std, min, max (ms)  [CUDA Events]
    Throughput    : samples / second
    Memory        : peak allocated delta (MB), memory per param (bytes)
    CPU latency   : FP32 baseline on CPU
    GPU util      : snapshot % via pynvml

Install:
    pip install torch fvcore tabulate pynvml
"""

import gc
import os
import time
import tempfile
import traceback
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from tabulate import tabulate

try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False
    print("[WARN] fvcore not found — FLOPs/MACs skipped.  pip install fvcore")

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_PYNVML = True
except Exception:
    HAS_PYNVML = False
    print("[WARN] pynvml not found — GPU utilisation skipped.  pip install pynvml")


# ══════════════════════════════════════════════════════════════════════════════
#  ▶▶  HARDCODED BENCHMARK CONFIGURATION  ◀◀
# ══════════════════════════════════════════════════════════════════════════════
from pl_models.models.ESAU_3D_transformer_like import ESAU_3D as ESAU_3D
from pl_models.models.MONAI.SwinUNETR import fetchSwinUNETR
from pl_models.models.VNET.VNet import VNet
from pl_models.models.MONAI.UNETR import fetch_UNETR


input_shape = (16,1,64,64,64)
BENCHMARK_CONFIGS: List[Dict[str, Any]] = [
    {
        "log_name":     "3DUNET_CONV",
        "model_class":  ESAU_3D,
        "model_args":   {
        "in_channels":1,
        "out_channels":1,
        "n_channels":32,
        "num_heads":[1,2,4,8],
        "res":True,
        "activation":True,
        "interpolation":"conv"
    },
        "input_shape":  input_shape,
        "weights_path": None,
    },
    {
        "log_name":     "3DUNET_NEAREST",
        "model_class":  ESAU_3D,
        "model_args":   {
        "in_channels":1,
        "out_channels":1,
        "n_channels":32,
        "num_heads":[1,2,4,8],
        "res":True,
        "activation":True,
        "interpolation":"nearest"
    },
        "input_shape": input_shape,
        "weights_path": None,
    },
        {
        "log_name":     "3DUNET_TRILINEAR",
        "model_class":  ESAU_3D,
        "model_args":   {
        "in_channels":1,
        "out_channels":1,
        "n_channels":32,
        "num_heads":[1,2,4,8],
        "res":True,
        "activation":True,
        "interpolation":"trilinear"
    },
        "input_shape":  input_shape,
        "weights_path": None,
    },
    {
        "log_name":     "SWINUENTR_CONV",
        "model_class":  fetchSwinUNETR,
        "model_args":   {
        "in_channels": 1,
        "out_channels": 1,
        "feature_size": 12,
        "depths": [2, 2, 2, 2],
        "num_heads": [3, 6, 12, 24],
        "norm_name": "instance",
        "spatial_dims": 3,
        "interpolation": None
    },
        "input_shape":  input_shape,
        "weights_path": None,
    },
    {
        "log_name":     "SWINUENTR_others",
        "model_class":  fetchSwinUNETR,
        "model_args":   {
        "in_channels": 1,
        "out_channels": 1,
        "feature_size": 12,
        "depths": [2, 2, 2, 2],
        "num_heads": [3, 6, 12, 24],
        "norm_name": "instance",
        "spatial_dims": 3,
        "interpolation": "nearest"
    },
        "input_shape":  input_shape,
        "weights_path": None,
    },
    {
        "log_name":     "VNet",
        "model_class":  VNet,
        "model_args":   {
    },
        "input_shape":  input_shape,
        "weights_path": None,
    },
    
        {
        "log_name":     "UNETR",
        "model_class":  fetch_UNETR,
        "model_args":   {        "in_channels": 1,
        "out_channels": 2,
        "img_size": [64, 64, 64],
        "feature_size": 16,
        "hidden_size": 768,
        "mlp_dim": 3072,
        "num_heads": 12,
        "pos_embed": "perceptron",
        "norm_name": "instance",
        "res_block": True,
        "dropout_rate": 0.0
    },
        "input_shape":  input_shape,
        "weights_path": None,
    },             
]

# ══════════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

N_WARMUP         = 10
N_RUNS           = 20     # run 20 batches, average the middle 4
DEVICE           = "cuda:3" if torch.cuda.is_available() else "cpu"
SEED             = 42
RUN_CPU_COMPARE  = False
SAVE_RESULTS_CSV = True
RESULTS_DIR      = "performacne_comparision"

FP16_BYTES       = 2      # torch.float16 occupies 2 bytes per element


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_class: Type[nn.Module], model_args: Dict) -> nn.Module:
    return model_class(**model_args)


def load_weights(model: nn.Module, weights_path: Optional[str]) -> nn.Module:
    if weights_path is None:
        return model
    state = torch.load(weights_path, map_location="cpu")
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    model.load_state_dict(state, strict=True)
    print(f"    Loaded weights: {weights_path}")
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int, int]:
    total      = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen     = total - trainable
    return total, trainable, frozen


def model_disk_size_mb(model: nn.Module) -> float:
    """Serialise FP32 state-dict to a temp file; return size in MB."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / (1024 ** 2)
    os.remove(path)
    return size


def compute_flops(model: nn.Module, dummy_fp32: torch.Tensor) -> Tuple[Optional[float], Optional[float]]:
    """Returns (GFLOPs, GMACs) for a single sample. Requires fvcore."""
    if not HAS_FVCORE:
        return None, None
    try:
        m      = deepcopy(model).cpu().float().eval()
        single = dummy_fp32[:1].cpu().float()
        flops  = FlopCountAnalysis(m, single)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        total  = flops.total()
        del m
        return total / 1e9, total / 2 / 1e9
    except Exception as exc:
        print(f"    [WARN] FLOPs failed: {exc}")
        return None, None


def get_gpu_util() -> Optional[float]:
    if not HAS_PYNVML:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    except Exception:
        return None


def timed_inference_fp16(
    model:    nn.Module,
    dummy:    torch.Tensor,
    device:   str,
    n_warmup: int,
    n_runs:   int,
) -> Dict[str, float]:
    """
    FP16 inference on GPU using CUDA Events for precise timing.
    Model and input are cast to float16 before any run.

    Returns:
        first_ms, mean_ms, std_ms, min_ms, max_ms,
        throughput_sps, peak_mem_delta_mb
    """
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    model    = model.to(device).half().eval()
    dummy    = dummy.to(device).half()

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        mem_before = torch.cuda.memory_allocated(device) / (1024 ** 2)

    # first-inference latency
    if use_cuda:
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(dummy)
    if use_cuda:
        torch.cuda.synchronize(device)
    first_ms = (time.perf_counter() - t0) * 1e3

    # warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)
    if use_cuda:
        torch.cuda.synchronize(device)

    # timed runs
    if use_cuda:
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]
        with torch.no_grad():
            for i in range(n_runs):
                starts[i].record()
                _ = model(dummy)
                ends[i].record()
        torch.cuda.synchronize(device)
        latencies = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    else:
        latencies = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = model(dummy)
                latencies.append((time.perf_counter() - t0) * 1e3)

    if use_cuda:
        mem_delta = torch.cuda.max_memory_allocated(device) / (1024 ** 2) - mem_before
    else:
        mem_delta = 0.0

    sorted_lat = sorted(latencies)
    n          = len(sorted_lat)
    mid        = n // 2
    middle4    = sorted_lat[mid - 2 : mid + 2]   # 4 central values after sorting
    mean_ms    = float(np.mean(middle4))
    return {
        "first_ms":          first_ms,
        "mean_ms":           mean_ms,                         # average of middle 4
        "std_ms":            float(np.std(middle4)),
        "min_ms":            float(np.min(latencies)),        # global min across all 20
        "max_ms":            float(np.max(latencies)),        # global max across all 20
        "throughput_sps":    dummy.shape[0] * 1000.0 / mean_ms,
        "peak_mem_delta_mb": mem_delta,
    }


def timed_inference_cpu_fp32(
    model:    nn.Module,
    dummy:    torch.Tensor,
    n_warmup: int,
    n_runs:   int,
) -> Dict[str, float]:
    """FP32 CPU baseline using perf_counter."""
    model = model.cpu().float().eval()
    dummy = dummy.cpu().float()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            latencies.append((time.perf_counter() - t0) * 1e3)
    sorted_lat = sorted(latencies)
    n          = len(sorted_lat)
    mid        = n // 2
    middle4    = sorted_lat[mid - 2 : mid + 2]
    mean_ms    = float(np.mean(middle4))
    return {
        "first_ms":       latencies[0],
        "mean_ms":        mean_ms,                        # average of middle 4
        "std_ms":         float(np.std(middle4)),
        "min_ms":         float(np.min(latencies)),
        "max_ms":         float(np.max(latencies)),
        "throughput_sps": dummy.shape[0] * 1000.0 / mean_ms,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CORE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_one(cfg: Dict[str, Any]) -> Dict[str, Any]:
    log_name     = cfg["log_name"]
    model_class  = cfg["model_class"]
    model_args   = cfg["model_args"]
    input_shape  = cfg["input_shape"]
    weights_path = cfg.get("weights_path", None)

    print(f"\n{'═'*64}\n  {log_name}\n{'═'*64}")
    set_seed(SEED)
    result: Dict[str, Any] = {"log_name": log_name}

    # ── build & load ──────────────────────────────────────────────────────────
    model = build_model(model_class, model_args)
    model = load_weights(model, weights_path)
    model.eval()

    # ── parameter efficiency metrics ──────────────────────────────────────────
    total_p, train_p, frozen_p = count_parameters(model)
    fp32_param_mb  = (total_p * 4)  / (1024 ** 2)   # 4 bytes  per FP32 param
    fp16_param_mb  = (total_p * 2)  / (1024 ** 2)   # 2 bytes  per FP16 param
    disk_mb        = model_disk_size_mb(model)

    result["total_params"]      = total_p
    result["trainable_params"]  = train_p
    result["frozen_params"]     = frozen_p
    result["trainable_ratio"]   = train_p / total_p if total_p > 0 else 0.0
    result["fp32_param_mb"]     = fp32_param_mb
    result["fp16_param_mb"]     = fp16_param_mb
    result["disk_mb"]           = disk_mb
    result["compression_ratio"] = disk_mb / fp32_param_mb if fp32_param_mb > 0 else None

    print(f"  Total params      : {total_p:,}")
    print(f"  Trainable params  : {train_p:,}  ({100*train_p/total_p:.1f}%)")
    print(f"  Frozen params     : {frozen_p:,}")
    print(f"  FP32 param mem    : {fp32_param_mb:.3f} MB")
    print(f"  FP16 param mem    : {fp16_param_mb:.3f} MB")
    print(f"  Disk size (FP32)  : {disk_mb:.3f} MB")

    # ── random FP32 input (fvcore needs FP32; inference will cast to FP16) ────
    set_seed(SEED)
    dummy_fp32 = torch.randn(*input_shape, dtype=torch.float32)
    print(f"  Input shape       : {list(input_shape)}")

    # ── FLOPs / MACs ──────────────────────────────────────────────────────────
    gflops, gmacs = compute_flops(model, dummy_fp32)
    result["gflops_per_sample"] = gflops
    result["gmacs_per_sample"]  = gmacs

    # params / GFlop and params / GMAC — higher = more compute-efficient use of params
    result["params_per_gflop"] = (total_p / gflops)  if gflops else None
    result["params_per_gmac"]  = (total_p / gmacs)   if gmacs  else None
    # GFlops / param — how much compute each parameter contributes
    result["gflops_per_param"] = (gflops / total_p)  if gflops else None

    if gflops is not None:
        print(f"  GFLOPs / sample   : {gflops:.5f}")
        print(f"  GMACs  / sample   : {gmacs:.5f}")
        print(f"  Params / GFlop    : {result['params_per_gflop']:.1f}")
        print(f"  GFlops / param    : {result['gflops_per_param']:.6f}")

    # ── FP16 inference on primary device ──────────────────────────────────────
    print(f"  ▸ FP16 on {DEVICE.upper()} ...")
    s = timed_inference_fp16(model, dummy_fp32, DEVICE, N_WARMUP, N_RUNS)
    result.update({f"fp16_{k}": v for k, v in s.items()})

    # memory efficiency: MB of peak activation memory per million params
    result["fp16_mem_per_mparam"] = (
        s["peak_mem_delta_mb"] / (total_p / 1e6) if total_p > 0 else None
    )

    print(f"    1st latency       : {s['first_ms']:.2f} ms")
    print(f"    Mean ± Std        : {s['mean_ms']:.2f} ± {s['std_ms']:.2f} ms")
    print(f"    Min / Max         : {s['min_ms']:.2f} / {s['max_ms']:.2f} ms")
    print(f"    Throughput        : {s['throughput_sps']:.1f} samples/sec")
    print(f"    Peak ΔMem         : {s['peak_mem_delta_mb']:.2f} MB")
    if result["fp16_mem_per_mparam"] is not None:
        print(f"    ΔMem / M-param    : {result['fp16_mem_per_mparam']:.3f} MB")

    # ── GPU utilisation snapshot ──────────────────────────────────────────────
    result["gpu_util_pct"] = get_gpu_util()
    if result["gpu_util_pct"] is not None:
        print(f"    GPU util          : {result['gpu_util_pct']:.1f}%")

    # ── derived efficiency metrics ────────────────────────────────────────────
    mean_ms  = s["mean_ms"]
    mean_sec = mean_ms / 1e3
    batch    = input_shape[0]

    # throughput-normalised latency: ms per sample (not per batch)
    result["ms_per_sample"] = mean_ms / batch

    # roofline proxy: observed TFLOP/s (FP16 theoretical peak on A100 = 312 TFLOP/s)
    # formula: (batch * GFLOPs_per_sample) / mean_sec  → GFLOPs/sec → divide by 1000 for TFLOP/s
    if gflops is not None:
        observed_tflops = (batch * gflops) / mean_sec / 1e3
        result["observed_tflops"]        = observed_tflops
        # arithmetic intensity: FLOPs / bytes moved  (FP16 params + activations rough estimate)
        bytes_moved                      = fp16_param_mb * (1024 ** 2)   # lower bound: weight traffic
        result["arithmetic_intensity"]   = (batch * gflops * 1e9) / bytes_moved if bytes_moved > 0 else None
        # compute efficiency: observed TFLOP/s / theoretical peak (A100 FP16 = 312 TFLOP/s)
        result["compute_efficiency_pct"] = (observed_tflops / 312.0) * 100
        print(f"    Observed TFLOP/s  : {observed_tflops:.4f}")
        print(f"    Arith intensity   : {result['arithmetic_intensity']:.2f} FLOP/byte")
        print(f"    Compute eff (A100): {result['compute_efficiency_pct']:.2f}%")
    else:
        result["observed_tflops"]        = None
        result["arithmetic_intensity"]   = None
        result["compute_efficiency_pct"] = None

    # memory bandwidth utilisation: GB/s of weight reads during inference
    # lower bound — weights read once per forward pass per sample in batch
    weight_bytes_gb = fp16_param_mb / 1024
    result["mem_bandwidth_gb_s"] = (batch * weight_bytes_gb) / mean_sec

    # latency per GFlop: how many ms does 1 GFlop cost on this model/hardware
    result["ms_per_gflop"] = mean_ms / (batch * gflops) if gflops else None

    # parameter utilisation: throughput (samples/s) per million parameters
    result["sps_per_mparam"] = s["throughput_sps"] / (total_p / 1e6) if total_p > 0 else None

    # memory efficiency: throughput per MB of FP16 parameter footprint
    result["sps_per_fp16_mb"] = s["throughput_sps"] / fp16_param_mb if fp16_param_mb > 0 else None

    print(f"    ms / sample       : {result['ms_per_sample']:.3f}")
    print(f"    Mem BW (lb) GB/s  : {result['mem_bandwidth_gb_s']:.3f}")
    print(f"    Thrpt / M-param   : {result['sps_per_mparam']:.2f}")
    print(f"    Thrpt / FP16-MB   : {result['sps_per_fp16_mb']:.2f}")

    del model, dummy_fp32
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY TABLE + CSV
# ══════════════════════════════════════════════════════════════════════════════

def _f(val: Any, fmt: str = ".2f") -> str:
    return "N/A" if val is None else format(val, fmt)


def print_summary(results: List[Dict[str, Any]]) -> None:
    headers = [
        "Model",
        # parameter counts
        "Params (M)", "Trainable %", "Frozen (M)",
        # memory footprint
        "FP16 Param MB", "Disk MB",
        # compute
        "GFLOPs", "GMACs",
        "Params/GFlop", "GFlops/Param",
        # FP16 inference timing
        "FP16 1st (ms)", "FP16 μ (ms)", "FP16 σ (ms)",
        "FP16 min (ms)", "FP16 max (ms)",
        "ms/sample",
        # throughput & memory
        "Thrpt (s/s)", "ΔMem (MB)", "ΔMem/Mparam",
        # roofline / efficiency
        "TFLOP/s", "Arith Int (F/B)", "Compute Eff %",
        "Mem BW lb (GB/s)", "ms/GFlop",
        # param utilisation
        "Thrpt/Mparam", "Thrpt/FP16MB",
        # gpu util
        "GPU Util (%)",
    ]
    rows = []
    for r in results:
        rows.append([
            r["log_name"],
            _f(r["total_params"] / 1e6, ".3f"),
            _f(r["trainable_ratio"] * 100, ".1f"),
            _f(r["frozen_params"] / 1e6, ".3f"),
            _f(r["fp16_param_mb"], ".3f"),
            _f(r["disk_mb"], ".3f"),
            _f(r.get("gflops_per_sample"), ".5f"),
            _f(r.get("gmacs_per_sample"),  ".5f"),
            _f(r.get("params_per_gflop"),  ".1f"),
            _f(r.get("gflops_per_param"),  ".6f"),
            _f(r.get("fp16_first_ms")),
            _f(r.get("fp16_mean_ms")),
            _f(r.get("fp16_std_ms")),
            _f(r.get("fp16_min_ms")),
            _f(r.get("fp16_max_ms")),
            _f(r.get("ms_per_sample"), ".4f"),
            _f(r.get("fp16_throughput_sps"), ".1f"),
            _f(r.get("fp16_peak_mem_delta_mb")),
            _f(r.get("fp16_mem_per_mparam"), ".3f"),
            _f(r.get("observed_tflops"), ".4f"),
            _f(r.get("arithmetic_intensity"), ".2f"),
            _f(r.get("compute_efficiency_pct"), ".3f"),
            _f(r.get("mem_bandwidth_gb_s"), ".3f"),
            _f(r.get("ms_per_gflop"), ".4f"),
            _f(r.get("sps_per_mparam"), ".2f"),
            _f(r.get("sps_per_fp16_mb"), ".2f"),
            _f(r.get("gpu_util_pct"), ".1f"),
        ])
    print("\n\n" + "═" * 64)
    print("  BENCHMARK SUMMARY")
    print("═" * 64)
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))


def save_csv(results: List[Dict[str, Any]], path: str) -> None:
    import csv
    if not results:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  CSV saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"\n{'═'*64}")
    print(f"  PyTorch FP16 Model Efficiency Benchmark")
    print(f"{'═'*64}")
    print(f"  Device   : {DEVICE.upper()}")
    print(f"  Warmup   : {N_WARMUP}   Timed runs : {N_RUNS}")
    print(f"  CPU cmp  : {RUN_CPU_COMPARE}")
    print(f"  Models   : {len(BENCHMARK_CONFIGS)}")

    all_results: List[Dict[str, Any]] = []

    for cfg in BENCHMARK_CONFIGS:
        try:
            all_results.append(benchmark_one(cfg))
        except Exception:
            print(f"\n[ERROR] Failed: {cfg['log_name']}")
            traceback.print_exc()

    print_summary(all_results)

    if SAVE_RESULTS_CSV:
        save_csv(all_results, os.path.join(RESULTS_DIR, "benchmark_results.csv"))


if __name__ == "__main__":
    main()