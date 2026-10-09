#!/usr/bin/env python3
import argparse
# import torch
from SynthSeg.predict import predict
import os
import time

def main():
    parser = argparse.ArgumentParser(description="Run SynthSeg on GPU (standalone version)")
    parser.add_argument("--i", "--input", dest="input", required=True,
                        help="Path to input MRI (.nii, .nii.gz, or .mgz)")
    parser.add_argument("--o", "--output", dest="output", required=True,
                        help="Path to output segmentation (.nii.gz or .mgz)")
    parser.add_argument("--vol", dest="vol", default=None,
                        help="Optional: path to CSV for volumetric measures")
    parser.add_argument("--robust-tissues", dest="robust_tissues", action="store_true",
                        help="Segment only CSF, GM, WM (3-class segmentation)")
    parser.add_argument("--robust-lesions", dest="robust_lesions", action="store_true",
                        help="Enable lesion-robust mode")
    parser.add_argument("--longitudinal", dest="longitudinal", action="store_true",
                        help="Enable longitudinal segmentation mode")
    parser.add_argument("--threads", dest="threads", type=int, default=4,
                        help="Number of CPU threads for preprocessing (default: 4)")
    parser.add_argument("--cpu", dest="cpu", action="store_true",
                        help="Force CPU mode (default: GPU if available)")
    parser.add_argument("--fast", dest="fast", action="store_true",
                        help="Run faster, lower-precision inference")
    args = parser.parse_args()

    # --- choose device ---
    gpu = torch.cuda.is_available() and not args.cpu
    device = "cuda" if gpu else "cpu"

    print(f"\n🧠 Running SynthSeg on {device.upper()} ...")
    start = time.time()

    # --- run segmentation ---
    predict(
        path_images=args.input,
        path_segmentations=args.output,
        path_volumes=args.vol,
        robust_tissues=args.robust_tissues,
        robust_lesions=args.robust_lesions,
        longitudinal=args.longitudinal,
        fast=args.fast,
        gpu=gpu,
        threads=args.threads,
    )

    print(f"\n✅ Finished in {time.time() - start:.1f} s ({device})")
    print(f"Output segmentation: {args.output}")
    if args.vol:
        print(f"Region volumes: {args.vol}")

if __name__ == "__main__":
    main()
