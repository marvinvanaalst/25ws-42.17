import os
import subprocess
from pathlib import Path
from typing import List
from tqdm import tqdm

# ----------------------------
# USER CONFIG
# ----------------------------

# List of dataset directories containing .nii.gz scans
DATASET_DIRS: List[str] = [
    # "/storage/an_inam/MR2MR/Data/10_Pat_t1/7T",
    # "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/23_ESAU_RA_1_3D_Trilinear_Tanh/generations",
    # "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/23_ESAU_RA_3D_Nearsest_Tanh/generations",
    # "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/22_ESAU_RA_3D_Conv_Bounded_Tanh/generations",
    # "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/VNET_L1/generations",
    # "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/VNET_L1_GAN/generations",
    # "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/VNET_L1_Percept/generations",
    # "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/251112_UNET_3D_L1_32_conv/generations",
    # "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/251112_UNET_3D_L1_32_nearest/generations",
    # "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/251112_UNET_3D_L1_32_trilinear/generations"
    # "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/30_ESAU_3D_L1_32_conv/generations",
    # "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/30_ESAU_3D_L1_32_nearest/generations",
    # "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/30_ESAU_3D_L1_32_trilinear/generations",
    # "/storage/an_inam/datasets/OCT_15_fin/generated"
    # "/storage/an_inam/MR2MR/Data/10_Pat_t1/3T_7TReg"
    
    
        # "/storage/an_inam/datasets/OCT_15_fin/generated",
    # 30ESAU3D is trilinear
    
    "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/30_ESAU_3D_L1_8_nearest/generations"
    "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/30_ESAU_3D_L1_16_nearest/generations",
    
    "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/ISBI_SUB_30_ESAU_3D_L1_32_conv/generations",
    "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/ISBI_SUB_30_ESAU_3D_L1_32_nearest/generations",
    "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/30_ESAU_3D_L1_32_conv/generations"
    

    "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/2026_01_18_SwinUNETR_Max_Conv/generations"
    "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/2026_01_18_SwinUNETR_Max_Nearest/generations",
    "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/2026_01_18_SwinUNETR_Max_trilinear/generations",
    
    "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/2026_01_18_UNETR_full_conv/generations"
    "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/2026_01_18_UNETR_full_nearest/generations",
    "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/2026_01_18_UNETR_full_trilinear/generations",
    
    
    "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/251201_ESAU_3D_L1_32_FullComplemetn_trilinear/generations",

        
    "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/logs/2026_01_18_SandEUNET_conv_max/generations"
    "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/ISBI_fin_SE-Residual_3D_nearest/generations",
    "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/151112_SEResidual_3D_L1_32_conv/generations",
    
    
        "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/VNET_L1/generations"
    "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/2026_VNET_L1_GAN01_18_UNETR_full_nearest/generations",
    "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/VNET_L1_Percept/generations",
    
]

# Output root directory
OUTPUT_DIR = "SEG_MASKS_robust"

# Path to SynthSeg_predict.py (from your example)
SYNTHSEG_SCRIPT = str(Path.home() / "SynthSeg/scripts/commands/SynthSeg_predict.py")

# Number of CPU threads to use
THREADS = 8

# CUDA device
CUDA_DEVICE = "2"  # corresponds to cuda:2


# ----------------------------
# CORE FUNCTION
# ----------------------------

def run_synthseg_on_folder(input_dir: str, output_root: str):
    """Run SynthSeg on all eligible .nii.gz scans in a folder."""
    dataset_name = input_dir.split('/')[-2]
    output_dir = Path(output_root) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    nii_files = sorted(Path(input_dir).rglob("*.nii.gz"))
    if not nii_files:
        print(f"[!] No NIfTI files found in {input_dir}")
        return

    for nii_path in nii_files:
        if "orig" in nii_path.name.lower():
            print(f"[-] Skipping {nii_path.name} (contains 'original')")
            continue
        if "150" in nii_path.name.lower():
            print(f"[-] Skipping {nii_path.name} (does not contain 300)")
            continue
        
        output_path = output_dir / nii_path.name
        cmd = [
            "python", SYNTHSEG_SCRIPT,
            "--i", str(nii_path),
            "--o", str(output_path),
            "--threads", str(THREADS),
            # "--robust",           # adds intensity-robust preprocessing
            # "--resample"         # restores output to original voxel resolution
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE

        print(f"[+] Running SynthSeg on: {nii_path.name}")
        try:
            result = subprocess.run(
                    cmd,
                    env=env,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
        except subprocess.CalledProcessError as e:
                print(f"[✗] Error while processing {nii_path.name}")
                print("---- STDOUT ----")
                print(e.stdout)
                print("---- STDERR ----")
                print(e.stderr)
                raise  # optional: stop execution if one fails
        subprocess.run(cmd, env=env, check=True)
        print(f"[✓] Saved: {output_path}\n")


# ----------------------------
# MAIN EXECUTION
# ----------------------------

if __name__ == "__main__":
    for dataset_dir in DATASET_DIRS:
        print(f"\n=== Processing Dataset: {dataset_dir} ===")
        run_synthseg_on_folder(dataset_dir, OUTPUT_DIR)

    print("\nAll SynthSeg runs completed successfully.")