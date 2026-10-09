import subprocess
from pathlib import Path

# After visual inspection this method is worse than FS skull_strip

def batch_fsl_bet(source_dir, dest_dir, f_threshold=0.2, g_bias=0, generate_mask=True):
    """
    Applies FSL BET to all .nii or .nii.gz files in source_dir and saves to dest_dir.
    
    Args:
        source_dir (str or Path): Directory with input volumes.
        dest_dir (str or Path): Directory to save skull-stripped outputs.
        f_threshold (float): BET -f value (brain intensity fraction).
        g_bias (float): BET -g value (vertical gradient bias).
        generate_mask (bool): Whether to output binary mask alongside brain.
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    nii_files = sorted(source_dir.glob("*.nii")) + sorted(source_dir.glob("*.nii.gz"))

    for input_path in nii_files:
        output_path = dest_dir / input_path.name  # Save with same name
        output_base = output_path.with_suffix('')  # Remove `.gz` or `.nii` suffix

        cmd = [
            'bet', str(input_path), str(output_base),
            '-f', str(f_threshold),
            '-g', str(g_bias)
        ]

        if generate_mask:
            cmd.append('-m')  # Saves mask as <output>_mask.nii.gz

        print(f"[RUNNING] {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[ERROR] {input_path.name}\n{result.stderr}")
        else:
            print(f"[OK] {input_path.name} → {output_path.name}")

# Example usage
batch_fsl_bet(
    source_dir='/Drive4T/inam/MRMRData/T1/skull_stripped/3T_t1',
    dest_dir='/Drive4T/inam/MRMRData/T1/testing_skulling_.35',
    f_threshold=0.35,   # Conservative
    # g_bias=0.0         # No bias
)
