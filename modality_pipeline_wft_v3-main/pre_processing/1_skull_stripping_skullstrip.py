import subprocess
from pathlib import Path
import os
from tqdm import tqdm
# After visiual inspection this method is better than FSL BET

def batch_skull_strip(source_dir, dest_dir, skullstrip_path):
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    mask_path = f"{dest_dir}/mask_7T"
    os.makedirs(mask_path, exist_ok=True)

    nii_files = sorted(source_dir.glob('*.nii')) + sorted(source_dir.glob('*.nii.gz'))
    loopy = tqdm(nii_files)
    for idx, file_path in enumerate(loopy):
        out_path = dest_dir / file_path.name

        cmd = [
            'python', str(skullstrip_path),
            '-i', str(file_path),
            '-o', str(out_path),
            '-m', str(dest_dir / 'mask_7T' / file_path.name),
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[ERROR] Failed on {file_path.name}")
            print(result.stderr)
        else:
            print(f"[OK] {file_path.name} → {out_path}")

# Example usage
batch_skull_strip(
    source_dir='/Drive4T/inam/datasets/20_patient/raws/preprocessing/raw/T1/7T',
    dest_dir='/Drive4T/inam/datasets/20_patient/raws/preprocessing/1_skull_strip/T1/7T',
    skullstrip_path='/Drive4T/inam/datasets/synthstrip-docker'  # This should be the .py file from Harvard
)
