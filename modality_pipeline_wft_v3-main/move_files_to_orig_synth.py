import os
import shutil
from pathlib import Path

def organize_scans(dir_list):
    """
    For each directory in dir_list:
      - Move the '_orig' file to <dir>_orig/
      - Move the synthetic file (without '_orig') to <dir>_synth/
      - Remove '_orig' from filename in destination
    """
    for directory in dir_list:
        directory = Path(directory)
        if not directory.is_dir():
            print(f"Skipping {directory} — not a directory")
            continue

        # Create output directories
        orig_dir = directory.parent / f"{directory.name}_orig"
        synth_dir = directory.parent / f"{directory.name}_synth"
        orig_dir.mkdir(exist_ok=True)
        synth_dir.mkdir(exist_ok=True)

        # Iterate files inside the directory
        for file_path in directory.glob("*.nii.gz"):
            filename = file_path.name

            if "_orig" in filename:
                new_name = filename.replace("_orig", "")
                dest = orig_dir / new_name
                print(f"Moving real scan: {file_path.name} → {dest}")
                shutil.move(str(file_path), str(dest))
            else:
                dest = synth_dir / filename
                print(f"Moving synthetic scan: {file_path.name} → {dest}")
                shutil.move(str(file_path), str(dest))

if __name__ == "__main__":
    base_path ="/storage/an_inam/MR2MR/modality_pipeline_wft_v3/isbi_logs_C"
    dir_list = [f"{base_path}/{file}/generations" for file in os.listdir(base_path)]
    organize_scans(dir_list)