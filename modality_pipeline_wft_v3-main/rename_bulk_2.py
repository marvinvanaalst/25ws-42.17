import os
import re
from pathlib import Path

# --- Set your root directory ---
root_dir = Path("/storage/an_inam/MR2MR/modality_pipeline_wft_v3/isbi_logs")

# --- Walk through all subdirectories recursively ---
for file_path in root_dir.rglob("*"):
    if file_path.is_file():
        filename = file_path.name

        # Remove '_saveEvery_epoch_epoch=300.pt_fake' substring
        new_name = filename.replace("_saveEvery_epoch_epoch=300.pt_fake", "")

        # Remove leading number + underscore (e.g., "5_sub-01_T1w..." → "sub-01_T1w...")
        new_name = re.sub(r"^\d+_", "", new_name)

        # If name changed, rename
        if new_name != filename:
            new_path = file_path.with_name(new_name)
            print(f"Renaming: {file_path} → {new_path}")
            file_path.rename(new_path)
