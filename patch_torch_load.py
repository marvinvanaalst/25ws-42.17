import re

files_to_patch = [
    "/home/ss_students/mtp/modality_pipeline_wft_v3-main/data_modules/opus_CachingPatchDataset.py",
    "/home/ss_students/mtp/modality_pipeline_wft_v3-main/components/testing_logic_patch.py",
    "/home/ss_students/mtp/modality_pipeline_wft_v3-main/data_modules/opus_CachingSliceDataset_finetune.py",
]

for fpath in files_to_patch:
    try:
        with open(fpath, "r") as f:
            content = f.read()

        # Replace torch.load(..., map_location=...) with weights_only=False
        new_content = re.sub(
            r'torch\.load\(([^,\)]+),\s*map_location=([^,\)]+)\)',
            r'torch.load(\1, map_location=\2, weights_only=False)',
            content
        )
        # Also replace standalone torch.load(path) if not already weights_only
        new_content = re.sub(
            r'torch\.load\(([^,\)]+)\)',
            r'torch.load(\1, weights_only=False)',
            new_content
        )
        
        with open(fpath, "w") as f:
            f.write(new_content)
        print(f"Patched torch.load in {fpath}")
    except Exception as e:
        print(f"Error patching {fpath}: {e}")
