import os
import shutil

input_root = "/home/ig/Datasets/pipeline/data/230patds/data"
output_root = "/home/ig/Datasets/pipeline/data/230patds/ftw_Compliant"

modalities = ["mask"]

# Create output structure
for mod in modalities:
    os.makedirs(os.path.join(output_root, mod), exist_ok=True)

for patient in os.listdir(input_root):
    patient_dir = os.path.join(input_root, patient)

    # skip non-directories
    if not os.path.isdir(patient_dir):
        continue

    for mod in modalities:
        src = os.path.join(patient_dir, f"{mod}.nii.gz")
        dst = os.path.join(output_root, mod, f"{patient}.nii.gz")

        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"Warning: Missing {src}")
