import os
import shutil
import json

# ===== USER CONFIG =====
root1 = "/storage/an_inam/PGI_data/only_rigid"  # original data directory
root2 = "/storage/an_inam/PGI_data/opusCompliant"  # target rearranged directory
json_file = os.path.join(root2, "patients.json")
modalities = {
    "ct.nii.gz": "CT",
    "mr.nii.gz": "MRI",
    "mask.nii.gz": "MASK"
}

# ===== CREATE TARGET FOLDERS =====
for folder in modalities.values():
    os.makedirs(os.path.join(root2, folder), exist_ok=True)

# ===== PROCESS PATIENTS =====
patient_list = []
from tqdm import tqdm
tqdm_loop = tqdm(sorted(os.listdir(root1)), desc="Processing patients", unit="patient")
for idx, patient_name in enumerate(tqdm_loop):
    patient_path = os.path.join(root1, patient_name)
    if not os.path.isdir(patient_path):
        continue  # skip non-folders

    patient_filename = f'{patient_name}.nii.gz'
    patient_list.append(patient_filename)


    # Copy each modality file
    for src_name, modality_folder in modalities.items():
        src_file = os.path.join(patient_path, src_name)
        if os.path.exists(src_file):
            dst_file = os.path.join(root2, modality_folder, patient_filename)
            shutil.copy2(src_file, dst_file)
        else:
            print(f"⚠ Missing file: {src_file}")

# patient_list_2 = []
# print(patient_list)
# for ele in patient_list:
#     patient_list.append(ele.replace('', '""'))
# print(patient_list_2)

# ===== SAVE YAML FILE =====
patient_list = [s.strip('"') for s in patient_list]
with open(json_file, "w") as f:
    json.dump(patient_list, f, indent=4)

print(f"✅ Done. Rearranged data in '{root2}' and saved patient list to '{json_file}'.")
