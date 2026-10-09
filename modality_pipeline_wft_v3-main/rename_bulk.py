import os
import re
from pathlib import Path

import os

# --- Configure directory ---
directory = Path("isbi_logs")

models = [f"{directory}/{model}" for model in os.listdir(directory)]
models = [f"{f_name}/generations" for f_name in models]

# print(models)
for model in models:
# --- Loop through files ---
    files = [f"{model}/{file}" for file in os.listdir(model)]
    for file_path in files:
        print("aaaa", file_path)
        # if file_path.is_file():
        #     filename = file_path.name
        # stemm = file_path.split('/')[-1].split('_')[1]
        # 1. Delete if '150' in filename
        if "150" in file_path:
            print(f"Deleting: {file_path}")
            os.remove(file_path)
            input(file_path)
            continue
        # if "_saveEvery_epoch_epoch=300.pt_fake" in stemm:
        # 2. Remove '_saveEvery_epoch_epoch=300.pt_fake' substring
        new_name = file_path.replace("_saveEvery_epoch_epoch=300.pt_fake", "")

        # 3. Remove leading number + underscore (e.g., '123_filename.pt' -> 'filename.pt')
        new_name = re.sub(r"^\d+_", "", new_name)
        print(new_name)
        print(file_path)
        input()
        # --- Rename if name changed ---
        if new_name != file_path:
            new_path = file_path.with_name(new_name)
            print(f"Renaming: {file_path} → {new_name}")
            file_path.rename(new_path)
