import os
import re

# Path to your folder
folder_paths = ["post_processing/SEG_MASKS/22_ESAU_RA_3D_Conv_Bounded_Tanh",
                "post_processing/SEG_MASKS/23_ESAU_RA_1_3D_Trilinear_Tanh",
                "post_processing/SEG_MASKS/23_ESAU_RA_3D_Nearsest_Tanh",
                "post_processing/SEG_MASKS/251112_UNET_3D_L1_32_conv",
                "post_processing/SEG_MASKS/251112_UNET_3D_L1_32_nearest",
                "post_processing/SEG_MASKS/251112_UNET_3D_L1_32_trilinear",
                "post_processing/SEG_MASKS/VNET_L1",
                "post_processing/SEG_MASKS/VNET_L1_GAN",
                "post_processing/SEG_MASKS/VNET_L1_Percept"
                ]
folder_paths = [
"/storage/an_inam/MR2MR/modality_pipeline_wft_v3/post_processing/SEG_MASKS_robust/30_ESAU_3D_L1_32_conv",
"/storage/an_inam/MR2MR/modality_pipeline_wft_v3/post_processing/SEG_MASKS_robust/30_ESAU_3D_L1_32_nearest",
"/storage/an_inam/MR2MR/modality_pipeline_wft_v3/post_processing/SEG_MASKS_robust/30_ESAU_3D_L1_32_trilinear"]
pattern = re.compile(r'^\d+_')  # matches leading digits followed by "_"
target_substring = "_saveEvery_epoch_epoch=300.pt_fake"

for folder_path in folder_paths:
    print("hello")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if target_substring in filename:
                new_filename = filename.replace(target_substring, "")
                new_path = os.path.join(folder_path, new_filename)
                os.rename(file_path, new_path)
        if os.path.isfile(file_path):
            if pattern.match(filename):
                new_filename = pattern.sub('', filename)  # remove the matched part
                new_path = os.path.join(folder_path, new_filename)
                os.rename(file_path, new_path)
                print(f"Renamed: {filename} → {new_filename}")