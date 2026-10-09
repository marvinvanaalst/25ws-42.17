import os

# Path to the folder containing the files
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

target_substring = "_saveEvery_epoch_epoch=300.pt_fake"

for folder_path in folder_paths:
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            if target_substring in filename:
                new_filename = filename.replace(target_substring, "")
                new_path = os.path.join(folder_path, new_filename)
                os.rename(file_path, new_path)
                print(f"Renamed: {filename} → {new_filename}")
            else:
                print("Deleating")
                print(file_path)
                aa = input("y/n")
                if aa == "x":
                    os.remove(file_path)
                    print(f"Deleted: {filename}")
