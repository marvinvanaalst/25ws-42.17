import os
from glob import glob
from tqdm import tqdm
from pathlib import Path

if __name__ == "__main__":
    from slicify_nii import process_all_scans_with_masks
else:
    from patchify.slicify.slicify_nii import process_all_scans_with_masks

def create_slicify_dataset(config_dict):
    scan_components = [config_dict['data']['direction']['source'], config_dict['data']['direction']['target'], config_dict['data']['direction']['mask_dir_name']]
    slice_var = f"{config_dict['slicify']['resize_to']}x{config_dict['slicify']['pad_to']}"
    
    if os.path.exists(f"{config_dict['data']['training']['subjects_path']}/slices/{slice_var}"):
        print(f"{config_dict['data']['training']['subjects_path']}/slices/{slice_var}", " Exists.")
        try:
            base_list = os.listdir(f"{config_dict['data']['training']['subjects_path']}/slices/{slice_var}/{config_dict['data']['direction']['mask_dir_name']}/all")
            for idx, ele in enumerate(config_dict['data']['training']['subjects']):
                if ele.split('.')[0] in base_list:
                    if idx+1 == len(config_dict['data']['training']['subjects']):
                        print("Skipping patching. Patch directories for all scans in the config file were found in the destination directory:")
                        print("Path : ", f"{config_dict['data']['training']['subjects_path']}/slices/{slice_var}/{config_dict['data']['direction']['source']}\n\n")
                        return
                    continue
                else:
                    print(ele ," MISS HIT")
                    break            
        except:
            pass
    
    # if os.path.exists(f"{config_dict['data']['training']['subjects_path']}/slices/{slice_var}"):
    #     if not os.path.exists(f"{config_dict['data']['training']['subjects_path']}/slices/{slice_var}/all"):
    #         os.makedirs(f"{config_dict['data']['training']['subjects_path']}/slices/{slice_var}/{config_dict['data']['direction']['mask_dir_name']}/all", exist_ok=True)
    #     if len(os.listdir(f"{config_dict['data']['training']['subjects_path']}/slices/{slice_var}/{config_dict['data']['direction']['mask_dir_name']}/all")) == len(config_dict['data']['training']['subjects']) or len(os.listdir(f"{config_dict['data']['training']['subjects_path']}/slices/{slice_var}/{config_dict['data']['direction']['mask_dir_name']}/all")) > len(config_dict['data']['training']['subjects']):
    #         print(f"Slices already exist at {config_dict['data']['training']['subjects_path']}/slices. Skipping patch creation.")
    #         return
    #     else:
    #         pass
    #         # os.rmdir(f"{config_dict['data']['training']['subjects_path']}/patches/{patch_size}X{patch_overlap}")
    #         # print(f"Removed existing patches directory at {config_dict['data']['training']['subjects_path']}/patches/{patch_size}X{patch_overlap}.")
    #         # print("Recreating patches...")
    print(f"Slices not found. Creating slices in: {config_dict['data']['training']['subjects_path']}/slices/ ...")
    for scan_type in scan_components:
        source_dir = f"{config_dict['data']['training']['subjects_path']}/{scan_type}"

        mask_dir = f"{config_dict['data']['training']['subjects_path']}/{config_dict['data']['direction']['mask_dir_name']}/"
        dest_dir = f"{config_dict['data']['training']['subjects_path']}/slices/{slice_var}/{scan_type}"
        os.makedirs(dest_dir, exist_ok=True)
        nii_files = sorted(glob(f"{source_dir}/*.nii.gz"))
        # print(nii_files)
        # print(f"Processing {scan_type}")
        # print(f"Found {len(nii_files)} .nii.gz files in {source_dir}")
        
        if not nii_files:
            raise FileNotFoundError(f"No .nii.gz files found in {source_dir}")

        print(f"Forcing axis to {config_dict['data']['slice_axis']} for all scans")
        process_all_scans_with_masks(
            config=config_dict,
            scan_dir=source_dir,
            mask_dir=mask_dir,  # Assuming masks are in the same directory
            output_dir=dest_dir,
            device=config_dict['device']['devices'][0],
            cache_in_ram='cuda',
            axis=config_dict['data']['slice_axis']
        )

    print(f"Finished slicing the scans. Slices saved to {config_dict['data']['training']['subjects_path']}/slices.")
