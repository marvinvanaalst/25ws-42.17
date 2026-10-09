import os
import torch
from torch.utils.data import Dataset
import json

if not os.path.exists('/Drive4T/inam/3T_7T_Synthesis/data.json'):
    raise FileNotFoundError("Configuration file 'data.json' not found in the dataloader path.")

with open('/Drive4T/inam/3T_7T_Synthesis/data.json', 'r') as f:
    config = json.load(f)

# self.subjects_list = self.params['data']['training']['subjects']
# self.data_dir = self.params['data']['training']['subjects_path']
# self.src_key = self.params['direction']['source']
# self.tgt_key = self.params['direction']['target']

from tqdm import tqdm

# Loader for one dataloader
class PatchGPUDatasetTrain(Dataset):
    def __init__(self, config, data_dir, sample_list, device='cuda'):
        """
        source_dir, target_dir, mask_dir: base directories
        sample_list: list of sample folder names to include
        device: 'cuda' or 'cpu' depending on caching preference
        """
        self.device = device
        self.cache = []  # List of dicts: {'source': ..., 'target': ..., 'mask': ...}
        source_dir = os.path.join(data_dir, config['direction']['source'])
        target_dir = os.path.join(data_dir, config['direction']['target'])
        mask_dir   = os.path.join(data_dir, 'masks_patches')
        print(f"Loading samples from: \nSource :{source_dir}, \nTarget{target_dir}, \nMask{mask_dir}")
        self._build_cache(source_dir, target_dir, mask_dir, sample_list)

    def _build_cache(self, source_dir, target_dir, mask_dir, sample_list):
        sample_loading_loop = tqdm(sample_list, desc=f"Loading samples onto {self.device}", unit="sample")
        for idx, sample_name in enumerate(sample_loading_loop):
            source_path = os.path.join(source_dir, sample_name)
            target_path = os.path.join(target_dir, sample_name)
            mask_path   = os.path.join(mask_dir,   sample_name)

            # Sort to ensure consistent ordering
            pt_files = sorted(os.listdir(source_path))

            for pt_file in pt_files:
                source_file = os.path.join(source_path, pt_file)
                target_file = os.path.join(target_path, pt_file)
                mask_file   = os.path.join(mask_path,   pt_file)

                # Load and push to GPU
                source_tensor = torch.load(source_file, map_location=self.device)
                target_tensor = torch.load(target_file, map_location=self.device)
                mask_tensor   = torch.load(mask_file,   map_location=self.device)

                # Ensure matching keys
                assert source_tensor.keys() == target_tensor.keys() == mask_tensor.keys(), \
                    f"Mismatched keys in {pt_file} for sample {sample_name}"

                # Push each tensor inside the dict to GPU
                # print(source_tensor['filename'],'\n', target_tensor['filename'], '\n',mask_tensor['filename'])
                # print(source_tensor['location'], '\n',target_tensor['location'], '\n', mask_tensor['location'])
                # print(source_tensor['affine'],'\n', target_tensor['affine'],'\n', mask_tensor['affine'])
                
                
                # input("Press Enter to continue...")  # Debugging line
                out_dict = {
                    'source': source_tensor['patch'],
                    'target': target_tensor['patch'],
                    'mask':   mask_tensor['patch'],
                }

                self.cache.append(out_dict)

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        return self.cache[idx]
    
if __name__ == "__main__":
    # Example usage
    all_patients = os.listdir('/Drive4T/inam/3T_7T_Synthesis/patches_ds/3T_t1_patches')
    print(f"Found {len(all_patients)} patients in the dataset.")
    
    sample_list = all_patients  # Use all patients or specify a subset
    with open('/Drive4T/inam/3T_7T_Synthesis/data.json', 'r') as f:
        config = json.load(f)


    dataset = PatchGPUDatasetTrain(config, data_dir='/Drive4T/inam/3T_7T_Synthesis/patches_ds', sample_list=sample_list, device='cuda:0')
    for i in range(len(dataset)):
        sample = dataset[i]
        print(f"Sample {i}: {sample['source'].shape}, {sample['target'].shape}, {sample['mask'].shape}")
        # input("Press Enter to continue...")
        # print(f"Sample {i}: {sample['source'].keys()}, {sample['target'].keys()}, {sample['mask'].keys()}")
    print(f"Dataset length: {len(dataset)}")
    input()
