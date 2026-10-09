import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import json
from torch.utils.data import Subset, DataLoader
import torchio as tio
from sklearn.model_selection import train_test_split

import random
class CachingSliceDataset(Dataset):
    def __init__(self, config, sample_list, device, data_grouping='slices', type_load='all', affine=None):
        """
        Unified dataset that supports both RAM and GPU caching.
        
        Args:
            data_dir (str): Path to the root directory.
            sample_list (List[str]): Sample folder names to load.
            cache_mode (str): 'gpu' or 'ram' — determines caching location.
        """
        if device.startswith("cuda") or device.startswith("cpu"):
            cache_mode = device
        else:
            raise ValueError(f"Invalid device: {device}. Use 'cuda[:x]' or 'cpu'.")
        
        self.complete_cache=True
        print("Loading", sample_list)
        
        self.config = config
        self.cache_mode = cache_mode
        self.device = device
        self.cache = []
        slice_var = f"{self.config['training']['slice_size']}x{self.config['training']['slice_size']}"
        # assert os.path.exists(f"{self.config['data']['training']['subjects_path']}/slices/{self.config['training']['patch_size']}X{self.config['training']['patch_overlap']}")
        data_dir = f"{self.config['data']['training']['subjects_path']}/{data_grouping}/"
        source_dir = os.path.join(data_dir,slice_var, f"{self.config['data']['direction']['source']}/{type_load}")
        target_dir = os.path.join(data_dir,slice_var, f"{self.config['data']['direction']['target']}/{type_load}")
        mask_dir   = os.path.join(data_dir,slice_var, f"{self.config['data']['direction']['mask_dir_name']}/{type_load}")

        print(f"[{cache_mode.upper()} CACHE] Loading samples from:\n"
              f"  Source : {source_dir}\n"
              f"  Target : {target_dir}\n"
              f"  Mask   : {mask_dir}")
        
        self._build_cache(source_dir, target_dir, mask_dir, sample_list)

    def _build_cache(self, source_dir, target_dir, mask_dir, sample_list):
        sample_loading_loop = tqdm(sample_list, desc=f"[{self.cache_mode.upper()}] Caching", unit="sample")

        for sample_name in sample_loading_loop:
            source_path = os.path.join(source_dir, sample_name)
            target_path = os.path.join(target_dir, sample_name)
            mask_path   = os.path.join(mask_dir,   sample_name)

            pt_files = sorted(os.listdir(source_path))
            

            for pt_file in pt_files:
                
                source_file = os.path.join(source_path, pt_file)
                target_file = os.path.join(target_path, pt_file)
                mask_file   = os.path.join(mask_path,   pt_file)

                if os.path.isfile(source_file) and os.path.isfile(target_file) and os.path.isfile(mask_file):
                    source_tensor = torch.load(source_file, map_location=self.device, weights_only=False)
                    target_tensor = torch.load(target_file, map_location=self.device, weights_only=False)
                    mask_tensor   = torch.load(mask_file, map_location=self.device, weights_only=False)

                    assert source_tensor.keys() == target_tensor.keys() == mask_tensor.keys(), \
                        f"[Key Mismatch] {pt_file} in sample {sample_name}"

                    out_dict = {
                        'source': source_tensor['data'],
                        'target': target_tensor['data'],
                        'mask':   mask_tensor['data'],
                    }
                    if 'affine' in source_tensor:
                        out_dict['affine'] = source_tensor['affine']
                        out_dict['original_size'] = source_tensor['original_size']
                        out_dict['padding'] = source_tensor['padding']
                        out_dict['path_scan'] = source_tensor['path_scan']
                        out_dict['path_mask'] = source_tensor['path_mask']
                    
                    self.cache.append(out_dict)

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        return self.cache[idx]

class CachedTorchIODataset(tio.SubjectsDataset):
    def __init__(self, subject_list, transform=None, cache_enabled=True):
        
        super().__init__(subject_list, transform=transform)
        self.cache_enabled = cache_enabled
        self._cache = {}  # will store subject tensors

    def __getitem__(self, index):
        if self.cache_enabled and index in self._cache:
            return self._cache[index]

        subject = super().__getitem__(index)

        if self.cache_enabled:
            self._cache[index] = subject

        return subject


class cache_slice_dataloader():
    def __init__(self, params_data, device, idx=0, n_folds=5):
        super().__init__()
        self.device = device
        assert device.startswith("cuda") or device.startswith("cpu")

        self.params = params_data
            
        self.data_dir = self.params['data']['training']['subjects_path']
        
        self.subjects_list = [ele.split('.')[0] for ele in self.params['data']['training']['subjects']] # Only the subject name.
        
        self.subjects_list = sorted(self.subjects_list)


        n_patients = len(self.subjects_list)

        # Calculate fold sizes
        base_size = n_patients // n_folds
        remainder = n_patients % n_folds
        fold_sizes = [base_size + (1 if i < remainder else 0) for i in range(n_folds)]
        fold_starts = [sum(fold_sizes[:i]) for i in range(n_folds)]

        # Get validation indices for this fold
        val_start = fold_starts[idx]
        val_end = val_start + fold_sizes[idx]
        val_idx = list(range(val_start, val_end))
        train_idx = list(range(0, val_start)) + list(range(val_end, n_patients))

        
        self.train_subset=  [self.subjects_list[i] for i in train_idx]
        self.val_subset =  [self.subjects_list[i] for i in val_idx]
        self.test_subset = self.val_subset
        # print('A', self.train_subset, len(self.train_subset))
        # print('B', self.val_subset, len(self.val_subset))
        # print('C', self.test_subset, len(self.test_subset))
        
        # input("Press enter to continue with the training and validation subsets")
        
        if self.params['fine_tunning'].get('pretrain_boolean', None) == True:
            if self.params['fine_tunning'].get('pretrain_evaluate_only', None)== True:
                input("Running in pretrained evaluation mode, press enter to continue")
                self.test_subset = self.subjects_list  # Use all subjects for testing
            
            
            
         
            
        # print("\033\n[91mTesting Sample", os.path.join(self.params['data']['training']['subjects_path'], self.params['data']['direction']['source'], f"{self.test_subset[0]}.nii.gz \n\033[0m"))
        print(f"Train subset: {len(self.train_subset)} samples, Val subset: {len(self.val_subset)} samples, Test subset: {len(self.test_subset)} samples")
        print(f"Train subset 5 samples\n: {self.train_subset[:5]}...")  # Show first 5 samples for brevity
        print(f"\033Val subset 5\033[0m\n samples: {self.val_subset[:5]}...")  # Show first 5 samples for brevity
        print(f"\033Test subset all sample\033[0m\n s: {self.test_subset}")  # Show first 5 samples for brevity
        
    def fetch_TrainValLoaders(self):
        """Fetches the training, validation, and test subsets as DataLoaders.
        """
        self.batch_size = self.params['training']['batch_size']

        return {
            'train':        DataLoader(CachingSliceDataset(self.params, self.train_subset,   self.device , type_load='non_empty'), batch_size=self.batch_size, shuffle=True),
            'validation':   DataLoader(CachingSliceDataset(self.params, self.val_subset,     self.device , type_load='non_empty'),  batch_size=self.batch_size, shuffle=False),
            'number_test_samples': int(len(self.test_subset)),
        }
    def fetch_testLoader(self, idx=0):
        """Fetches the test subset as a DataLoader.
        
        Args:
            idx (int): Index of the sample to fetch.
        
        Returns:
            DataLoader: DataLoader for the test subset.
        """
        return {
            'test' : DataLoader(CachingSliceDataset(self.params, [self.test_subset[idx]], self.device, type_load='all', affine=True), batch_size=self.batch_size, shuffle=False),
            'test_name' : self.test_subset[idx],
            }

    
if __name__ == "__main__":
    sample_index = 4
    with open('/Drive4T/inam/3T_7T_Synthesis/configs/VNET_Paper_precompute.json', 'r') as f:
        config = json.load(f)
    for k, v in config.items():
        print(f"{k}: {v}")
    cached_dataloader = cache_patch_dataloader(config, device='cuda:0', idx=sample_index).fetch_allloaders()
    input()
    test_dataloader = cached_dataloader['test']
    from utils.eval_metrics import VolumeIQAMetrics
    DEVICE = 'cuda:0'
    
    iqa_metrics = VolumeIQAMetrics(device=DEVICE)
    
    
    for train_batch_idx, batch in enumerate(test_dataloader['queue']):
        fake_output = batch['target']['data'].to(DEVICE)
        test_dataloader['aggregator'].add_batch(fake_output, batch['location'])
    print(test_dataloader['original'][0]['name'])
    output = (test_dataloader['aggregator'].get_output_tensor().to(DEVICE))*(test_dataloader['original'][0]['mask']['data'].to(DEVICE))
    output = torch.clip(output, min=0.0, max=1.0)
    
    # print(f"Output shape: {output.shape}")
    # print(f"Original shape: {orig[0]['target']['data'].shape}")
    # input()
    
    
    
    test_metrics = iqa_metrics.compute(output.unsqueeze(1), test_dataloader['original'][0]['target']['data'].to(DEVICE).unsqueeze(1)*test_dataloader['original'][0]['mask']['data'].to(DEVICE))
    # test_metrics = iqa_metrics.compute(output.unsqueeze(1), test_dataloader['original'][0]['target']['data'].to(DEVICE).unsqueeze(1))
    
    metrics = iqa_metrics.value()
    iqa_metrics.reset()
    print(f"Test Metrics for sample {sample_index}: PSNR: {metrics['PSNR']}, SSIM: {metrics['SSIM']}, MAE: {metrics['MAE']}, MSE: {metrics['MSE']}")
    sample_test_metics = {
        'sample_index': sample_index,
        'test_psnr': metrics['PSNR'],
        'test_ssim': metrics['SSIM'],
        'test_mae': metrics['MAE'],
        'test_mse': metrics['MSE']
    }
    
    print(f"{test_dataloader['original'][0]['name']}_fake.nii.gz")
    # input(0)
    
    # final_results.append(sample_test_metics)
    # csv_logger_testing.log(sample_test_metics)
    os.makedirs('generations', exist_ok=True)
    affine = batch['source']['affine'][0]
    image = tio.ScalarImage(tensor=output.cpu(), affine=affine)
    image.save(f"generations/{sample_index}_{test_dataloader['original'][0]['name']}_fake.nii.gz")
    test_dataloader['original'][0]['target'].save(f"generations/{sample_index}_{test_dataloader['original'][0]['name']}_orig.nii.gz")
    
    
    # for i, sample in enumerate(cached_dataloader['train']):
    #     # print(sample)
    #     for k, v in sample.items():
    #         print(f"{k}: {v.shape}")
    # input("Press Enter to continue...")
        # print(f"Train Sample {i}: {sample['source'].shape}, {sample['target'].shape}, {sample['mask'].shape}")

        
    # # Example usage
    # all_patients = os.listdir('/Drive4T/inam/3T_7T_Synthesis/patches_ds/3T_t1_patches')
    # print(f"Found {len(all_patients)} patients in the dataset.")
    
    # sample_list = all_patients  # Use all patients or specify a subset
    # with open('/Drive4T/inam/3T_7T_Synthesis/data.json', 'r') as f:
    #     config = json.load(f)


    # dataset = CachingPatchDataset(config, data_dir='/Drive4T/inam/3T_7T_Synthesis/patches_ds', sample_list=sample_list, device='cuda:0')
    # for i in range(len(dataset)):
    #     sample = dataset[i]
    #     print(f"Sample {i}: {sample['source'].shape}, {sample['target'].shape}, {sample['mask'].shape}")
    #     # input("Press Enter to continue...")
    #     # print(f"Sample {i}: {sample['source'].keys()}, {sample['target'].keys()}, {sample['mask'].keys()}")
    # print(f"Dataset length: {len(dataset)}")
    # input()
    
    
