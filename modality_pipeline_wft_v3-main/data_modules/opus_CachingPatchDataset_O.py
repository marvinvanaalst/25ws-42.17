import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import json
from torch.utils.data import Subset, DataLoader
import torchio as tio

import random
class CachingPatchDataset(Dataset):
    def __init__(self, config, sample_list, device):
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
        # print("\n\n Caching ", [ele for ele in sample_list])
        
        self.config = config
        self.cache_mode = cache_mode
        self.device = device
        self.cache = []
        self.dynamic_loading = None
        self.built_cache = None
        # print(f"{self.config['data']['training']['subjects_path']}/patches/{self.config['patchify']['patch_size']}X{self.config['patchify']['patch_size']}")
        # input()
        
        patch_size = config['patchify']['patch_size']
        patch_overlap = config['patchify']['patch_overlap']
        if 'intensity_augmentations' not in config:
            config['intensity_augmentations'] = {}
            config['intensity_augmentations']['mri_intensity_percentile'] = [0, 100]
        intensity = config['intensity_augmentations']['mri_intensity_percentile'][1]
    
        print("Target Scan Intensity = ", intensity)
        
        
        
        
        # assert os.path.exists(f"{self.config['data']['training']['subjects_path']}/patches/{self.config['patchify']['patch_size']}X{self.config['patchify']['patch_overlap']}")
        assert os.path.exists(os.path.exists(f"{config['data']['training']['subjects_path']}/patches/{patch_size}X{patch_overlap}_intensity={intensity}"))
        data_dir = f"{self.config['data']['training']['subjects_path']}/patches/{self.config['patchify']['patch_size']}X{self.config['patchify']['patch_overlap']}"
        data_dir = f"{config['data']['training']['subjects_path']}/patches/{patch_size}X{patch_overlap}_intensity={intensity}"
        source_dir = os.path.join(data_dir, self.config['data']['direction']['source'])
        target_dir = os.path.join(data_dir, self.config['data']['direction']['target'])
        mask_dir   = os.path.join(data_dir, self.config['data']['direction']['mask_dir_name'])

        # print(f"[{cache_mode.upper()} CACHE] Loading samples from:\n"
        #       f"  Source : {source_dir}\n"
        #       f"  Target : {target_dir}\n"
        #       f"  Mask   : {mask_dir}")
        if self.config["device"].get("cache_device", "same_gpu") == "same_gpu":
            self.built_cache = True
            self._build_cache(source_dir, target_dir, mask_dir, sample_list, device=device)
        elif self.config["device"].get("cache_device", "ram") == "ram":
            self.built_cache = True
            self._build_cache(source_dir, target_dir, mask_dir, sample_list, device='cpu')
        elif self.config["device"].get("cache_device", "none") == "none":
            self.built_cache = False
            self.cache.append(out_dict)
            for sample_name in sample_list:
                source_path = os.path.join(source_dir, sample_name)
                target_path = os.path.join(target_dir, sample_name)
                mask_path   = os.path.join(mask_dir,   sample_name)

                pt_files = sorted(os.listdir(source_path))

                for pt_file in pt_files:
                    source_file = os.path.join(source_path, pt_file)
                    target_file = os.path.join(target_path, pt_file)
                    mask_file   = os.path.join(mask_path,   pt_file)

                    source_tensor = torch.load(source_file, map_location=self.device)
                    target_tensor = torch.load(target_file, map_location=self.device)
                    mask_tensor   = torch.load(mask_file,   map_location=self.device)

                    assert source_tensor.keys() == target_tensor.keys() == mask_tensor.keys(), \
                        f"[Key Mismatch] {pt_file} in sample {sample_name}"

                    out_dict = {
                        'source': source_file,
                        'target': target_file,
                        'mask':   mask_file,
                    }
                
        else:
            raise ValueError("Missing Preload Stratergy. Keep ['ram', 'same_gpu', 'none']")

    def _build_cache(self, source_dir, target_dir, mask_dir, sample_list, device):
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

                source_tensor = torch.load(source_file, map_location=self.device)
                target_tensor = torch.load(target_file, map_location=self.device)
                mask_tensor   = torch.load(mask_file,   map_location=self.device)

                assert source_tensor.keys() == target_tensor.keys() == mask_tensor.keys(), \
                    f"[Key Mismatch] {pt_file} in sample {sample_name}"

                out_dict = {
                    'source': source_tensor['patch'],
                    'target': target_tensor['patch'],
                    'mask':   mask_tensor['patch'],
                }
                
                self.cache.append(out_dict)

    def __len__(self):
        # if self.built_cache:
        return len(self.cache)

    def __getitem__(self, idx):
        if self.built_cache:
            return self.cache[idx]
        elif not self.built_cache:
            temp = self.cache[idx]
            
            source_tensor = torch.load(temp['source'], map_location=self.device)
            target_tensor = torch.load(temp['target'], map_location=self.device)
            mask_tensor   = torch.load(temp['mask'],   map_location=self.device)

            return {
                'source': source_tensor['patch'],
                'target': target_tensor['patch'],
                'mask':   mask_tensor['patch'],
            }
            

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


class cache_patch_dataloader():
    def __init__(self, params_data, device, idx = 0):
        super().__init__()
        print("Running no rescaled internsity. MRIxFields Edition")
        self.device = device
        assert device.startswith("cuda") or device.startswith("cpu")

        self.params = params_data
        self.data_dir = self.params['data']['training']['subjects_path']
        
        num_test_samples = self.params['training']['splits'].get('num_test_samples',1)
        test_samples_floor = idx*num_test_samples
        test_sample_ceil = test_samples_floor + num_test_samples
        
        self.subjects_list = [ele.split('.')[0] for ele in self.params['data']['training']['subjects']] # Only the subject name.
        self.subjects_list = sorted(self.subjects_list)
        
        
        self.test_subset = self.subjects_list[test_samples_floor:test_sample_ceil]

        train_val_samples = [s for i, s in enumerate(self.subjects_list) if s not in self.test_subset]

        
        num_samples_validation = self.params['training']['splits']['validation_samples']
        assert type(num_samples_validation) == int
        
        num_samples = len(train_val_samples)
        all_indices = list(range(num_samples))
        random.seed(42) 
        random.shuffle(all_indices)
        # Validation is number of samples based.
        if num_samples_validation == 0:
            val_indices = []
        else:
            val_indices = all_indices[:num_samples_validation]
        train_indices = all_indices[num_samples_validation:]
        self.train_subset = Subset(train_val_samples, train_indices)
        self.val_subset   = Subset(train_val_samples, val_indices)
        
        """Change Start"""
        if self.params['fine_tunning'].get('pretrain_boolean', None) == True:
            if self.params['fine_tunning'].get('pretrain_evaluate_only', None)== True:
                self.test_subset = [*sorted(self.subjects_list)]  # Use all subjects for testing
                self.train_subset = [self.subjects_list[0]] # Using so as not to break the code. Not used down the line
                self.val_subset = [self.subjects_list[0]]
                print(f"Running in pretrained evaluation mode. Evaluating on all {len(self.test_subset)} samples. Press enter to continue")
                
        print("\n\Loading Data to memory. Data Set Configuration: \n idx==",idx)
        print("Testing Set    --  ", [ele for ele in self.test_subset])
        print("Validation Set --  ", [ele for ele in self.val_subset])
        print("Training Set   --  ", [ele for ele in self.train_subset])
        
        """Change End"""
        # print(self.test_subset[0])
        # input("Check 1, expecting no .nii.ngz.")
        if "intensity_augmentations" not in self.params:
            self.params['intensity_augmentations']['mri_intensity_percentile'] = [0,100]
        
        self.test_loaders_list = []
        for i in range(len(self.test_subset)):
            # print("Test Sample", os.path.join(self.params['data']['training']['subjects_path'], self.params['data']['direction']['source'], f"{self.test_subset[i]}.nii.gz"))
            self.test_subject_tio = tio.Subject(
                name=self.test_subset[i].split('.')[0],
                source = tio.ScalarImage(os.path.join(self.params['data']['training']['subjects_path'], self.params['data']['direction']['source'], f"{self.test_subset[i]}.nii.gz")),
                target = tio.ScalarImage(os.path.join(self.params['data']['training']['subjects_path'], self.params['data']['direction']['target'], f"{self.test_subset[i]}.nii.gz")),
                mask   = tio.LabelMap(os.path.join(self.params['data']['training']['subjects_path'], self.params['data']['direction']['mask_dir_name'], f"{self.test_subset[i]}.nii.gz"))
            )
            # print("name", self.test_subset[0].split('.')[0])
            self.sampler = tio.data.GridSampler(
                        subject=self.test_subject_tio,
                        patch_size=self.params['patchify']['patch_size'],
                        patch_overlap=self.params['patchify']['patch_overlap'],
                        # patch_overlap=0,
                        
            )
            self.dataset_object_test = CachedTorchIODataset(
                subject_list=[self.test_subject_tio],
                # transform=tio.Compose([
                #     tio.RescaleIntensity(out_min_max=(0, 1), percentiles=tuple(self.params['intensity_augmentations']['mri_intensity_percentile'])),
                # ]),
                cache_enabled=False
            )
            num_patches = int(len(self.sampler))
            self.aggregator = tio.inference.GridAggregator(self.sampler , 'hann')
            self.patches_queue = tio.Queue(
                subjects_dataset=self.dataset_object_test,
                max_length=num_patches,                  # max # of patches in queue
                samples_per_volume=num_patches,          # how many patches to extract per volume
                sampler = self.sampler,
                num_workers=self.params['training']['num_workers'],
                shuffle_subjects=False,
                shuffle_patches=False
            )
            self.test_loaders_list.append({'queue':  tio.SubjectsLoader(self.patches_queue, batch_size=self.params['training']['batch_size']),
                            'aggregator': self.aggregator, 
                            'original' : self.dataset_object_test # return the original subject for saving
                            })
            
        
    def fetch_allloaders(self, config):
        """Fetches the training, validation, and test subsets as DataLoaders.
        """
        self.batch_size = self.params['training']['batch_size']

        if config['training']['epochs']>0:
            return {
                'train':        DataLoader(CachingPatchDataset(self.params, self.train_subset,   self.device ), batch_size=self.batch_size, shuffle=True),
                'validation':   DataLoader(CachingPatchDataset(self.params, self.val_subset,     self.device),  batch_size=self.batch_size, shuffle=False),
                'test':        self.test_loaders_list
            }
        else:
             return {
                'train':        None,
                'validation':   None,
                'test':        self.test_loaders_list
            }           

    
if __name__ == "__main__":
    sample_index = 4
    with open('/Drive4T/inam/3T_7T_Synthesis/configs/VNET_Paper_precompute.json', 'r') as f:
        config = json.load(f)
    for k, v in config.items():
        print(f"{k}: {v}")
    cached_dataloader = cache_patch_dataloader(config, device='cuda:0', idx=sample_index).fetch_allloaders()
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
    
    
