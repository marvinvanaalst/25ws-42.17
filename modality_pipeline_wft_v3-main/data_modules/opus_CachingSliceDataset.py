import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import json
from torch.utils.data import Subset, DataLoader
import torchio as tio
from sklearn.model_selection import train_test_split

import random

# ---------------------------------------------------------------------------
# Cache strategies (`device.cache_stratergy` in the experiment config)
# ---------------------------------------------------------------------------
#   'same_gpu' : eagerly torch.load every slice straight onto the training GPU.
#                Fastest per step; costs VRAM proportional to the dataset.
#   'ram'      : eagerly torch.load every slice into host memory and copy each
#                batch to the GPU in the training loop (pinned + non_blocking).
#   'none'     : cache nothing — keep the file list and torch.load each slice on
#                demand in __getitem__, from worker processes when configured.
# Anything else falls back to 'same_gpu' with a warning.
CACHE_MODE_ALIASES = {
    'same_gpu': 'same_gpu', 'gpu': 'same_gpu', 'cuda': 'same_gpu',
    'ram': 'ram', 'cpu': 'ram', 'host': 'ram',
    'none': 'none', 'lazy': 'none', 'disk': 'none',
}


def resolve_cache_mode(config):
    """Canonical cache strategy for `config`, or 'same_gpu' if it is unreadable."""
    raw = config.get('device', {}).get('cache_stratergy', 'same_gpu')
    mode = CACHE_MODE_ALIASES.get(str(raw).strip().lower())
    if mode is None:
        print(f"[Cache] Unknown cache_stratergy {raw!r} — falling back to 'same_gpu'. "
              f"Valid values: {sorted(set(CACHE_MODE_ALIASES.values()))}.")
        mode = 'same_gpu'
    return mode


class CachingSliceDataset(Dataset):
    def __init__(self, config, sample_list, device, data_grouping='slices', type_load='all',
                 affine=None, cache_mode=None):
        """Pre-computed slice dataset with a configurable cache location.

        Args:
            config (dict): the experiment config.
            sample_list (List[str]): sample folder names to load.
            device (str): 'cuda[:x]' or 'cpu' — the *training* device.
            cache_mode (str): 'same_gpu' | 'ram' | 'none'. Defaults to whatever
                `device.cache_stratergy` says (see `resolve_cache_mode`).
        """
        if not (device.startswith("cuda") or device.startswith("cpu")):
            raise ValueError(f"Invalid device: {device}. Use 'cuda[:x]' or 'cpu'.")

        self.config = config
        self.device = device
        self.cache_mode = cache_mode or resolve_cache_mode(config)
        # Only 'same_gpu' loads onto the training device; the other two stay on
        # the host and let the training loop do the copy.
        self.map_location = device if self.cache_mode == 'same_gpu' else 'cpu'
        self.complete_cache = self.cache_mode != 'none'

        self.cache = []        # materialised samples ('same_gpu' / 'ram' only)
        self.slice_files = []  # (source, target, mask) triples — always built

        slice_var = f"{self.config['slicify']['resize_to']}x{self.config['slicify']['pad_to']}"
        data_dir = f"{self.config['data']['training']['subjects_path']}/{data_grouping}/"
        source_dir = os.path.join(data_dir,slice_var, f"{self.config['data']['direction']['source']}/{type_load}")
        target_dir = os.path.join(data_dir,slice_var, f"{self.config['data']['direction']['target']}/{type_load}")
        mask_dir   = os.path.join(data_dir,slice_var, f"{self.config['data']['direction']['mask_dir_name']}/{type_load}")

        self._index_files(source_dir, target_dir, mask_dir, sample_list)
        if self.complete_cache:
            self._build_cache()

    def _index_files(self, source_dir, target_dir, mask_dir, sample_list):
        """Collect the (source, target, mask) triples that exist on all three sides."""
        for sample_name in sample_list:
            source_path = os.path.join(source_dir, sample_name)
            target_path = os.path.join(target_dir, sample_name)
            mask_path   = os.path.join(mask_dir,   sample_name)

            pt_files = sorted(os.listdir(source_path))

            for pt_file in pt_files:
                source_file = os.path.join(source_path, pt_file)
                target_file = os.path.join(target_path, pt_file)
                mask_file   = os.path.join(mask_path,   pt_file)

                if os.path.isfile(source_file) and os.path.isfile(target_file) and os.path.isfile(mask_file):
                    self.slice_files.append((source_file, target_file, mask_file))

    def _load(self, idx):
        """Read one slice triple off disk into a batch dict."""
        source_file, target_file, mask_file = self.slice_files[idx]

        source_tensor = torch.load(source_file, map_location=self.map_location)
        target_tensor = torch.load(target_file, map_location=self.map_location)
        mask_tensor   = torch.load(mask_file,   map_location=self.map_location)

        assert source_tensor.keys() == target_tensor.keys() == mask_tensor.keys(), \
            f"[Key Mismatch] {source_file}"

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
            out_dict['target_affine'] = target_tensor['affine']

        return out_dict

    def _build_cache(self):
        loading_loop = tqdm(range(len(self.slice_files)),
                            desc=f"[{self.cache_mode.upper()}] Caching", unit="slice")
        for idx in loading_loop:
            self.cache.append(self._load(idx))

    def __len__(self):
        return len(self.cache) if self.complete_cache else len(self.slice_files)

    def __getitem__(self, idx):
        return self.cache[idx] if self.complete_cache else self._load(idx)

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
    def __init__(self, params_data, device, idx=0, cache_mode=None):
        """`cache_mode` pins the strategy, ignoring `device.cache_stratergy`.

        Callers that consume `batch['source']` directly (rather than moving it
        with `.to(DEVICE)`) must pin 'same_gpu' — anything else hands them host
        tensors and the model call fails.
        """
        super().__init__()
        self.device = device
        self._cache_mode_override = cache_mode
        assert device.startswith("cuda") or device.startswith("cpu")
        random.seed(42) 

        self.params = params_data
            
        self.data_dir = self.params['data']['training']['subjects_path']
        
        num_test_samples = self.params['training']['splits'].get('num_test_samples',1)
        test_samples_floor = idx*num_test_samples
        test_sample_ceil = test_samples_floor + num_test_samples

        self.subjects_list = [ele.split('.')[0] for ele in self.params['data']['training']['subjects']] # Only the subject name.
        self.subjects_list = sorted(self.subjects_list)
        random.shuffle(self.subjects_list)
        self.test_subset = self.subjects_list[test_samples_floor:test_sample_ceil]

        train_val_samples = [s for i, s in enumerate(self.subjects_list) if s not in self.test_subset]


        num_samples_validation = self.params['training']['splits']['validation_samples']
        assert type(num_samples_validation) == int

        num_samples = len(train_val_samples)
        all_indices = list(range(num_samples))
        random.shuffle(all_indices)
        # Validation is number of samples based.
        if num_samples_validation == 0:
            val_indices = []
        else:
            val_indices = all_indices[:num_samples_validation]
        train_indices = all_indices[num_samples_validation:]
        self.train_subset = Subset(train_val_samples, train_indices)
        self.val_subset   = Subset(train_val_samples, val_indices)




        if self.params['fine_tunning'].get('pretrain_boolean', None) == True:
            if self.params['fine_tunning'].get('pretrain_evaluate_only', None)== True:
                self.test_subset = [*sorted(self.subjects_list)]  # Use all subjects for testing
                self.train_subset = [self.subjects_list[0]] # Using so as not to break the code. Not used down the line
                self.val_subset = [self.subjects_list[0]]
                print(f"Running in pretrained evaluation mode. Evaluating on all {len(self.test_subset)} samples. Press enter to continue")
                
        print("\n\nLoading Data to memory. Data Set Configuration: \n idx==",idx)
        print("Testing Set    --  ", [ele for ele in self.test_subset])
        print("Validation Set --  ", [ele for ele in self.val_subset])
        print("Training Set   --  ", [ele for ele in self.train_subset])

        # batch_size is needed by fetch_testLoader / fetch_intermediateLoader even
        # when fetch_TrainValLoaders was never called (epochs == 0 runs).
        self.batch_size = self.params['training']['batch_size']
           
        # # print("\033\n[91mTesting Sample", os.path.join(self.params['data']['training']['subjects_path'], self.params['data']['direction']['source'], f"{self.test_subset[0]}.nii.gz \n\033[0m"))
        # print(f"Train subset: {len(self.train_subset)} samples, Val subset: {len(self.val_subset)} samples, Test subset: {len(self.test_subset)} samples")
        # print(f"Train subset 5 samples\n: {self.train_subset[:5]}...")  # Show first 5 samples for brevity
        # print(f"\033Val subset 5\033[0m\n samples: {self.val_subset[:5]}...")  # Show first 5 samples for brevity
        # print(f"\033Test subset all sample\033[0m\n s: {self.test_subset}")  # Show first 5 samples for brevity
        

        # -- cache strategy -------------------------------------------------
        self.cache_mode = self._cache_mode_override or resolve_cache_mode(self.params)
        # Workers only pay off when __getitem__ actually touches the disk, and
        # CUDA tensors cannot cross a fork — so only 'none' gets subprocesses.
        self.num_workers = int(self.params['device'].get('num_workers', 4)) if self.cache_mode == 'none' else 0
        # Pinning is what makes `.to(DEVICE, non_blocking=True)` in the training
        # loop an async copy; pointless when the tensors are already on the GPU.
        self.pin_memory = self.cache_mode != 'same_gpu' and self.device.startswith('cuda')
        print(f"[Cache] strategy={self.cache_mode} | device={self.device} | "
              f"num_workers={self.num_workers} | pin_memory={self.pin_memory}")

    def _make_dataset(self, sample_list, type_load, affine=None):
        return CachingSliceDataset(self.params, sample_list, self.device,
                                   type_load=type_load, affine=affine,
                                   cache_mode=self.cache_mode)

    def _make_loader(self, dataset, shuffle, persistent=False):
        """DataLoader wired for the active cache strategy.

        `persistent` is only for the long-lived train/val loaders — the test and
        intermediate loaders are rebuilt per subject, so keeping their workers
        alive would just pile up idle processes.
        """
        # An empty split (e.g. validation_samples == 0) has nothing for workers
        # to do — spawning them would only leave idle processes behind.
        num_workers = self.num_workers if len(dataset) else 0
        kwargs = dict(batch_size=self.batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=self.pin_memory)
        if num_workers > 0:
            kwargs['prefetch_factor'] = 4
            kwargs['persistent_workers'] = persistent
        return DataLoader(dataset, **kwargs)

    def fetch_TrainValLoaders(self, build_train=True):
        """Fetches the training and validation subsets as DataLoaders.

        build_train=False skips caching the train/val slices altogether, which
        matters for evaluation-only runs (epochs == 0) where loading every
        training slice onto the GPU is pure waste.
        """
        self.batch_size = self.params['training']['batch_size']

        if not build_train:
            return {
                'train': [],
                'validation': [],
                'number_test_samples': int(len(self.test_subset)),
            }

        return {
            'train':        self._make_loader(self._make_dataset(self.train_subset, 'non_empty'), shuffle=True,  persistent=True),
            'validation':   self._make_loader(self._make_dataset(self.val_subset,   'non_empty'), shuffle=False, persistent=True),
            'number_test_samples': int(len(self.test_subset)),
        }

    def fetch_intermediateLoader(self, idx=0):
        """A full-volume ('all' slices) loader over a *validation* subject.

        Used for the periodic `sample_volume_during_training` reconstruction so
        that intermediate visual checks never touch the held-out test subjects.
        Returns None when no validation subject was allocated.
        """
        val_names = [ele for ele in self.val_subset]
        if not val_names:
            return None
        idx = idx % len(val_names)
        return {
            'test': self._make_loader(self._make_dataset([val_names[idx]], 'all', affine=True), shuffle=False),
            'test_name': val_names[idx],
        }

    def fetch_testLoader(self, idx=0):
        """Fetches the test subset as a DataLoader.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            DataLoader: DataLoader for the test subset.
        """
        return {
            'test' : self._make_loader(self._make_dataset([self.test_subset[idx]], 'all', affine=True), shuffle=False),
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
    
    
