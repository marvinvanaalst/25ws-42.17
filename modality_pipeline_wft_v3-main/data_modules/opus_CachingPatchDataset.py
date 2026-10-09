import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import json
from torch.utils.data import Subset, DataLoader
import torchio as tio
import random

SPATIAL = (1, 2, 3)

def random_flip_pair(src, tgt, mask, p=0.5):
    dims = [d for d in SPATIAL if torch.rand(()) < p]
    if dims:
        src = torch.flip(src, dims)
        tgt = torch.flip(tgt, dims)
        mask = torch.flip(mask, dims)
    return src, tgt, mask


class CachingPatchDataset(Dataset):
    """
    Patch dataset with three externally selected caching strategies, chosen by
    config["device"]["cache_device"]:

        "same_gpu" : preload every patch onto the training GPU (self.device).
                     Fastest at train time, costs GPU memory, and REQUIRES
                     DataLoader(num_workers=0, pin_memory=False) — CUDA tensors
                     do not survive the fork into worker processes and cannot
                     be pinned.

        "ram"      : preload every patch into CPU RAM. Workers hand CPU tensors
                     to the training loop, which does the .to(device) move.
                     Works with num_workers>0 and pin_memory=True.

        "none"     : store only file paths; each patch is torch.load-ed to CPU
                     on demand inside __getitem__ (i.e. by the DataLoader worker).
                     Lowest memory, highest per-step I/O.
    """

    VALID_MODES = ("same_gpu", "ram", "none")

    def __init__(self, config, sample_list, device):
        self.config = config
        self.device = device  # the training device, e.g. "cuda:0"

        # ---- the single external switch ------------------------------------
        cache_mode = config["device"].get("cache_stratergy", "same_gpu")
        self.tta_flips = config.get("tta_flips", False)
        if self.tta_flips:
            self.tta_flips = config["tta_flips"].get("enable", False)
            self.tta_flips_p = config["tta_flips"].get("p", 0.5) # dead variable, same always. Uniform over all flips. Makes sense since output considers all flips unformly.
            
        
        if cache_mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid cache_device={cache_mode!r}. Use one of {self.VALID_MODES}."
            )
        self.cache_mode = cache_mode

        # Where load-time storage should live. "ram" and "none" both target CPU;
        # only "same_gpu" targets the actual training device.
        self.cache_target = device if cache_mode == "same_gpu" else "cpu"

        # Whether tensors are preloaded (True) or paths are indexed lazily (False).
        self.built_cache = cache_mode in ("same_gpu", "ram")
        self.cache = []  # holds tensor-dicts (preloaded) OR path-dicts (lazy)

        # ---- resolve directories -------------------------------------------
        patch_size = config["patchify"]["patch_size"]
        patch_overlap = config["patchify"]["patch_overlap"]

        if 'intensity_augmentations' not in config:
            config['intensity_augmentations'] = {}
            config['intensity_augmentations']['mri_intensity_percentile'] = [0, 100]
        intensity = config['intensity_augmentations']['mri_intensity_percentile'][1]
        print("Target Scan Intensity =", intensity)

        data_dir = os.path.join(
            config["data"]["training"]["subjects_path"],
            "patches",
            f"{patch_size}X{patch_overlap}_intensity={intensity}",
        )
        assert os.path.exists(data_dir), f"Patch dir not found: {data_dir}"

        source_dir = os.path.join(data_dir, config["data"]["direction"]["source"])
        target_dir = os.path.join(data_dir, config["data"]["direction"]["target"])
        mask_dir   = os.path.join(data_dir, config["data"]["direction"]["mask_dir_name"])

        # ---- build cache or index ------------------------------------------
        if self.built_cache:
            self._build_cache(source_dir, target_dir, mask_dir, sample_list,
                              device=self.cache_target)
        else:
            self._build_index(source_dir, target_dir, mask_dir, sample_list)

    def _iter_patch_files(self, source_dir, target_dir, mask_dir, sample_list):
        """Yield (sample_name, pt_file, source_file, target_file, mask_file)."""
        for sample_name in sample_list:
            source_path = os.path.join(source_dir, sample_name)
            target_path = os.path.join(target_dir, sample_name)
            mask_path   = os.path.join(mask_dir,   sample_name)
            for pt_file in sorted(os.listdir(source_path)):
                yield (
                    sample_name,
                    pt_file,
                    os.path.join(source_path, pt_file),
                    os.path.join(target_path, pt_file),
                    os.path.join(mask_path,   pt_file),
                )

    def _build_cache(self, source_dir, target_dir, mask_dir, sample_list, device):
        """Preload every patch onto `device` (cpu for RAM mode, cuda for GPU mode)."""
        items = list(self._iter_patch_files(source_dir, target_dir, mask_dir, sample_list))
        for sample_name, pt_file, source_file, target_file, mask_file in tqdm(
            items, desc=f"[{self.cache_mode.upper()}] caching", unit="patch"
        ):
            source_tensor = torch.load(source_file, map_location=device, weights_only=False)
            target_tensor = torch.load(target_file, map_location=device, weights_only=False)
            mask_tensor   = torch.load(mask_file, map_location=device, weights_only=False)

            assert source_tensor.keys() == target_tensor.keys() == mask_tensor.keys(), \
                f"[Key Mismatch] {pt_file} in sample {sample_name}"
            
            source =  source_tensor["patch"].to(torch.float16)
            target =  target_tensor["patch"].to(torch.float16)
            mask = mask_tensor["patch"].to(torch.float16)
            v_min_source = torch.as_tensor(source_tensor["v_min"], dtype=torch.float16, device=device)
            v_min_target = torch.as_tensor(target_tensor["v_min"], dtype=torch.float16, device=device)
            v_max_source = torch.as_tensor(source_tensor["v_max"], dtype=torch.float16, device=device)
            v_max_target = torch.as_tensor(target_tensor["v_max"], dtype=torch.float16, device=device)
            self.cache.append({
                "source": source,
                "target": target,
                "mask": mask  ,
                "v_min_source" : v_min_source,
                "v_min_target": v_min_target,
                "v_max_source" : v_max_source,
                "v_max_target": v_max_target
            })

    def _build_index(self, source_dir, target_dir, mask_dir, sample_list):
        """Store only paths; tensors are loaded lazily in __getitem__."""
        for _, _, source_file, target_file, mask_file in self._iter_patch_files(
            source_dir, target_dir, mask_dir, sample_list
        ):
            self.cache.append({
                "source": source_file,
                "target": target_file,
                "mask":   mask_file,
            })

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        if self.built_cache:
            if self.tta_flips:
                sample = self.cache[idx]
                source, target, mask = random_flip_pair(sample["source"], sample["target"], sample["mask"])
                return {
                    "source": source,
                    "target" : target,
                    "mask" : mask,
                    "v_min_source" : sample["v_min_source"],
                    "v_min_target": sample["v_min_target"],
                    "v_max_source" : sample["v_max_source"],
                    "v_max_target": sample["v_max_target"]
                }
                
            return self.cache[idx]

        # Lazy path: always load to CPU here so num_workers>0 stays safe.
        entry = self.cache[idx]
        
        source_tensor = torch.load(entry["source"], map_location="cpu", weights_only=False)
        target_tensor = torch.load(entry["target"], map_location="cpu", weights_only=False)
        mask_tensor   = torch.load(entry["mask"], map_location="cpu", weights_only=False)

        v_min_source = source_tensor["v_min"]
        v_min_target = target_tensor["v_min"]
        v_max_source = source_tensor["v_max"]
        v_max_target = target_tensor["v_max"]
        
        source_tensor = source_tensor["patch"]
        target_tensor = target_tensor["patch"]
        mask_tensor   = mask_tensor["patch"]

        if self.tta_flips: #Checked for correctneesssssssss
            dims = [d for d in SPATIAL if torch.rand(()) < self.tta_flips_p ]
            if dims:
                source_tensor = source_tensor.flip(dims)
                target_tensor = target_tensor.flip(dims)
                mask_tensor= mask_tensor.flip(dims)
        return {
            "source": source_tensor.to(torch.float16),
            "target": target_tensor.to(torch.float16),
            "mask":   mask_tensor.to(torch.float16),
            "v_min_source" : torch.as_tensor(v_min_source, dtype=torch.float16),
            "v_min_target": torch.as_tensor(v_min_target, dtype=torch.float16),
            "v_max_source" : torch.as_tensor(v_max_source, dtype=torch.float16),
            "v_max_target": torch.as_tensor(v_max_target, dtype=torch.float16)
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
            self.test_loaders_list.append(self.create_test_sample(self.test_subset[i]))
            
        self.val_loaders_list = []
        for i in range(len(self.val_subset)):
            self.val_loaders_list.append(self.create_test_sample(self.val_subset[i]))
            break # Just 1 sample
        
        
    def fetch_allloaders(self, config):
        """Fetches the training, validation, and test subsets as DataLoaders.
        """
        self.batch_size = self.params['training']['batch_size']

        if config['training']['epochs']>0:
            return {
                'train':        DataLoader(CachingPatchDataset(self.params, self.train_subset,   self.device ), batch_size=self.batch_size, shuffle=True),
                'validation':   DataLoader(CachingPatchDataset(self.params, self.val_subset,     self.device),  batch_size=self.batch_size, shuffle=False),
                'test':        self.test_loaders_list,
                'intermediate_testing': self.val_loaders_list
            }
        else:
             return {
                'train':        None,
                'validation':   None,
                'test':        self.test_loaders_list,
                'intermediate_testing': None
                
            }           
    def create_test_sample(self, sample):
                    # print("Test Sample", os.path.join(self.params['data']['training']['subjects_path'], self.params['data']['direction']['source'], f"{self.test_subset[i]}.nii.gz"))
            self.test_subject_tio = tio.Subject(
                name= sample.split('.')[0],
                source = tio.ScalarImage(os.path.join(self.params['data']['training']['subjects_path'], self.params['data']['direction']['source'], f"{sample}.nii.gz")),
                target = tio.ScalarImage(os.path.join(self.params['data']['training']['subjects_path'], self.params['data']['direction']['target'], f"{sample}.nii.gz")),
                mask   = tio.LabelMap(os.path.join(self.params['data']['training']['subjects_path'], self.params['data']['direction']['mask_dir_name'], f"{sample}.nii.gz"))
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
                transform=tio.Compose([
                    tio.RescaleIntensity(out_min_max=(-1, 1), percentiles=tuple(self.params['intensity_augmentations']['mri_intensity_percentile']),  include=['source']),
                    tio.RescaleIntensity(out_min_max=(0, 1), percentiles=tuple(self.params['intensity_augmentations']['mri_intensity_percentile']),  include=['target']),
                ]),
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
            return {'queue':  tio.SubjectsLoader(self.patches_queue, batch_size=self.params['training']['batch_size'], num_workers=0),
                            'aggregator': self.aggregator, 
                            'original' : self.dataset_object_test # return the original subject for saving
                            }
        
        
