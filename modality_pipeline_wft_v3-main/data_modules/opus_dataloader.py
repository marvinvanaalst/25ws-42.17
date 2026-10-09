import torchio as tio
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
import random

class tioSetup:
    def __init__(self, params_data, params_training, idx=0):
        super().__init__()
        self.params = params_data
        self.training_params = params_training
        self.subjects_list = self.params['data']['training']['subjects']
        self.data_dir = self.params['data']['training']['subjects_path']
        self.src_key = self.params['direction']['source']
        self.tgt_key = self.params['direction']['target']
                # Step 1: create list of dicts like [{'source': 'path', 'target': 'path', 'mask': 'path'}, ...]
        data_dicts = []
        
        rescale01 = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100))
        
        self.subjects_list = sorted(self.subjects_list)
        for ldx, fname in enumerate(tqdm((self.subjects_list), desc="Setting Up TIO Subjects", unit="subject")):
            source_path = os.path.join(self.data_dir, self.src_key, fname)
            target_path = os.path.join(self.data_dir, self.tgt_key, fname)
            mask_path = os.path.join(self.data_dir, "mask", fname)

            # Existence checks like your TorchIO version
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Missing source: {source_path}")
            if not os.path.exists(target_path):
                raise FileNotFoundError(f"Missing target: {target_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Missing target: {mask_path}")
            
            item = tio.Subject(
                    name=fname,
                    source = rescale01(tio.ScalarImage(source_path)),
                    target = rescale01(tio.ScalarImage(target_path)),
                    mask =  tio.LabelMap(mask_path)
            )

            data_dicts.append(item)
        preprocessing = tio.Compose([])

        self.test_subset = [data_dicts[idx]]
        train_ds = [d for i, d in enumerate(data_dicts) if i != idx]
        num_samples = len(train_ds)
        all_indices = list(range(num_samples))
        random.shuffle(all_indices)
        val_indices = all_indices[:2]
        print('bad Incidcies train')
        train_indices = all_indices[:2]
        
        self.train_subset = Subset(train_ds, train_indices)
        self.val_subset   = Subset(train_ds, val_indices)
        print(f"\nTraining samples: {len(self.train_subset)}, \nValidation samples: {len(self.val_subset)}, \nTest samples: {len(self.test_subset)}")
    
    def fetch_samples(self):
        """Fetches the training, validation, and test subsets.
        """    
        return self.train_subset, self.val_subset, self.test_subset
    
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

class patch_Dataloader():
    def __init__(self, dataset_object, params_data, params_training):
        self.dataset_object = dataset_object
        self.params_data = params_data
        self.training_params = params_training
        self.sampler = tio.data.GridSampler(
                    subject=self.dataset_object[0],
                    patch_size=self.training_params['training']['patch_size'],
                    patch_overlap=self.training_params['training']['patch_overlap'],
        )
        num_patches = int(len(self.sampler))
        # print(num_patches)
        # self.aggregator = tio.inference.GridAggregator(self.sampler , 'hann')
        self.patches_queue = tio.Queue(
            subjects_dataset=dataset_object,
            max_length=num_patches*10,                  # max # of patches in queue
            samples_per_volume=num_patches,          # how many patches to extract per volume
            sampler = self.sampler,
            num_workers=self.training_params['training']['num_workers'],
            shuffle_subjects=True,
            shuffle_patches=True
        )
    def fetch_loader(self):
        return tio.SubjectsLoader(
            self.patches_queue,
            batch_size=16,
            num_workers=0,  # this must be 0
        )

class patch_Dataloader_inference():
    def __init__(self, dataset_object, params_data, params_training):
        self.dataset_object = dataset_object
        self.params_data = params_data
        self.training_params = params_training
        self.sampler = tio.data.GridSampler(
                    subject=self.dataset_object[0],
                    patch_size=self.training_params['training']['patch_size'],
                    patch_overlap=self.training_params['training']['patch_overlap'],
        )
        num_patches = int(len(self.sampler))
        print(num_patches)
        self.aggregator = tio.inference.GridAggregator(self.sampler , 'hann')
        self.patches_queue = tio.Queue(
            subjects_dataset=dataset_object,
            max_length=num_patches*3,                  # max # of patches in queue
            samples_per_volume=num_patches,          # how many patches to extract per volume
            sampler = self.sampler,
            num_workers=self.training_params['training']['num_workers'],
            shuffle_subjects=True,
            shuffle_patches=True
        )
    def fetch_loader_and_sampler(self):
        return tio.SubjectsLoader(
            self.patches_queue,
            batch_size=16,
            num_workers=0,  # this must be 0
        ), self.aggregator, self.dataset_object # return the original subject for saving

def all_data_handler(params_data, params_training, idx=0):
    """Fetches the training, validation, and test subsets.
    """
    train_samples, val_samples, test_samples = tioSetup(params_data, params_training, idx).fetch_samples()
    
    # print("Left out sample", test_samples[0]['name'])
    train_dataset = CachedTorchIODataset(train_samples, transform=None, cache_enabled=True)
    val_dataset = CachedTorchIODataset(val_samples, transform=None, cache_enabled=True)
    test_dataset = CachedTorchIODataset(test_samples, transform=None, cache_enabled=True)
    
    train_dataloader = patch_Dataloader(train_dataset, params_data, params_training)
    val_dataloader = patch_Dataloader(val_dataset, params_data, params_training)
    test_dataloader = patch_Dataloader_inference(test_dataset, params_data, params_training)
    
    return train_dataloader, val_dataloader, test_dataloader

if __name__ == "__main__":
    import json
    with open('/Drive4T/inam/3T_7T_Synthesis/data.json', 'r') as f:
        params_data = json.load(f)
    with open('/Drive4T/inam/3T_7T_Synthesis/configs/VNET_Paper.json', 'r') as f:
        experiment = json.load(f)
        
    train_samples, val_samples, test_sampples = tioSetup(params_data, experiment, idx=0).fetch_samples()
    
    train_dataset = CachedTorchIODataset(train_samples, transform=None, cache_enabled=True)
    val_dataset = CachedTorchIODataset(val_samples, transform=None, cache_enabled=True)
    test_dataset = CachedTorchIODataset(test_sampples, transform=None, cache_enabled=True)
    
    train_dataloader = patch_Dataloader(train_dataset, params_data, experiment).fetch_loader()
    val_dataloader = patch_Dataloader(val_dataset, params_data, experiment).fetch_loader()
    test_dataloader, agg, orig = patch_Dataloader_inference(test_dataset, params_data, experiment).fetch_loader_and_sampler()
    
    # for idx, batch in enumerate(train_dataloader):
    #     print(f"Batch {idx+1}:")
    #     for k, v in batch.items():
    #         print(f"{k}")
    #         # if k != 'location':
    #         print(f"{v}")
    #     # print(batch)
    #     break
        
    for idx, batch in enumerate(test_dataloader):
        for xx in batch.keys():
            print(xx)
        # input()
        input, target = batch['source']['data'], batch['target']['data']
        print("location", batch['location'][0])
        agg.add_batch(batch['source']['data'], batch['location'])
        # print(f"Batch {idx+1}:")
    output = agg.get_output_tensor()
    affine = batch['source']['affine'][0]
    image = tio.ScalarImage(tensor=output, affine=affine)
    image.save(f"outxput_{idx}_16.nii.gz")
    print(orig[0])
    orig[0]['source'].save(f"orig_{idx}.nii.gz")
    
    
    
    # for idx, batch in enumerate(test_dataloader):
    #     print(f"Batch {idx+1}:")
    #     for k, v in batch.items():
    #         print(f"{k}")
    #         # if k != 'location':
    #         print(f"{v}")
    #     # print(batch)
    #     break