import os
import json
import pytorch_lightning as pl
from torch.utils.data import random_split

import numpy as np
# editted cd /home/an_inam/miniconda3/envs/MICC/lib/python3.11/site-packages/torchio/data/

# import monai.data.Dataloader as monaiDataLoader
import torch

import torchio as tio
import random
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm


class tioPatchLoader(pl.LightningDataModule):
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
        
        rescale01 = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=tuple(self.params['intensity_augmentations']['mri_intensity_percentile']))
        
        self.subjects_list = sorted(self.subjects_list)
        for ldx, fname in enumerate(tqdm((self.subjects_list), desc="Loading subjects", unit="subject")):
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
                    # name=fname,
                    source = rescale01(tio.ScalarImage(source_path)),
                    target = rescale01(tio.ScalarImage(target_path)),
                    mask =  tio.LabelMap(mask_path)
            )
            # item = tio.Subject({
            #         # name=fname,
            #         'source' : rescale01(tio.ScalarImage(source_path)),
            #         'target':= rescale01(tio.ScalarImage(target_path)),
            #         'mask' :tio.LabelMap(mask_path)
            #     }
            # )
            data_dicts.append(item)
        preprocessing = tio.Compose([
                # tio.Resample(1),                       # resample to 1mm isotropic spacing
                # tio.CropOrPad((128, 128, 128)),        # crop or pad to fixed patch size
                # tio.ZNormalization(),                  # zero-mean, unit-std (per image)
                # tio.EnsureShapeMultiple(8),            # pad shape to be divisible by 8 (for UNets etc.)
            ])

        # Step 2: define transforms — MONAI expects key-mapped dictionary transforms

        test_ds = [data_dicts[idx]]
        train_ds = [d for i, d in enumerate(data_dicts) if i != idx]
        num_samples = len(train_ds)
        all_indices = list(range(num_samples))
        random.shuffle(all_indices)
        val_indices = all_indices[:2]
        train_indices = all_indices[2:]
        
        train_subset = Subset(train_ds, train_indices)
        val_subset   = Subset(train_ds, val_indices)

        print(f"\nTraining samples: {len(train_subset)}, \nValidation samples: {len(val_subset)}, \nTest samples: {len(test_ds)}")
        # print(f"\nTest samples: {test_ds[0]['source'].split('/')[-1]}, \n")
        # input(self.params)
        # Step 3: create dataset (CacheDataset is fast for small/medium datasets)
        self.train_ds =  tio.SubjectsDataset(train_subset, transform=preprocessing)
        self.val_ds =  tio.SubjectsDataset(val_subset, transform=preprocessing)
        self.test_ds =  tio.SubjectsDataset(test_ds, transform=preprocessing)     
        # samplerR =    tio.GridSampler(patch_size=self.training_params['training']['patch_size'], patch_overlap=self.training_params['training']['patch_size'])
        self.sampler = tio.data.GridSampler(
                            subject=self.train_ds[0],
                            patch_size=self.training_params['training']['patch_size'],
                            patch_overlap=self.training_params['training']['patch_overlap'],
        )
        # for i, subject in enumerate(self.train_ds):
        #     print(f"Subject {i}: {subject.name}")
        #     print(f"Source shape: {subject['source'][tio.DATA].shape}")
        #     print(f"Target shape: {subject['target'][tio.DATA].shape}")
        #     print(f"Mask shape: {subject['mask'][tio.DATA].shape}")
        num_patches = int(len(self.sampler))
        # input(num_patches)
        self.aggregator = tio.inference.GridAggregator(self.sampler, 'hann')
        
        self.training_patches = tio.Queue(
            subjects_dataset=self.train_ds,
            max_length=50,                  # max # of patches in queue
            samples_per_volume=num_patches,          # how many patches to extract per volume
            sampler = self.sampler,
            # num_workers=self.training_params['training']['num_workers'],
            shuffle_subjects=True,
            shuffle_patches=True
        )
        self.validation_patches = tio.Queue(
            subjects_dataset=self.val_ds,
            max_length=50,                  # max # of patches in queue
            samples_per_volume=num_patches,          # how many patches to extract per volume
            sampler = self.sampler,
            # num_workers=self.training_params['training']['num_workers'],
            shuffle_subjects=True,
            shuffle_patches=False
        )
        # self.testing_patches = tio.Queue(
        #     subjects_dataset=self.test_ds,
        #     max_length=100,                  # max # of patches in queue
        #     samples_per_volume=num_patches,          # how many patches to extract per volume
        #     sampler = sampler,
        #     # num_workers=self.training_params['training']['num_workers'],
        #     shuffle_subjects=True,
        #     shuffle_patches=False
        # )
                
# Create patch dataset
        # self.train_ds = PatchDataset(data=train_subset, patch_func=sampler)
        # self.val_ds = PatchDataset(data=val_subset, patch_func=sampler)
        # self.test_ds = PatchDataset(data=test_ds, patch_func=sampler)
        
        
        # self.train_ds = SmartCacheDataset(data=patch_ds_train, transform=self.train_val_transforms)
        # self.val_ds = SmartCacheDataset(data=patch_ds_val, transform=self.train_val_transforms)
        # self.test_ds = SmartCacheDataset(data=patch_ds_test, transform=self.test_transforms)
        

        
        # self.train_ds = monaiDataLoader(data=train_subset, transform=self.train_val_transforms, num_workers=4)
        # self.val_ds = monaiDataLoader(data=val_subset, transform=self.train_val_transforms, num_workers=4)
        # self.test_ds = monaiDataLoader(data=test_ds, transform=self.test_transforms, num_workers=4)
        
        
        # self.train_ds = CacheDataset(data=train_subset, transform=self.train_val_transforms, num_workers=4)
        # self.val_ds = CacheDataset(data=val_subset, transform=self.train_val_transforms, num_workers=4)
        # self.test_ds = CacheDataset(data=test_ds, transform=self.test_transforms, num_workers=4)
        
        # self.full_dataset = Dataset(data=data_dicts, transform=self.transforms)

        # Step 4: split


    def setup(self, stage=None):
        pass
    def train_dataloader(self):
        # return SubjectDataLoader(self.training_patches,
        return  DataLoader(self.test_ds,    
                                 
                          batch_size=self.training_params['training']['batch_size'],
                          shuffle=True,
                          num_workers=self.training_params['training']['num_workers'],
                          )

    def val_dataloader(self):
        # return SubjectDataLoader(self.validation_patches,
        return  DataLoader(self.test_ds,    
                                 
                          batch_size=self.training_params['training']['batch_size'],
                          shuffle=False,
                          num_workers=self.training_params['training']['num_workers'],
                          )
    def predict_dataloader(self):
        for ele in self.test_ds:
            print(ele)
            for key, value in ele.items():
                print(f"{key}: {value.shape}")
                print(ele['source'].shape)
            input("Next batch?")
        # return  SubjectDataLoader(self.test_ds,
        return  DataLoader(self.test_ds,    
                          batch_size=self.training_params['training']['batch_size'],
                          shuffle=False,
                          num_workers=self.training_params['training']['num_workers'],
                          )
    def fetch_sampler_aggregator_sampler(self):
        return self.sampler, self.aggregator
# Debug/test hook
if __name__ == "__main__":
    with open('/Drive4T/inam/3T_7T_Synthesis/data.json', 'r') as f:
        params_data = json.load(f)
    with open('/Drive4T/inam/3T_7T_Synthesis/configs/VNET_Paper.json', 'r') as f:
        params_training = json.load(f)

    dm = tioPatchLoader(params_data, params_training)
    dm.setup()
    loader = dm.train_dataloader()
    for i, batch in enumerate(loader):
        print(batch.keys())
        print(f"Batch {i}:, {len(loader)}")
        
        for k, v in batch['source'].items():
            print(f"  {k}: {v}")
        print("len path, ", len(batch['source']['path']))
        print("affine len, ", len(batch['source']['affine']))
        
        input()
        print(f"  Source shape: {batch['source'][tio.DATA].shape}")
        print(f"  Target shape: {batch['target'][tio.DATA].shape}")
        print(f"  Mask shape: {batch['mask'][tio.DATA].shape}")
        # if i == 2:
        #     break
    # print(f"Total dataset size: {len(loader)}")
    # input("Len Dataset")
    # print("Type", type(loader))
    # for i in range(0, len(loader), 30):  # print every 30th sample
    #     sample = loader[i]
    #     print(f"\nSample {i}")
    #     print(f"  Image shape: {sample['image'].shape}")
    #     print(f"  Label shape: {sample['label'].shape}")
    # for eleA in loader:
    #     print("Type", type(eleA))
    #     print(len(eleA))
    #     for ele in eleA:
    #         print(ele.keys())
    #         for key, value in ele.items():
    #             print(f"{key}: {value.shape}")
    #         input("Next batch?")
    # azithro = []
    # for _ in range(15):
    #     for batch in loader:
    #         # print(batch)
    #         # input(type(batch))
    #         azithro.append(batch['source'])
    #         print(batch['source'].shape)
    #         # break
    # print(torch.all(azithro[0]) == torch.all(azithro[1]) == torch.all(azithro[2]))
    
    # if torch.all(azithro[0] == azithro[1]):
    #     print("Same sampling in batch")
    # else:
    #     print("Different sampling in batch")
    # print("DIFF INDICES")
    # diff_indices = torch.nonzero(azithro[0] != azithro[1])
    # print(diff_indices)
    
    # for i in range(len(azithro)):
    #     for j in range(i + 1, len(azithro)):
    #         # Check if two tensors are exactly equal (same shape + same values)
    #         print(f"Comparing {i} and {j}")
    #         print(azithro[i].shape, azithro[j].shape)
    #         if torch.equal(azithro[i], azithro[j]):
    #             print("Breaking the habit")
    #             break  # Break inner loop
    #     else:
    #         continue  # Only triggered if inner loop didn't break
    #     break        # If inner loop broke, exit outer loop too


# [in] [imp ] The batch size here referes to the number of samples being loadeed. Then the fucntion creates the patches from the samples.