import os
import json
import pytorch_lightning as pl
from torch.utils.data import random_split
from monai.data import CacheDataset, Dataset, SmartCacheDataset,DataLoader

from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, ConcatItemsd, CropForegroundd,
    RandCropByPosNegLabeld, EnsureTyped, Compose,ScaleIntensityRangePercentilesd
)

# import monai.data.Dataloader as monaiDataLoader
import torch

from monai.data import PatchDataset
from monai.transforms import RandSpatialCropSamplesd

import random
from torch.utils.data import Subset

class patchDataModule(pl.LightningDataModule):
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
        self.subjects_list = sorted(self.subjects_list)
        for ldx, fname in enumerate(self.subjects_list):
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
            
            item = {"source": source_path, "target": target_path, "mask": mask_path}
            data_dicts.append(item)

        # Step 2: define transforms — MONAI expects key-mapped dictionary transforms


        self.train_val_transforms = Compose([
            LoadImaged(keys=["source", "target", "mask"]),
            EnsureChannelFirstd(keys=["source", "target", "mask"]),

            Orientationd(keys=["source", "target", "mask"], axcodes="RAS"),
            ScaleIntensityRangePercentilesd(keys=["source", "target"], lower=0, upper=100, b_min=0.0, b_max=1.0, clip=True),

            RandCropByPosNegLabeld(
                keys=["source", "target","mask"],
                label_key="mask",
                spatial_size=self.training_params['training']['patch_size'],  # patch size from training params
                pos=1,neg=0.1,
                num_samples=self.training_params['training']['samples_per_sample'],  # this gives 4 patches per volume
                image_key="source",
                image_threshold=0,
            ),

            EnsureTyped(keys=["source", "target", "mask"]),
        ])
        self.test_transforms = Compose([
            LoadImaged(keys=["source", "target"], image_only=False),
            LoadImaged(keys=["mask"], image_only=False),  # Load mask as image only
            EnsureChannelFirstd(keys=["source", "target", "mask"]),

            Orientationd(keys=["source", "target", "mask"], axcodes="RAS"),
            ScaleIntensityRangePercentilesd(keys=["source", "target"], lower=0, upper=100, b_min=0.0, b_max=1.0, clip=True),


            # EnsureTyped(keys=["source", "target", "mask"], dtype=float),
            
            # Delegating sampling to the PL Model Module
        ])

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
        print(f"\nTest samples: {test_ds[0]['source'].split('/')[-1]}, \n")

        # Step 3: create dataset (CacheDataset is fast for small/medium datasets)
        self.train_ds = CacheDataset(data=train_subset, transform=self.train_val_transforms, cache_rate=1.0)
        self.val_ds = CacheDataset(data=val_subset, transform=self.train_val_transforms, cache_rate=1.0)
        self.test_ds = CacheDataset(data=test_ds, transform=self.test_transforms, cache_rate=1.0)        
                
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
        return DataLoader(self.train_ds,
                          batch_size=self.training_params['training']['batch_size'],
                          shuffle=True,
                          num_workers=self.training_params['training']['num_workers'])

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.training_params['training']['batch_size'],
                          shuffle=False,
                          num_workers=self.training_params['training']['num_workers'])
    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.training_params['patch_inference']['batch_size'],
                          shuffle=False,
                          num_workers=self.training_params['training']['num_workers'])

# Debug/test hook
if __name__ == "__main__":
    with open('/storage/an_inam/MR2MR/patch_pipeline/data.json', 'r') as f:
        params_data = json.load(f)
    with open('/storage/an_inam/MR2MR/patch_pipeline/configs/VNET_custom_3D.json', 'r') as f:
        params_training = json.load(f)

    dm = patchDataModule(params_data, params_training)
    dm.setup()
    loader = dm.train_dataloader()
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
    azithro = []
    for _ in range(15):
        for batch in loader:
            # print(batch)
            # input(type(batch))
            azithro.append(batch['source'])
            print(batch['source'].shape)
            # break
    print(torch.all(azithro[0]) == torch.all(azithro[1]) == torch.all(azithro[2]))
    
    if torch.all(azithro[0] == azithro[1]):
        print("Same sampling in batch")
    else:
        print("Different sampling in batch")
    print("DIFF INDICES")
    diff_indices = torch.nonzero(azithro[0] != azithro[1])
    print(diff_indices)
    
    for i in range(len(azithro)):
        for j in range(i + 1, len(azithro)):
            # Check if two tensors are exactly equal (same shape + same values)
            print(f"Comparing {i} and {j}")
            print(azithro[i].shape, azithro[j].shape)
            if torch.equal(azithro[i], azithro[j]):
                print("Breaking the habit")
                break  # Break inner loop
        else:
            continue  # Only triggered if inner loop didn't break
        break        # If inner loop broke, exit outer loop too


# [in] [imp ] The batch size here referes to the number of samples being loadeed. Then the fucntion creates the patches from the samples.