import os
import json
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from monai.data import CacheDataset, Dataset
from monai.transforms import (
    LoadImaged,
    # AddChanneld,
    EnsureChannelFirstd,
    ScaleIntensityRangePercentilesd,
    ResizeWithPadOrCropd,
    Compose,
    ToTensord,
    SqueezeDimd,
    ResizeD
)

from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)
from monai.config import print_config
import random
from torch.utils.data import Subset

class VolumeDataModuleMONAI(pl.LightningDataModule):
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
        self.transforms = Compose([
            LoadImaged(keys=["source", "target", "mask"]),
            EnsureChannelFirstd(keys=["source", "target", "mask"]),
            # ScaleIntensityRanged(
                #     keys=["source", "target"],   # which keys in the input dict to transform
                #     a_min=0, a_max=3000,         # input intensity range [a_min, a_max]
                #     b_min=0.0, b_max=1.0,        # output range [b_min, b_max]
                #     clip=True                    # clamp values to [b_min, b_max] after scaling
                # )
            ScaleIntensityRangePercentilesd(keys=["source", "target"], lower=0, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
            ResizeD(keys=["source", "target", "mask"], spatial_size=(256, 256, 256), mode="trilinear",  align_corners=True ),
            # ResizeWithPadOrCropd(keys=["source", "target", "mask"], spatial_size=(256, 256, 256)),
            ToTensord(keys=["source", "target", "mask"]),
            SqueezeDimd(keys=["source", "target", "mask"], dim=0),  # Squeeze the channel dimension
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


        # Step 3: create dataset (CacheDataset is fast for small/medium datasets)
        self.train_ds = CacheDataset(data=train_subset, transform=self.transforms, num_workers=4)
        self.val_ds = CacheDataset(data=val_subset, transform=self.transforms, num_workers=4)
        self.test_ds = CacheDataset(data=test_ds, transform=self.transforms, num_workers=4)
        
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
                          batch_size=self.training_params['training']['batch_size'],
                          shuffle=False,
                          num_workers=self.training_params['training']['num_workers'])

# Debug/test hook
if __name__ == "__main__":
    with open('/storage/an_inam/MR2MR/patch_pipeline/data.json', 'r') as f:
        params_data = json.load(f)
    with open('/storage/an_inam/MR2MR/patch_pipeline/configs/ESAU.json', 'r') as f:
        params_training = json.load(f)

    dm = VolumeDataModuleMONAI(params_data, params_training)
    dm.setup()
    loader = dm.train_dataloader()
    for batch in loader:
        print(batch['source'].shape, batch['target'].shape)
        if 'mask' in batch:
            print(batch['mask'].shape)
        break
