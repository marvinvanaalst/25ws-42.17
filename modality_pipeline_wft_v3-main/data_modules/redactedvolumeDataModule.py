import torchio as tio
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import json
# from box import Box
import os

class volumeDataModule(pl.LightningDataModule):
    def __init__(self, params_data, params_training):
        super().__init__()
        self.params = params_data
        self.training_params = params_training
        
        self.subjects_list = self.params['data']['training']['subjects']
        print(f"Subjects list: {self.subjects_list}")
        
        print("Checking if all subjects exist in the specified paths...")
        for s in self.subjects_list:
            if not os.path.exists(f"{self.params['data']['training']['subjects_path']}/{self.params['direction']['source']}/{s}"):
                raise FileNotFoundError(f"Source image for subject {s} not found at {self.params['data']['training']['subjects_path']}/{self.params['direction']['source']}/{s}")
            else:
                print(f"Source image for subject {s} found at {self.params['data']['training']['subjects_path']}/{self.params['direction']['source']}/{s}")
            if not os.path.exists(f"{self.params['data']['training']['subjects_path']}/{self.params['direction']['target']}/{s}"):
                raise FileNotFoundError(f"Target image for subject {s} not found at {self.params['data']['training']['subjects_path']}/{self.params['direction']['target']}/{s}")
            else:
                print(f"Target image for subject {s} found at {self.params['data']['training']['subjects_path']}/{self.params['direction']['target']}/{s}")
            if not os.path.exists(f"{self.params['data']['training']['subjects_path']}/mask/{s}"):
                print(f"Mask for subject {s} not found at {self.params['data']['training']['subjects_path']}/mask/{s}, proceeding without mask.")
            else:
                print(f"Mask for subject {s} found at {self.params['data']['training']['subjects_path']}/mask/{s}")
                
        print("All subjects exist in the specified paths. Proceeding to create the dataset...")
        
        for s in self.subjects_list:
            source_path = f"{self.params['data']['training']['subjects_path']}/{self.params['direction']['source']}/{s}"
            print(f"source_path = {source_path}, type = {type(source_path)}")
        
        
        subjects = [

            tio.Subject(
                source=tio.ScalarImage(f"{self.params['data']['training']['subjects_path']}/{self.params['direction']['source']}/{s}"),
                target=tio.ScalarImage(f"{self.params['data']['training']['subjects_path']}/{self.params['direction']['target']}/{s}"),
                mask=tio.LabelMap(f"{self.params['data']['training']['subjects_path']}/mask/{s}") if os.path.exists(f"{self.params['data']['training']['subjects_path']}/mask/{s}") else None
                
            )
            for s in self.subjects_list
        ]
        for s in self.subjects_list:
            print(type(f"{self.params['data']['training']['subjects_path']}/mask/{s}"))
        input("Xcadd")
        for i in range(len(subjects)-1):
            if subjects[i].source.shape != subjects[i].target.shape:
                print(f"Shape mismatch for subject {i}: {subjects[i].source.shape} vs {subjects[i].target.shape}")
            if subjects[i].mask is not None and subjects[i].mask.shape != subjects[i].source.shape:
                print(f"Mask shape mismatch for subject {i}: {subjects[i].mask.shape} vs {subjects[i].source.shape}")
                
            if subjects[i].source.shape != subjects[i+1].source.shape:
                print(f"Shape mismatch for source {i} and {i+1}: {subjects[i].source.shape} vs {subjects[i+1].target.shape}")
            if subjects[i].target.shape != subjects[i+1].target.shape:
                print(f"Shape mismatch for target {i} and {i+1}: {subjects[i].source.shape} vs {subjects[i+1].target.shape}")
            if subjects[i].mask.shape != subjects[i+1].mask.shape:
                print(f"Shape mismatch for mask {i} and {i+1}: {subjects[i].mask.shape} vs {subjects[i+1].mask.shape}")
        
        print("All subjects checked. No shape mismatches found.")

        transform = tio.Compose([
            tio.RescaleIntensity((0, 1)),
            tio.Resize((256,256,256))
        ])
        self.dataset = tio.SubjectsDataset(
            subjects,
            transform=transform
        )

        
    def setup(self, stage=None):
        pass



        # Code to ensure the size of all the datasamples is the same
        
    def train_dataloader(self):
        return DataLoader(self.dataset, 
                          batch_size=self.training_params['training']['batch_size'], 
                          num_workers=self.training_params['training']['num_workers'],
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, 
                          batch_size=self.training_params['training']['batch_size'], 
                          num_workers=self.training_params['training']['num_workers'],
                          )
    
if __name__ == "__main__":
    with open('/storage/an_inam/MR2MR/patch_pipeline/data.json', 'r') as f:
        params_data = json.load(f)
    with open('/storage/an_inam/MR2MR/patch_pipeline/configs/3DSwin_perceptual.json', 'r') as f:
        params_training = json.load(f)
    datamodule = volumeDataModule(params_data, params_training)
    train_loader = datamodule.train_dataloader()
    for batch in train_loader:
        print(batch)
        print(batch.keys())
        
        print(batch['source'][tio.DATA].shape, batch['target'][tio.DATA].shape, batch['mask'][tio.DATA].shape if 'mask' in batch else 'No mask')
        
        break
    
    # en