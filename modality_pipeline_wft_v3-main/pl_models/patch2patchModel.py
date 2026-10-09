import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchio as tio

import sys
sys.path.append('/storage/an_inam/MR2MR/patch_pipeline')
print(sys.path)
import pl_models.models.ESAU_net
from torch.utils.data import DataLoader

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics import MeanAbsoluteError, MeanSquaredError
import torchio as tio
import nibabel as nib
import importlib.util
import os
import inspect
from tqdm import tqdm
def pruge_extra_channels(tensor1, tensor2):
    """
    Prune extra channels from the tensors if they have more than one channel.
    """
    if tensor1.shape[0] != 1:
        print(f"Expected tensor1 to have batch size 1, but got {tensor1.shape[0]} channels.")
    if tensor2.shape[0] != 1:
        print(f"Expected tensor2 to have batch size 1, but got {tensor2.shape[0]} channels.")
    while len(tensor1.shape) != len(tensor2.shape):
        if len(tensor1.shape) > len(tensor2.shape):
            if tensor1.shape[1] == 1:
                tensor1 = tensor1.squeeze(1)
            else:
                tensor1 = tensor1.squeeze(0)
        elif len(tensor2.shape) > len(tensor1.shape):
            if tensor2.shape[1] == 1:
                tensor2 = tensor2.squeeze(1)
            else:
                tensor2 = tensor2.squeeze(0)
    return tensor1, tensor2

def dyanamic_import_model(model_path, model_class):
    """
    Dynamically import a model from a given path.
    """
    module_name = os.path.splitext(os.path.basename(model_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_class = getattr(module, model_class)
    return model_class

def filter_kwargs(func, kwargs):
    sig = inspect.signature(func)
    valid_params = sig.parameters
    return {k: v for k, v in kwargs.items() if k in valid_params}

class patch2patchModel(pl.LightningModule):
    def __init__(self, params, sampler, aggregator):
        super().__init__()
        self.params = params
        self.sampler = sampler
        self.aggregator = aggregator
        print("Aggregator: ", self.aggregator)
        # Intializing model from model path given as argument in params
        model_path = params["model_info"]["path"] 
        model_class = params["model_info"]["model_class"]
        
        model_params = filter_kwargs( model_class.__init__, params['model'])
        
        model = dyanamic_import_model(model_path, model_class)
        self.model = model(**model_params)
        
        # if self.params['model_info']['name'] == 'ESAU':
        #     model_params = filter_kwargs( pl_models.models.ESAU_net.ESAU.__init__, params['model'])
        #     self.model = pl_models.models.ESAU_net.ESAU(**model_params)
        # else:
        #     ValueError(f"Unsupported model name: {self.params['model_info']['name']}. Expected 'ESAU'.")
        
        self.lr = params['training']['learning_rate']
        
        # Loss function elaboration pending
        self.loss_fn = nn.L1Loss()
        
        # inside __init__ of vol_2_vol class
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)  # assumes inputs are normalized
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.mae = MeanAbsoluteError()
        self.mse = MeanSquaredError()
        
        self.save_hyperparameters()
        
        if self.params["loss"]["name"] == "restormerCabonier":
            from pl_models.models.losses.restormer_loss import UnifiedAdaptiveLoss
            # self.loss_fn = UnifiedAdaptiveLoss(use_perceptual=self.params["loss"]["use_perceptual"], perceptual_layer=self.params["loss"]["perceptual_layer"])
            self.loss_fn = UnifiedAdaptiveLoss()
        
        
    def forward(self, x):
        x = self.model(x)
        return x
    # def transfer_batch_to_device(self, batch, device, dataloader_idx):
    #     return batch
    def training_step(self, batch, batch_idx):
            
        if not self.params['training']['mono_channel']:
            # Convert input and target tensors to mono-channel if specified
            input_tensor, target_tensor, mask_tensor = batch['source'].squeeze(1), batch['target'].squeeze(1), batch['mask'].squeeze(1)
        else:
            input_tensor, target_tensor, mask_tensor = batch['source'], batch['target'], batch['mask']
            
        input_tensor = input_tensor * mask_tensor
        target_tensor = target_tensor * mask_tensor
    
        output = self(input_tensor)
        loss = self.loss_fn(output, target_tensor)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if not self.params['training']['mono_channel']:
            # Convert input and target tensors to mono-channel if specified
            input_tensor, target_tensor, mask_tensor = batch['source'].squeeze(1), batch['target'].squeeze(1), batch['mask'].squeeze(1)
        else:
            input_tensor, target_tensor, mask_tensor = batch['source'], batch['target'], batch['mask']
        
        input_tensor = input_tensor * mask_tensor
        target_tensor = target_tensor * mask_tensor

        output = self(input_tensor)

        # Compute loss
        loss = self.loss_fn(output, target_tensor)

        # Compute metrics
        psnr_val = self.psnr(output, target_tensor)
        ssim_val = self.ssim(output, target_tensor)
        mae_val = self.mae(output, target_tensor)
        mse_val = self.mse(output, target_tensor)

        # Logging to all loggers (WandB, TensorBoard, CSV)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True , sync_dist=True)
        self.log('val_psnr', psnr_val, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_ssim', ssim_val, on_step=False, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        self.log('val_mae', mae_val, on_step=False, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        self.log('val_mse', mse_val, on_step=False, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)

        return {
            'val_loss': loss,
            'val_psnr': psnr_val,
            'val_ssim': ssim_val,
            'val_mae': mae_val,
            'val_mse': mse_val
        }
        
    def predict_step(self, batch, batch_idx):
        # Over engineered to only handle one sample at a time
        print("ASDASDASDASDASDA00000")
        
        for k, v in batch.items():
            print(f"Key: {k}, Value shape: {v.shape if isinstance(v, torch.Tensor) else v}")
        # input("Check 1 shape of subject being input to GridSSampler: )")
        # sampler = tio.data.GridSampler(
        #             subject=batch,
        #             patch_size=self.params['training']['patch_size'],
        #             patch_overlap=self.params['training']['patch_overlap']
        #             )
        num_patches = int(len(self.sampler))
        self.training_patches = tio.Queue(
            subjects_dataset=self.batch,
            max_length=300,                  # max # of patches in queue
            samples_per_volume=num_patches,          # how many patches to extract per volume
            sampler = self.sampler,
            # num_workers=self.training_params['training']['num_workers'],
            shuffle_subjects=False,
            shuffle_patches=False
        )
        temp_test_dataloader = DataLoader(self.test_ds,
                          batch_size=self.params['training']['batch_size'],
                          shuffle=False,
                          num_workers=self.params['training']['num_workers'])
        print("ASDASDASDASDASDA")
        for xxcs, batch in enumerate(tqdm(temp_test_dataloader, desc="Predicting on test set")):
            if not self.params['training']['mono_channel']:
            # Convert input and target tensors to mono-channel if specified
                input_tensor, target_tensor, mask_tensor = batch['source'][tio.DATA].squeeze(1), batch['target'][tio.DATA].squeeze(1), batch['mask'][tio.DATA].squeeze(1)
            else:
                input_tensor, target_tensor, mask_tensor = batch['source'][tio.DATA], batch['target'][tio.DATA], batch['mask'][tio.DATA]
        
            input_tensor = input_tensor * mask_tensor
            target_tensor = target_tensor * mask_tensor
            
            predicted_patch = self(input_tensor)   
            location = predicted_patch[tio.LOCATION]
            predicted_patch = predicted_patch * mask_tensor
            
            self.patch_aggregator.add_batch(predicted_patch, location)
        output = self.patch_aggregator.get_output()
        affine = batch['source']['affine'][0]
        image = tio.ScalarImage(tensor=output, affine=affine)
        temp_Subject = tio.Subject(
                                predicted=image,
                                target = batch['target'][tio.DATA],
                                source=batch['source'][tio.DATA],
                                mask=batch['mask'][tio.DATA],
                                target_meta_dict=batch['target_meta_dict'],
                                source_meta_dict=batch['source_meta_dict'],
                                location=batch[tio.LOCATION]
                                )
        
        psnr_val = self.psnr(output, target_tensor)
        ssim_val = self.ssim(output, target_tensor)
        mae_val = self.mae(output, target_tensor)
        mse_val = self.mse(output, target_tensor)
        
        self.log('test_psnr', psnr_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_ssim', ssim_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_mae', mae_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_mse', mse_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {
            'file_name': batch['source_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.')[0],  # Extract filename without extension
            'model_prediction':temp_Subject,
            'metrics':{
            'test_psnr': psnr_val,
            'test_ssim': ssim_val,
            'test_mae': mae_val,
            'test_mse': mse_val,
            }
        }
            
    # def predict_step(self, batch, batch_idx):
    #     prediction = self(batch['source'][tio.DATA].to(self.device))
    #     return {
    #         "pred": prediction.cpu(),
    #         "source": batch['source'][tio.DATA].cpu(),
    #         "target": batch['target'][tio.DATA].cpu(),
    #         "mask": batch['mask'][tio.DATA].cpu(),
    #         "source_meta_dict": batch['source_meta_dict'],
    #         "target_meta_dict": batch['target_meta_dict'],
    #         "location": batch[tio.LOCATION]
    #     }

    def configure_optimizers(self):
            optmimizer =  torch.optim.Adam(self.parameters(), lr=self.lr)
            if self.params['training']['scheduler'] == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optmimizer, T_max=self.params['training']['epochs']//self.params['training']['cosine_restart_after'], eta_min=1e-8)
                return {
                    'optimizer': optmimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'epoch',
                        'frequency': 1,
                    }
                }
            elif self.params['training']['scheduler'] == 'SingleCycleCosine':
                total_steps = self.trainer.estimated_stepping_batches
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                                optmimizer,
                                max_lr=float(self.params['training']['learning_rate']),               # 🔼 peak learning rate
                                total_steps=total_steps,          # total number of steps = epochs * steps_per_epoch
                                pct_start=0.3,        # 🔁 what % of steps for warmup (rest for cooldown)
                                anneal_strategy='cos',# ⌛ decay shape: 'linear' or 'cos'
                                div_factor=1000.0,      # 🔻 initial_lr = max_lr / div_factor
                                final_div_factor=1e4, # 🔻 final_lr = max_lr / final_div_factor
                            )
                return {
                    'optimizer': optmimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'epoch',
                        'frequency': 1,
                    }
                }
            elif self.params['training']['scheduler'] == 'None':
                return optmimizer
            
if __name__ == "__main__":
    import json
    with open('/storage/an_inam/MR2MR/patch_pipeline/configs/3DUnet.json', 'r') as f:
        params = json.load(f)
    print(params)
    input("Model class loaded")
    model = vol_2_vol(params)
    model.load_from_checkpoint("/storage/an_inam/MR2MR/patch_pipeline/logs/ESAU_MSE_b2_BatchSize_2_LR_0.001/checkpoints/epoch=458-val_mae=0.06886855-val_psnr=17.35323334-val_ssim=0.57729757-val_mse=0.01839402.ckpt")
    print(model)
