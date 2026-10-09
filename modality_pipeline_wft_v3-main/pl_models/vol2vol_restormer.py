import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchio as tio

import sys
sys.path.append('/storage/an_inam/MR2MR/patch_pipeline')
print(sys.path)
import pl_models.models.ESAU_net


from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics import MeanAbsoluteError, MeanSquaredError


import importlib.util
import os
import inspect

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

class vol_2_vol(pl.LightningModule):
    def __init__(self, params, lr=1e-3):
        super().__init__()
        self.params = params

        # Intializing model from model path given as argument in params
        model_path = params["model_info"]["path"] 
        model_class = params["model_info"]["model_class"]
        
        model_params = filter_kwargs( pl_models.models.ESAU_net.ESAU.__init__, params['model'])
        
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
        
        
    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):

        input_tensor = batch['source'].squeeze(1)  # Assuming input is of shape [B, C, D, H, W]
        # target_tensor = batch['target'][tio.DATA].float()
        target_tensor = batch['target'].squeeze(1)  # Assuming target is of shape [B, C, D, H, W]
        if batch['mask'] is not None:
            mask_tensor = batch['mask'].squeeze(1)  # Assuming mask is of shape [B, C, D, H, W]
            input_tensor = input_tensor * mask_tensor
            target_tensor = target_tensor * mask_tensor
        
        output = self(input_tensor)
        loss = self.loss_fn(output, target_tensor)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_tensor = batch['source']
        target_tensor = batch['target']
        mask_tensor = batch.get('mask', None)  # safer way to handle optional mask

        if mask_tensor is not None:
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
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_psnr', psnr_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_ssim', ssim_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mae', mae_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mse', mse_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {
            'val_loss': loss,
            'val_psnr': psnr_val,
            'val_ssim': ssim_val,
            'val_mae': mae_val,
            'val_mse': mse_val
        }
        
    def test_step(self, batch, batch_idx):
        input_tensor = batch['source']
        target_tensor = batch['target']
        mask_tensor = batch.get('mask', None)  # safer way to handle optional mask

        if mask_tensor is not None:
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
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_psnr', psnr_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_ssim', ssim_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_mae', mae_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_mse', mse_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {
            'test_loss': loss,
            'test_psnr': psnr_val,
            'test_ssim': ssim_val,
            'test_mae': mae_val,
            'test_mse': mse_val
        }

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
