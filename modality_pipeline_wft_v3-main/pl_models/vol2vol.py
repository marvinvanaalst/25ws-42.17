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

from monai.inferers import sliding_window_inference
import nibabel as nib
import importlib.util
import os
import inspect

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
        
        if self.params["loss"]["name"] == "restormerCabonier":
            from pl_models.models.losses.restormer_loss import UnifiedAdaptiveLoss
            # self.loss_fn = UnifiedAdaptiveLoss(use_perceptual=self.params["loss"]["use_perceptual"], perceptual_layer=self.params["loss"]["perceptual_layer"])
            self.loss_fn = UnifiedAdaptiveLoss()
        
        
    def forward(self, x):
        x = self.model(x)
        return x

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
        
    def test_step(self, batch, batch_idx):
        # Over engineered to only handle one sample at a time
        if not self.params['training']['mono_channel']:
            # Convert input and target tensors to mono-channel if specified
            input_tensor, target_tensor, mask_tensor = batch['source'].squeeze(1), batch['target'].squeeze(1), batch['mask'].squeeze(1)
        else:
            input_tensor, target_tensor, mask_tensor = batch['source'], batch['target'], batch['mask']
        input_tensor = input_tensor * mask_tensor
        target_tensor = target_tensor * mask_tensor
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        if self.params['model_info']['modelType'] == "vol2vol":
            output = self(input_tensor)
        elif self.params['model_info']['modelType'] == "patch2patch":
            with torch.no_grad():
                # from monai.inferers.utils import compute_importance_map
                if not self.params['training']['mono_channel']:
                    class ChannelsLastWrapper(torch.nn.Module):
                        def __init__(self, model):
                            super().__init__()
                            self.model = model

                        def forward(self, x):
                            # x: (B, C, D, H, W) from MONAI
                            # we assume C=1, so remove channel and permute
                            # to (B, D, H, W) → (B, H, W, D)
                            x = x.squeeze(1).permute(0, 2, 3, 1)  # D, H, W → H, W, D
                            out = self.model(x)
                            # if needed, restore to (B, C, D, H, W)
                            out = out.unsqueeze(1).permute(0, 1, 4, 2, 3)
                            return out
                    self.model = ChannelsLastWrapper(self.model)
# Patch compute_importance_map to store map on CPU
                # def compute_importance_map_cpu(patch_size, mode, device=None):
                #     return compute_importance_map(patch_size, mode, device=torch.device("cpu"))

                # # Replace MONAI's function (temporary override)
                # import monai.inferers.utils
                # monai.inferers.utils.compute_importance_map = compute_importance_map_cpu
                
                # # model_cpu = self.model.cpu()
                # input_tensor_cpu = input_tensor.cpu()
                output = sliding_window_inference(
                    inputs=input_tensor,
                    roi_size=self.params['training']['patch_size'],
                    sw_batch_size=self.params['patch_inference']['inference_batch_size'],
                    predictor=self.model,
                    overlap=self.params['patch_inference']['sampler_overlap_percentage'],
                    mode=self.params['patch_inference']['smoothening_method']  # smoother blending at edges
                )  # outpu
        
        # print("Output shape: ", output.shape)
        if  self.params['training']['mono_channel']:
            
            output= output.squeeze(0)  # remove singleton dimensions if any
            target_tensor = target_tensor.squeeze(0)  # remove singleton dimensions if any
            input_tensor = input_tensor.squeeze(0)  # remove singleton dimensions if any
            mask_tensor = mask_tensor.squeeze(0)  # remove singleton dimensions if any
        else:
            output = output.squeeze(0)
        # Compute metrics
        psnr_val = self.psnr(output, target_tensor)
        ssim_val = self.ssim(output, target_tensor)
        mae_val = self.mae(output, target_tensor)
        mse_val = self.mse(output, target_tensor)

        # Logging to all loggers (WandB, TensorBoard, CSV)
        self.log('test_psnr', psnr_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_ssim', ssim_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_mae', mae_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_mse', mse_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # print("IP ", input_tensor.shape, "Synth ", target_tensor.shape, "Tar ", output.shape)
        # Savving Generation to disk
        if self.params['training']['save_generation']:
            
            filenames = batch["source_meta_dict"]["filename_or_obj"][0].split('/')[-1].split('.')[0]  # Extract filename without extension
            
            source_nib = nib.Nifti1Image(input_tensor.cpu().numpy()[0], affine = batch["source_meta_dict"]["original_affine"].cpu().numpy()[0])
            pred_nib = nib.Nifti1Image(output.cpu().numpy()[0], affine =batch["target_meta_dict"]["original_affine"].cpu().numpy()[0])
            target_nib = nib.Nifti1Image(target_tensor.cpu().numpy()[0],affine = batch["target_meta_dict"]["original_affine"].cpu().numpy()[0])

            mask_nib = nib.Nifti1Image(mask_tensor.cpu().numpy()[0],affine = batch["target_meta_dict"]["original_affine"].cpu().numpy()[0])
            
            
            os.makedirs(f"logs/{self.params['experiment_name']}/test_generation/", exist_ok=True)
            nib.save(source_nib, f"logs/{self.params['experiment_name']}/test_generation/{filenames}_Real3T.nii.gz")
            nib.save(pred_nib, f"logs/{self.params['experiment_name']}/test_generation/{filenames}_Synth7T.nii.gz")
            nib.save(target_nib, f"logs/{self.params['experiment_name']}/test_generation/{filenames}_Real7T.nii.gz")
            nib.save(mask_nib, f"logs/{self.params['experiment_name']}/test_generation/{filenames}_mask.nii.gz")
            
        return {
            'test_psnr': psnr_val,
            'test_ssim': ssim_val,
            'test_mae': mae_val,
            'test_mse': mse_val,
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
