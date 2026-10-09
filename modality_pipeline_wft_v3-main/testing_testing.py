from pytorch_lightning import Trainer

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

import torch
import json
from box import Box
import sys

import warnings
warnings.simplefilter("default")  # or "always"
import os



argv = sys.argv
if len(argv) < 2:
    raise ValueError("Expected argv config file. Using default configuration.")
    
else:
    experimnet_config = argv[1]
    
    # Default configuration file if not provided as an argument


# experimnet_config = 'configs/3DUnet.json'
    
with open('/storage/an_inam/MR2MR/patch_pipeline/data.json', 'r') as f:
    params_data = json.load(f)    
    
with open(experimnet_config, 'r') as f:
    experiment = json.load(f)
    
# input(experiment)

# Dataloader 
if experiment["model_info"]["modelType"] == '3DPatch':
    from data_modules.patchDataModule import patchDataModule
    dataModule = patchDataModule(params_data, experiment)
elif experiment['model_info']['modelType'] == 'tio3D':
    from data_modules.volumeDataModule import volumeDataModule
    dataModule = volumeDataModule(params_data, experiment)
elif experiment['model_info']['modelType'] == 'vol2vol':
    from data_modules.MONAIvolumeDataModule import VolumeDataModuleMONAI
    dataModule = VolumeDataModuleMONAI(params_data, experiment)
else:
    raise ValueError(f"Unsupported model type: {experiment.model_info.model_type}. Expected types are {experiment.model_info.possibleTypes}.")


if experiment['model_info']['modelType'] == 'vol2vol':
    from pl_models.vol2vol import vol_2_vol
    model = vol_2_vol(experiment)
    # input("cc")
else:
    raise ValueError(f"Unsupported model type: {experiment.model.type}. Expected types are '3DUNET' or 'VNet'.")


exp_name=f'{experiment["experiment_name"]}_BatchSize_{experiment["training"]["batch_size"]}_LR_{experiment["training"]["learning_rate"]}'
    
# Init DataModule and Model
# Callbacks
save_checkpoint = ModelCheckpoint(
    dirpath=f"logs/{exp_name}/checkpoints",
    save_last=True,
    # every_n_epochs=experiment['checkpoint']['save_frequency_epoch'], 
    save_top_k=experiment['checkpoint']['save_topk'],
    monitor=experiment['checkpoint']['monitor'],
    filename='{epoch}-{val_mae:.8f}-{val_psnr:.8f}-{val_ssim:.8f}-{val_mse:.8f}',
)

trainer = Trainer(max_epochs=experiment['training']['epochs'], 
                  callbacks=[save_checkpoint], 
                  accelerator="auto", 
                  devices=1)


# weights_path = f"logs/{exp_name}/checkpoints"
# weights = os.listdir(weights_path)
# max_stratergy = 'ssim'

# max_contender = weights[0]
# for element in weights:
#     temp = element.split('-')
#     for edx, temp_element in enumerate(temp):
#         if max_stratergy in temp_element:
#             if float(temp_element.split('=')[1]) > float(max_contender.split('-')[edx].split('=')[1]):
#                 max_contender = element

# print(f"Best model based on {max_stratergy} is: {max_contender}")
# best_model_path = os.path.join(weights_path, max_contender)

# model.load_from_checkpoint(best_model_path,strict=False)



# Load safely with weights_only to avoid unpickling errors
ckpt = torch.load("/storage/an_inam/MR2MR/patch_pipeline/logs/ESAU128_MSE_b2^3_BatchSize_1_LR_0.001/checkpoints/last.ckpt", weights_only=False)
model.load_state_dict(ckpt['state_dict'], strict=True)

# Print top-level keys
print("Top-level keys in checkpoint:", ckpt.keys())

# trainer.test(model, datamodule=dataModule)