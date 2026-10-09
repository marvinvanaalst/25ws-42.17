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

wand_boolean = True

# from torch._utils import _rebuild_tensor_v2
# from torch.serialization import add_safe_globals
# # torch.serialization.add_safe_globals([_reconstruct])
# add_safe_globals({'_rebuild_tensor_v2': _rebuild_tensor_v2})

argv = sys.argv
if len(argv) < 2:
    raise ValueError("Expected argv config file. Using default configuration.")
    
else:
    experimnet_config = argv[1]

from pytorch_lightning import Trainer, Callback
    
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
    from data_modules.redactedvolumeDataModule import volumeDataModule
    dataModule = volumeDataModule(params_data, experiment)
elif experiment['model_info']['modelType'] == 'vol2vol':
    from data_modules.volumeDataModule import VolumeDataModuleMONAI
    dataModule = VolumeDataModuleMONAI(params_data, experiment)
else:
    raise ValueError(f"Unsupported model type: {experiment.model_info.model_type}. Expected types are {experiment.model_info.possibleTypes}.")


if experiment['model_info']['modelType'] == 'vol2vol':
    from pl_models.vol2vol import vol_2_vol
    model = vol_2_vol(experiment)
    # input("cc")
else:
    raise ValueError(f"Unsupported model type: {experiment.model.type}. Expected types are '3DUNET' or 'VNet'.")


from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger

exp_name=f'{experiment["experiment_name"]}_BatchSize_{experiment["training"]["batch_size"]}_LR_{experiment["training"]["learning_rate"]}_scheduler_{experiment["training"]["scheduler"]}'
if wand_boolean:
    wandb_logger = WandbLogger(project="MR2MR", name=exp_name)

tb_logger = TensorBoardLogger(save_dir='logs', name=f'{exp_name}/tb_logs')
csv_logger = CSVLogger(save_dir='logs',  name=f'{exp_name}/csv_logs')

if wand_boolean:
    loggers = [wandb_logger, tb_logger, csv_logger]
else:
    loggers = [tb_logger, csv_logger]
    
# Init DataModule and Model
# Callbacks
save_checkpoint = ModelCheckpoint(
    dirpath=f"logs/{exp_name}/checkpoints",
    # save_last=True,
    # every_n_epochs=experiment['checkpoint']['save_frequency_epoch'], 
    save_top_k=experiment['checkpoint']['save_topk'],
    monitor=experiment['checkpoint']['monitor'],
    filename='{epoch}-{val_mae:.8f}-{val_psnr:.8f}-{val_ssim:.8f}-{val_mse:.8f}',
)

# Trainer
trainer = Trainer(max_epochs=experiment['training']['epochs'], 
                  callbacks=[save_checkpoint], 
                  logger=loggers,
                  accelerator="auto", 
                  devices=1,
                  benchmark=True,
                #   profiler="simple",)
        )

trainer.fit(model, datamodule=dataModule)
# print("Asynchronous apparently")


weights_path = f"logs/{exp_name}/checkpoints"
weights = os.listdir(weights_path)
max_stratergy = 'ssim'

max_contender = weights[0]
for element in weights:
    temp = element.split('-')
    for edx, temp_element in enumerate(temp):
        if max_stratergy in temp_element:
            if float(temp_element.split('=')[1]) > float(max_contender.split('-')[edx].split('=')[1]):
                max_contender = element

print(f"Best model based on {max_stratergy} is: {max_contender}")

ckpt_path = os.path.join(weights_path, max_contender)
# model.load_from_checkpoint(ckpt_path)
ckpt = torch.load(ckpt_path, weights_only=False)
model.load_state_dict(ckpt["state_dict"])
spsp = trainer.test(model, datamodule=dataModule)

print("Test results:")
print(spsp)
# ###### ###### ###### ###### ###### ##### ###### ###### ###### ###### ###### ####
# best_model_path = os.path.join(weights_path, max_contender)
# input(f"Best model path: {best_model_path}")

# model.load_state_dict(torch.load(best_model_path)['state_dict'], weights_only=False)
# trainer.test(model, datamodule=dataModule)