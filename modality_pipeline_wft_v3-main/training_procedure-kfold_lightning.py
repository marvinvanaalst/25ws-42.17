from pytorch_lightning import Trainer

# from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

import torch
import json
# from box import Box
import sys
import csv
import torchio as tio
import warnings
warnings.simplefilter("default")  # or "always"
import os

wand_boolean = False
print("WAND DB BOOLEAN", wand_boolean)

# input("Check if EXP NAME is correct in the config file")
# input("Check if GPU is correct in the config file")
# from torch._utils import _rebuild_tensor_v2
# from torch.serialization import add_safe_globals
# # torch.serialization.add_safe_globals([_reconstruct])
# add_safe_globals({'_rebuild_tensor_v2': _rebuild_tensor_v2})

argv = sys.argv
if len(argv) < 2:
    raise ValueError("Expected argv config file. Using default configuration.")
    
else:
    experimnet_config = argv[1]
    
    # Default configuration file if not provided as an argument


# experimnet_config = 'configs/3DUnet.json'
    
with open('/Drive4T/inam/3T_7T_Synthesis/data.json', 'r') as f:
    params_data = json.load(f)    
    
with open(experimnet_config, 'r') as f:
    experiment = json.load(f)
    
# input(experiment)

# Dataloader 


if experiment['model_info']['modelType'] == 'vol2vol':
    from pl_models.vol2vol import vol_2_vol
    model_fn = vol_2_vol
    # model = vol_2_vol(experiment)
    # input("cc")
elif experiment['model_info']['modelType'] == 'patch2patch':
    from pl_models.patch2patchModel import patch2patchModel
    model_fn = patch2patchModel
    # model = patch2patch(experiment)
else:
    raise ValueError(f"Unsupported model type: {experiment.model.type}. Expected types are '3DUNET' or 'VNet'.")


# from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from pytorch_lightning import Callback

# Init DataModule and Model
# Callbacks
exp_name=f'{experiment["experiment_name"]}'
csv_logger = CSVLogger(save_dir='logs',  name=f'{exp_name}/csv_logs')

final_results = []


dir_path = "/path/to/directory"

if os.path.isdir(f'logs/{experiment["experiment_name"]}'):
    input(f"\n\n\033[1;94mDirectory .logs/{experiment['experiment_name']} already exists. Do you want to continue?\033[0m\n")
else:
    print("\n\nUnique experiment name. Proceeding with training.")
    


for sample_index in range(len(params_data['data']['training']['subjects'])):
    if experiment["model_info"]["modelType"] == 'bmv':
        raise ValueError("BMV model type is not supported in this script. Please use a different model type.")
        from data_modules.patchDataModule import patchDataModule
        dataModule = patchDataModule(params_data, experiment, idx=sample_index)
    elif experiment['model_info']['modelType'] == 'patch2patch':
        # input("Patch2Patch model type selected")
        from data_modules.tioPatchDataModule import tioPatchLoader
        dataModule = tioPatchLoader(params_data, experiment, idx=sample_index)
    elif experiment['model_info']['modelType'] == 'vol2vol':
        from data_modules.tioPatchDataModule import patchDataModule
        dataModule = patchDataModule(params_data, experiment, idx=sample_index)
    else:
        raise ValueError(f"Unsupported model type: {experiment.model_info.model_type}. Expected types are {experiment.model_info.possibleTypes}.")


    # exp_name=f'{experiment["experiment_name"]}_BatchSize_{experiment["training"]["batch_size"]}_LR_{experiment["training"]["learning_rate"]}_scheduler_{experiment["training"]["scheduler"]}'
    exp_name=f'{experiment["experiment_name"]}'
    
    if wand_boolean:
        wandb_logger = WandbLogger(project="MR2MR_WACV", name=f'{exp_name}_{sample_index}')

    print("No tensor board logger for now")
    # tb_logger = TensorBoardLogger(save_dir='logs', name=f'{exp_name}/tb_logs/{sample_index}/')
    csv_logger = CSVLogger(save_dir='logs',  name=f'{exp_name}/csv_logs/{sample_index}')

    if wand_boolean:
        # loggers = [wandb_logger, tb_logger, csv_logger]
        loggers = [wandb_logger, csv_logger]
        
    else:
        loggers = [csv_logger]
        
        # loggers = [tb_logger, csv_logger]
    
    # class CustomMetricLogger(Callback):
    #     def on_test_epoch_end(self, trainer, pl_module):
    #         # Manually log train metrics to train_logger
    #         metrics = {
    #             "epoch": trainer.current_epoch,
    #             "train_loss": trainer.callback_metrics.get("train_loss"),
    #             "train_acc": trainer.callback_metrics.get("train_acc"),
    #         }
    #         train_logger.log_metrics(metrics, step=trainer.global_step)

    
    save_checkpoint = ModelCheckpoint(
    dirpath=f"logs/{exp_name}/checkpoints/{sample_index}",
    # save_last=True,
    # every_n_epochs=experiment['checkpoint']['save_frequency_epoch'], 
    save_top_k=experiment['checkpoint']['save_topk'],
    monitor=experiment['checkpoint']['monitor'],
    filename='{epoch}-{val_mae:.8f}-{val_psnr:.8f}-{val_ssim:.8f}-{val_mse:.8f}',
    )
    if experiment['model_info']['modelType'] == 'patch2patch':
        sampler, aggregator = dataModule.fetch_sampler_aggregator_sampler()
        model = model_fn(experiment, sampler=sampler, aggregator= aggregator)
    else:
        model = model_fn(experiment)

    # Trainer
    trainer = Trainer(max_epochs=experiment['training']['epochs'], 
                    callbacks=[save_checkpoint], 
                    logger=loggers,
                    accelerator="auto", 
                    devices=experiment['device']['devices'],
                    benchmark=experiment['device']['benchmark'],
                    strategy=experiment['device']['strategy'],
                    # num_sanity_val_steps=0
                    #   profiler="simple",)
            )

    print("Trainer not runnung yet")
    trainer.fit(model, datamodule=dataModule)
    # print("Asynchronous apparently")


    weights_path = f"logs/{exp_name}/checkpoints/{sample_index}"
    weights = os.listdir(weights_path)
    max_stratergy = 'ssim'

    max_contender = weights[0]
    print("Using arbirary strategy to select best model")
    # for element in weights:
    #     temp = element.split('-')
    #     for edx, temp_element in enumerate(temp):
    #         if max_stratergy in temp_element:
    #             print('akr', float(temp_element.split('=')[1]))
    #             print('akr',float(max_contender.split('-')[edx].split('=')[1]))
    #             input()
    #             if float(temp_element.split('=')[1]) > float(max_contender.split('-')[edx].split('=')[1]):
    #                 max_contender = element

    # print(f"Best model based on {max_stratergy} is: {max_contender}")

    ckpt_path = os.path.join(weights_path, max_contender)
    # model.load_from_checkpoint(ckpt_path)
    ckpt = torch.load(ckpt_path, weights_only=False)
    print("Loading model from checkpoint:", ckpt_path)
    model.load_state_dict(ckpt["state_dict"])
    print("Model loaded successfully from checkpoint.")
    predictions = trainer.predict(model, dataloaders=dataModule)
    print("Predictions made successfully.")
    final_results.append(*predictions['metrics'])
    output = predictions['model_prediction']
    file_name = output["file_name"]
    output['predicted'].save(f"logs/{exp_name}/test_generation/{sample_index}_{file_name}_Synthetic_{params_data['direction']['target']}.nii.gz")
    output['target'].save(f"logs/{exp_name}/test_generation/{sample_index}_{file_name}_Real_{params_data['direction']['target']}.nii.gz")
    output['source'].save(f"logs/{exp_name}/test_generation/{sample_index}_{file_name}_Real_{params_data['direction']['source']}.nii.gz")
    
    
    
    
    if wand_boolean:
        wandb_logger.experiment.finish()
    
    del trainer
    
    
with open(f"logs/{exp_name}/final_metrics.csv", mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=final_results[0].keys())  # keys become columns
    writer.writeheader()  # writes: epoch,train_loss,val_loss
    writer.writerows(final_results)  # writes each dictionary as a row

import wandb
run = wandb.init(project="MR2MR", name=f'{exp_name}_results')
for element in final_results:
    wandb.log({
        "test_psnr": element['test_psnr'],
        "test_ssim": element['test_ssim'],
        "test_mae": element['test_mae'],
        "test_mse": element['test_mse']
    })
num_samples = len(final_results)
psnr, ssim, mae, mse = 0, 0, 0, 0
for result in final_results:
    psnr += result['test_psnr']
    ssim += result['test_ssim']
    mae += result['test_mae']
    mse += result['test_mse']
psnr /= num_samples
ssim /= num_samples
mae /= num_samples
mse /= num_samples


final_data = {
    "average_test_psnr": psnr,
    "average_test_ssim": ssim,
    "average_test_mae": mae,
    "average_test_mse": mse,
    "epoch": experiment['training']['epochs'],
}
if wand_boolean:
    wandb.log(final_data)
print(f"Average PSNR: {psnr}, SSIM: {ssim}, MAE: {mae}, MSE: {mse}")
# Save the final results
with open(f"logs/{exp_name}/final_metrics_avg.csv", mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=final_data.keys())
    writer.writeheader()
    writer.writerow(final_data)