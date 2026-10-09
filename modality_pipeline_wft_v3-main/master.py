from utils_custom.utils_opus import model_fetcher

import torch
import json
# from box import Box
import sys
import csv
import torchio as tio
import warnings
warnings.simplefilter("default")  # or "always"
import shutil

import os
from tqdm import tqdm
wand_boolean = False
print("WAND DB BOOLEAN", wand_boolean)
from utils_custom.csv_logger import CSVLogger
from utils_custom.LightningStyleCheckpoint import LightningStyleCheckpoint
argv = sys.argv

import random
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)                      # Python random module
    np.random.seed(seed)                   # Numpy
    torch.manual_seed(seed)                # CPU tensors
    torch.cuda.manual_seed(seed)           # Current GPU
    torch.cuda.manual_seed_all(seed)       # All GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Ensures deterministic convs
    
if len(argv) < 2:
    raise ValueError("Expected argv config file. Using default configuration.")
    
else:
    experimnet_config = argv[1]

avg_weighting = True
if avg_weighting:
    from merge_model_weights import average_model_weights
    
with open(experimnet_config, 'r') as f:
    experiment = json.load(f)

# input(experiment)

from utils_custom.eval_metrics import VolumeIQAMetrics

# Dataloader 

##??
if experiment['model_info']['modelType'] == 'vol2vol':
    pass
elif experiment['model_info']['modelType'] == 'patch2patch' or experiment['model_info']['modelType'] == "pre_computed_patch2patch":
    from utils_custom.utils_opus import model_fetcher
    model_initializer, model_loader_params = model_fetcher(experiment).fetch_model()
    # model = patch2patch(experiment)
else:
    raise ValueError(f"Unsupported model type: {experiment.model.type}. Expected types are '3DUNET' or 'VNet'.")

exp_name=f'{experiment["experiment_name"]}'


final_results = []



if os.path.isdir(f'logs/{experiment["experiment_name"]}'):
    print(f"\n\n\033[1;94mDirectory .logs/{experiment['experiment_name']} already exists. Do you want to continue?\033[0m\n")
else:
    print("\n\nUnique experiment name. Proceeding with training.")
    
os.makedirs(f'logs/{exp_name}', exist_ok=True)
os.makedirs(f'logs/{exp_name}/generations', exist_ok=True)
os.makedirs(f'logs/{exp_name}/checkpoints', exist_ok=True)
os.makedirs(f'logs/{exp_name}/metrics', exist_ok=True)


if wand_boolean:
    import wandb
    wandb.init(
    project=experiment["Project_Name"],       # your wandb project
    name=experiment["experiment_name"], # optional run name
    config=experiment
)

from utils_custom.composite_loss import compile_loss_fn
DEVICE = experiment['device']['devices'][0]
print(f"Using device: {DEVICE}")
shutil.copy(experimnet_config, f"logs/{exp_name}/config.json")
iqa_metrics = VolumeIQAMetrics(data_range=1.0, device = DEVICE)

if experiment['resume_from_sample']['do']:
    csv_logger_average = CSVLogger(log_dir=f'logs/{exp_name}/metrics', filename=f'average_testing_.csv', resume=True)
else:
    csv_logger_average = CSVLogger(log_dir=f'logs/{exp_name}/metrics', filename=f'average_testing_.csv', resume=False)
    
for sdx in range(len(experiment['data']['training']['subjects'])):
    sample_index = sdx
    if experiment["resume_from_sample"]["do"] and sample_index < experiment["resume_from_sample"]["index"]:
        print(f"Skipping sample {sample_index} as per resume configuration.")
        continue
    csv_logger_train = CSVLogger(log_dir=f'logs/{exp_name}/metrics', filename=f'{sample_index}_metircs_train_.csv')
    csv_logger_testing = CSVLogger(log_dir=f'logs/{exp_name}/metrics', filename=f'{sample_index}_metircs_testing_.csv')
    
    
    if experiment['model_info']['modelType'] == 'patch2patch':
        from data_modules.opus_dataloader import all_data_handler
        train_daloader_fetcher, val_dataloader_fetcher, test_dataloader_fetcher = all_data_handler(experiment['not_corrected_for_compute'], experiment, idx=sample_index)
    elif experiment['model_info']['modelType'] == 'pre_computed_patch2patch':
        from data_modules.opus_CachingPatchDataset import cache_patch_dataloader
        from patchify.patchify_tio import create_patchify_dataset
        create_patchify_dataset(experiment)
        cache_dl = cache_patch_dataloader(experiment, DEVICE, idx=sample_index).fetch_allloaders()
        train_dataloader, val_dataloader, test_dataloader = cache_dl['train'], cache_dl['validation'], cache_dl['test']
    elif experiment['model_info']['modelType'] == 'vol2vol':
        pass
    else:
        raise ValueError(f"Unsupported model type: {experiment.model_info.model_type}. Expected types are {experiment.model_info.possibleTypes}.")

    checkpoint_callback = LightningStyleCheckpoint(
        save_dir=f'logs/{exp_name}/checkpoints/{sample_index}',
        monitor='val_ssim',
        mode='max',
        top_k=experiment['checkpoint']['top_k'],
        save_every_n_epochs=experiment['checkpoint']['save_every_n_epochs'],
        save_weights_only=True,
        filename_template='{epoch:02d}-{monitor}={score:.4f}.pt'
    )
    checkpoint_callback_2 = LightningStyleCheckpoint(
        save_dir=f'logs/{exp_name}/checkpoints/{sample_index}',
        monitor='val_loss',
        mode='min',
        top_k=experiment['checkpoint']['top_k'],
        save_every_n_epochs=10000,
        # save_every_n_epochs=experiment['checkpoint']['save_every_n_epochs'],
        
        save_weights_only=True,
        filename_template='{epoch:02d}-{monitor}={score:.4f}.pt'
    )
    if avg_weighting:
        avg_annealing_checkpoint_callback = LightningStyleCheckpoint(
            save_dir=f'logs/{exp_name}/checkpoints_averaging/{sample_index}',
            monitor='val_loss',
            mode='min',
            # top_k=experiment['checkpoint']['top_k'],
            save_every_n_epochs=experiment['checkpoint']['save_every_n_epochs'],
            save_weights_only=True,
            filename_template='{epoch:02d}-{monitor}={score:.4f}.pt'
        )
    
    loss_fn = compile_loss_fn(experiment)
    

    # print(f"Evaluating sample {sample_index} {orig[0]['name']}")
    
    model = model_initializer(model_loader_params).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=experiment['training']['learning_rate'], weight_decay=experiment['training']['weight_decay'])
    
    
    epoch = 0
    for epoch_idx, _ in enumerate(tqdm(range(experiment['training']['epochs']) , desc=f"Epoch {epoch}")):
        loss_value = 0.0
        num_samples = 0
        model.train()
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Training", leave=False)):
            output = model(batch['source'])
            loss = loss_fn(output, batch['target'])
            loss.backward()
            loss_value += loss.item() * batch['source'].shape[0]
            num_samples += batch['source'].shape[0]
            optimizer.step()
            optimizer.zero_grad()
        loss_value /= num_samples

        # print("validation")
        model.eval()
        val_error, num_samples = 0.0, 0.0
        val_tqdm_loop = tqdm(val_dataloader, leave=False)
        # if epoch_idx > experiment['training']['no_val_till']:
        with torch.no_grad():
            for val_batch_idx, val_batch in enumerate(val_tqdm_loop):
                val_output = model(val_batch['source'])
                val_loss = loss_fn(val_output, val_batch['target'])
                real_target = val_batch['target']
                # psnr, ssim = iqa_metrics.compute(val_output, real_target, val_batch['mask'])
                val_error += val_loss.item() * batch['source'].shape[0]
                num_samples += val_batch['source'].shape[0]
            val_loss = val_error / num_samples
            # metrics = iqa_metrics.value()
            # iqa_metrics.reset()
            val_tqdm_loop.set_description(f"Val Loss={val_loss:.4f}")
        epoch += 1

        metrics = {
            'epoch': epoch_idx + 1,
            'train_loss': loss_value,
            'val_loss': val_loss,
            # 'val_psnr': metrics['PSNR'],
            # 'val_ssim': metrics['SSIM'],
            # 'val_mae': metrics['MAE'],
            # 'val_mse': metrics['MSE'],
            # 'val_mmsim': metrics['MaskedSSIM']
        }
            
        # if wand_boolean:
        #     wandb.log({
        #         "epoch": epoch_idx + 1,
        #         f"train/loss_{sample_index}": loss_value,
        #         f"val/loss_{sample_index}": val_loss,
        #         f"val/psnr_{sample_index}":  metrics['PSNR'],
        #         f"val/ssim_{sample_index}":  metrics['SSIM'],
        #         f"val/mae_{sample_index}":  metrics['MAE'],
        #         f"val/mse_{sample_index}":  metrics['MSE']
        #     })
        csv_logger_train.log(metrics)
        # checkpoint_callback.save(model, epoch, metrics)
        checkpoint_callback_2.save(model, epoch, metrics)
        if avg_weighting:
            avg_annealing_checkpoint_callback.save(model, epoch, metrics)
    
    if avg_weighting:
        average_model_weights(f'logs/{exp_name}/checkpoints_averaging/{sample_index}', f'logs/{exp_name}/checkpoints/{sample_index}', model_loader_params, model_initializer, f"{experiment['training']['epochs']}_{experiment['checkpoint']['save_every_n_epochs']}")
    all_weights = os.listdir(f"logs/{exp_name}/checkpoints/{sample_index}")
    all_weights = [os.path.join(f"logs/{exp_name}/checkpoints/{sample_index}", path) for path in all_weights]

    weight_tqdm_loop = tqdm(all_weights, desc="Evaluating Weights", leave=False)
    
    
    for wdx, weight_path in enumerate(weight_tqdm_loop):
        # print(weight_path)
        # input()
        
        for ele in weight_path.split('/'):
            if 'epoch' in ele:
                epoch = int(ele.split('=')[1].split('.')[0])
        weight_prefix = f"{weight_path.split('/')[-1].split('-')[0]}"
        weight_tqdm_loop.set_description(f"Evaluating Weight: {weight_prefix} Epoch: {epoch}")
        
        # print("weight_prefix", weight_prefix)
        ckpt = torch.load(weight_path, weights_only=False)
        if "state_dict" not in ckpt:
            model.load_state_dict(ckpt)
        else:
            model.load_state_dict(ckpt["state_dict"])
        # print(f"Loaded model from {weight_path}")
        
        
        # Evaluate the model on the test set
        model.eval()
        with torch.no_grad():
            for train_batch_idx, batch in enumerate(test_dataloader['queue']):
                fake_output = model(batch['source']['data'].to(DEVICE))
                # fake_output = fake_output*(batch['mask']['data'].to(DEVICE))
                test_dataloader['aggregator'].add_batch(fake_output, batch['location'])
            output = (test_dataloader['aggregator'].get_output_tensor().to(DEVICE))*(test_dataloader['original'][0]['mask']['data'].to(DEVICE))
            output = torch.clip(output, min=0.0, max=1.0)
            
            # print(f"Output shape: {output.shape}")
            # print(f"Original shape: {orig[0]['target']['data'].shape}")
            # input()
            test_metrics = iqa_metrics.compute(output.unsqueeze(1), test_dataloader['original'][0]['target']['data'].to(DEVICE).unsqueeze(1)*test_dataloader['original'][0]['mask']['data'].to(DEVICE),test_dataloader['original'][0]['mask']['data'].to(DEVICE))
            metrics = iqa_metrics.value()
            iqa_metrics.reset()
            
            print(f"Test Metrics for sample {sample_index}: PSNR: {metrics['PSNR']}, SSIM: {metrics['SSIM']}, MAE: {metrics['MAE']}, MSE: {metrics['MSE']}")
            sample_test_metics = {
                'sample_index': sample_index,
                'weight_prefix': weight_prefix,
                'test_psnr': metrics['PSNR'],
                'test_ssim': metrics['SSIM'],
                'test_mae': metrics['MAE'],
                'test_mse': metrics['MSE']
            }
            
            final_results.append(sample_test_metics)
            csv_logger_testing.log(sample_test_metics)
            affine = batch['source']['affine'][0]
            image = tio.ScalarImage(tensor=output.cpu(), affine=affine)
            image.save(f"logs/{exp_name}/generations/{sample_index}_{test_dataloader['original'][0]['name']}_{weight_prefix}_fake.nii.gz")
            test_dataloader['original'][0]['target'].save(f"logs/{exp_name}/generations/{sample_index}_{test_dataloader['original'][0]['name']}_orig.nii.gz")
            
        
    del model