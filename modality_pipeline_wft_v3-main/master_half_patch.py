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
from torch.cuda.amp import autocast, GradScaler
import os
from tqdm import tqdm
wand_boolean = False
print("WAND DB BOOLEAN", wand_boolean)
from utils_custom.csv_logger import CSVLogger
from utils_custom.LightningStyleCheckpoint import LightningStyleCheckpoint
from utils_custom.load_weights import load_model_state, load_training_state
from torch.optim.lr_scheduler import LambdaLR
import math
argv = sys.argv

from torch.utils.tensorboard import SummaryWriter

import random
import numpy as np

from augmentations.tta import *

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

avg_weighting = False
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

writer = SummaryWriter(log_dir=f"runs/{exp_name}")

if os.path.isdir(f'logs/{experiment["experiment_name"]}'):
    print(f"\n\n\033[1;94mDirectory .logs/{experiment['experiment_name']} already exists. Do you want to continue?\033[0m\n")
else:
    print("\n\nUnique experiment name. Proceeding with training.")
   
if os.path.exists(f"logs/{exp_name}"):
    input(f"logs/{exp_name} Already exists. Do you want to continue?")
    
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
cache_mode = experiment["device"].get("cache_stratergy", "same_gpu")

# gin_augments = experiment.get("pretrain_augments", {}).get("gin", {}).get("apply", False)
# gin_mask = experiment.get("pretrain_augments", {}).get("gin", {}).get("mask", False)
# gin_arguments = experiment.get("pretrain_augments", {}).get("gin", {}).get("gin_arguments", {})

block_augments = experiment.get("pretrain_augments",{}).get("block_augments",{}).get("apply", False)
block_config = experiment.get("pretrain_augments",{}).get("block_augments",{}).get("config", {})

pre_post_norm = experiment.get("pretrain_augments", {}).get("pre_post_norm", {}).get("apply", False)
pre_post_norm_type = experiment.get("pretrain_augments", {}).get("pre_post_norm", {}).get("type", False)

tta_flip_enabled = experiment.get("tta_flips", {}).get("enable", False)


# gin_mask = experiment["pretrain_augments"]["mask"].get("apply", False)
from augmentations.block_augments import augment, GIN3D
from components.testing_logic_patch import  test_on_sample
val_test_sample = test_on_sample(experiment)

if block_config.get('gin_apply', False):
    print("\n\n\n GIN+MASK Augments enabled.")    
    gin_net = GIN3D().to(DEVICE)
else:
    gin_net = None

if pre_post_norm:
    from augmentations.pre_post_norm import normalize_per_sample_masked, denormalize

if os.path.exists(f"logs/{exp_name}/config.json"):
    print("Skipping config file copying,  source and destination are the same file.")
else:
    shutil.copy(experimnet_config, f"logs/{exp_name}/config.json")


iqa_metrics = VolumeIQAMetrics(data_range=1.0, device = DEVICE)

if experiment['resume_from_sample']['do']:
    csv_logger_average = CSVLogger(log_dir=f'logs/{exp_name}/metrics', filename=f'average_testing_.csv', resume=True)
else:
    csv_logger_average = CSVLogger(log_dir=f'logs/{exp_name}/metrics', filename=f'average_testing_.csv', resume=False)



for sdx in range(len(experiment['data']['training']['subjects'])//experiment['training']['splits']['num_test_samples']):
    sample_index = sdx
    if experiment["resume_from_sample"]["do"] and sample_index < experiment["resume_from_sample"]["index"]:
        print(f"Skipping sample {sample_index} as per resume configuration.")
        continue
    
    
    if experiment['model_info']['modelType'] == 'patch2patch':
        from data_modules.opus_dataloader import all_data_handler
        train_daloader_fetcher, val_dataloader_fetcher, test_dataloader_fetcher = all_data_handler(experiment['not_corrected_for_compute'], experiment, idx=sample_index)
    elif experiment['model_info']['modelType'] == 'pre_computed_patch2patch':
        from data_modules.opus_CachingPatchDataset import cache_patch_dataloader
        from patchify.patchify_tio import create_patchify_dataset
        create_patchify_dataset(experiment)
        cache_dl = cache_patch_dataloader(experiment, DEVICE, idx=sample_index).fetch_allloaders(experiment)
        train_dataloader, val_dataloader, test_dataloader, intermediate_testing_loader = cache_dl['train'], cache_dl['validation'], cache_dl['test'], cache_dl['intermediate_testing']
    elif experiment['model_info']['modelType'] == 'vol2vol':
        pass
    else:
        raise ValueError(f"Unsupported model type: {experiment.model_info.model_type}. Expected types are {experiment.model_info.possibleTypes}.")

    # checkpoint_callback = LightningStyleCheckpoint(
    #     save_dir=f'logs/{exp_name}/checkpoints/{sample_index}',
    #     monitor='val_ssim',
    #     mode='max',
    #     top_k=experiment['checkpoint']['top_k'],
    #     save_every_n_epochs=experiment['checkpoint']['save_every_n_epochs'],
    #     save_weights_only=True,
    #     filename_template='{epoch:02d}-{monitor}={score:.4f}.pt'
    # )
    checkpoint_callback_2 = LightningStyleCheckpoint(
        save_dir=f'logs/{exp_name}/checkpoints/{sample_index}',
        monitor='val_loss',
        mode='min',
        top_k=experiment['checkpoint']['top_k'],
        save_every_n_epochs=experiment['checkpoint']['save_every_n_epochs'],
        # save_every_n_epochs=experiment['checkpoint']['save_every_n_epochs'],
        save_weights_only=True,
        filename_template='{epoch:02d}-{monitor}={score:.4f}.pt'
    )
    checkpoint_callback_val_loss = LightningStyleCheckpoint(
        save_dir=f'logs/{exp_name}/checkpoints/{sample_index}',
        monitor='val_loss',
        mode='min',
        top_k=experiment['checkpoint']['top_k'],
        save_every_n_epochs=2000000,
        # save_every_n_epochs=experiment['checkpoint']['save_every_n_epochs'],
        save_weights_only=False,
        filename_template='{epoch:02d}-{monitor}={score:.4f}.pt'
    )
    if avg_weighting:
        avg_annealing_checkpoint_callback = LightningStyleCheckpoint(
            save_dir=f'logs/{exp_name}/checkpoints_averaging/{sample_index}',
            monitor='val_loss',
            mode='min',
            # top_k=experiment['checkpoint']['top_k'],
            save_every_n_epochs=experiment['checkpoint']['save_every_n_epochs']//2,
            save_weights_only=True, 
            filename_template='{epoch:02d}-{monitor}={score:.4f}.pt'
        )
    scaler = GradScaler()
    loss_fn = compile_loss_fn(experiment)
    
    def get_linear_warmup_scheduler(optimizer, num_warmup_steps):
        def lr_lambda(current_step):
            return min(1.0, float(current_step) / float(max(1, num_warmup_steps)))
        return LambdaLR(optimizer, lr_lambda)

    # print(f"Evaluating sample {sample_index} {orig[0]['name']}")
    model = model_initializer(**model_loader_params).to(DEVICE)
    
    if experiment.get("finetunning", False):
        if experiment['finetunning']['do'] == True:
            ckpt = torch.load(experiment['finetunning']['weight_path'], weights_only=False)
            if "state_dict" not in ckpt:
                model.load_state_dict(ckpt)
            else:
                model.load_state_dict(ckpt["state_dict"])
            print("Initialized model to path ", experiment['finetunning']['weight_path'])
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=experiment['training']['learning_rate'], weight_decay=experiment['training']['weight_decay'])
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    if experiment['training']['epochs']>0:
        num_warmup_steps = int(len(train_dataloader) * experiment['training']['epochs']*experiment['training']['warmup_percentage_steps'])
        scheduler = get_linear_warmup_scheduler(optimizer, num_warmup_steps)



    if experiment.get("resume_training", {}).get('do', False):
            # ckpt = torch.load(experiment['finetunning']['weight_path'], weights_only=False)
            epoch, split_index = load_training_state(experiment['resume_training']['weight_path'], model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
            print("Initialized model to path ", experiment['resume_training']['weight_path'])
            if sdx<split_index:
                print("Skipping to match Split to weight. Skipping: ", sdx)
                continue
            csv_logger_train = CSVLogger(log_dir=f'logs/{exp_name}/metrics', filename=f'{sample_index}_metircs_train_.csv', resume=True)
            csv_logger_testing = CSVLogger(log_dir=f'logs/{exp_name}/metrics', filename=f'{sample_index}_metircs_testing_.csv',resume=True)
            
    else:        
        epoch = 0
        csv_logger_train = CSVLogger(log_dir=f'logs/{exp_name}/metrics', filename=f'{sample_index}_metircs_train_.csv', resume=False)
        csv_logger_testing = CSVLogger(log_dir=f'logs/{exp_name}/metrics', filename=f'{sample_index}_metircs_testing_.csv',resume=False)
        
    start_epoch = epoch
    total_epochs = experiment['training']['epochs']
    
    pbar = tqdm(range(start_epoch, total_epochs), desc=f"Epoch {start_epoch}")
    
    for epoch_idx in pbar:
        pbar.set_description(f"Epoch {epoch_idx}")
        loss_value = 0.0
        num_samples = 0
        model = model.float()
        model.train()
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Training", leave=False)):
            optimizer.zero_grad()
            if block_config.get('gin_apply', False):
                gin_net.reinit()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                if cache_mode == 'same_gpu':
                    source, target, mask = batch['source'],  batch['target'], batch['mask']
                else:
                    source, target, mask = batch['source'].to(DEVICE),  batch['target'].to(DEVICE), batch['mask'].to(DEVICE)
                if pre_post_norm:
                    source, _ = normalize_per_sample_masked(x=source, mask=mask, method = pre_post_norm_type)
                    # Fixed target optimiation
                    # target, _ = normalize_per_sample_masked(x=target, mask=mask, method = pre_post_norm_type)
                if block_augments:
                    source, aug_mask, weighted_loss_dict = augment(x = source, config=block_config, gin=gin_net)
                    output = model(source)
                    if weighted_loss_dict['weighted_loss']:
                        loss = weighted_loss_dict['weighted_loss_weight'][0]* loss_fn(output.float()*mask, target*mask) + weighted_loss_dict['weighted_loss_weight'][1]*loss_fn(output.float()*aug_mask, target*aug_mask)
                    else:
                        loss = loss_fn((output.float()*mask)*aug_mask, (target*mask)*aug_mask)
                else:                    
                    output = model(source)
                    # loss = 0.1* loss_fn(output.float()*mask, target*mask) + 1.0*loss_fn(output.float()*aug_mask, target*aug_mask)
                    loss = loss_fn(output.float()*mask, target*mask)

                  
                # output = model(batch['source'])
                # loss = loss_fn(output.float(), batch['target'])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss_value += loss.item() * source.shape[0]
            num_samples += source.shape[0]
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            # break
        loss_value /= num_samples
        writer.add_scalar(f"loss/{sdx}_train_loss", loss.item(), epoch_idx+1)
        # print("validation")
        model.eval()
        val_error, num_samples = 0.0, 0.0
        
        if len(val_dataloader):
            val_tqdm_loop = tqdm(val_dataloader, leave=False)
            # if epoch_idx > experiment['training']['no_val_till']:
            
            with torch.no_grad():
                for val_batch_idx, val_batch in enumerate(val_tqdm_loop):
                    with torch.autocast(device_type="cuda"):  # Enable FP16-safe inference
                        if cache_mode == 'same_gpu':
                            source, target, mask = batch['source'],  batch['target'], batch['mask']
                        else:
                            source, target, mask = batch['source'].to(DEVICE),  batch['target'].to(DEVICE), batch['mask'].to(DEVICE)
                        if pre_post_norm:
                            source, norm_dict = normalize_per_sample_masked(x=source, mask=mask, method = pre_post_norm_type)
                            # target, _ = normalize_per_sample_masked(x=target, mask=mask, method = pre_post_norm_type)
                        if block_augments:
                            source, aug_mask = augment(x = source, config=block_config, gin=gin_net)
                            val_output = model(source)
                            loss = loss_fn((val_output.float()*mask)*aug_mask, (target*mask)*aug_mask)
                        else:                    
                            val_output = model(source)
                            # loss = 0.1* loss_fn(output.float()*mask, target*mask) + 1.0*loss_fn(output.float()*aug_mask, target*aug_mask)
                            loss = loss_fn(val_output.float()*mask, target*mask)
                            
                        val_output = model(source)
                        val_loss = loss_fn(val_output.float(), target.float())
                    
                    # psnr, ssim = iqa_metrics.compute(val_output, real_target, val_batch['mask'])
                    val_error += val_loss.item() * source.shape[0] 
                    num_samples += source.shape[0]
                    # print("XXXXX", num_samples, "num_samples")
                val_loss = val_error / num_samples
                # metrics = iqa_metrics.value()
                # iqa_metrics.reset()
                val_tqdm_loop.set_description(f"Val Loss={val_loss:.4f}")
            writer.add_scalar(f"loss/{sdx}_val_loss", loss.item(), val_loss+1)

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
            if wand_boolean:
                wandb.log({
                    "epoch": epoch_idx + 1,
                    "split": sdx, 
                    f"train/loss_{sample_index}": loss_value,
                    f"val/loss_{sample_index}": val_loss,
                })
        else:
            metrics = {
                'epoch': epoch_idx + 1,
                'train_loss': loss_value,
            }
            if wand_boolean:
                wandb.log({ 
                "split": sdx, 
                "epoch": epoch_idx + 1,
                f"train/loss_{sample_index}": loss_value,
            })
        
        csv_logger_train.log(metrics)
        # checkpoint_callback.save(model, epoch, metrics)
        checkpoint_callback_2.save(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,step=epoch_idx, epoch=epoch_idx+1, split=sdx, logs = metrics)
        if len(val_dataloader):
            pass
            # checkpoint_callback_val_loss.save(model, epoch_idx+1, metrics)
        if avg_weighting:
            avg_annealing_checkpoint_callback.save(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,step=epoch, epoch=epoch_idx+1, logs = metrics)
        
        print("type(epoch_idx)", type(epoch_idx))
        # print({type(experiment["sample_volume_during_training"]["every"])}, type(experiment["sample_volume_during_training"]["every"]))
        
        if experiment.get("sample_volume_during_training",{}).get("do" , False):
            print('hehe', f"{epoch_idx+1}, {experiment['sample_volume_during_training']['every']}")
            print(epoch_idx+1% experiment['sample_volume_during_training']['every'])
            assert len(val_dataloader) > 0, "Make sure validation set exists to sample for validation"
            if epoch_idx+1%experiment["sample_volume_during_training"]["every"]:
                
                val_test_sample.test_sample(model=model, sample_idx=sdx, epoch=epoch_idx+1, test_dataloader=intermediate_testing_loader)
    
    if avg_weighting:
        average_model_weights(f'logs/{exp_name}/checkpoints_averaging/{sample_index}', f'logs/{exp_name}/checkpoints/{sample_index}', model_loader_params, model_initializer, f"{experiment['training']['epochs']}_{experiment['checkpoint']['save_every_n_epochs']}")
        
    all_weights = os.listdir(f"logs/{exp_name}/checkpoints/{sample_index}")
    all_weights = [os.path.join(f"logs/{exp_name}/checkpoints/{sample_index}", path) for path in all_weights]

    if experiment.get('only_evaluate_weigths_with',None):
        all_weights = [ele for ele in all_weights if experiment['only_evaluate_weigths_with'] in ele]
    
    # print(all_weights)
    # input("SS")
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
        # ckpt = torch.load(weight_path, weights_only=False)
        load_model_state(model=model, path= weight_path, device=DEVICE)
        # print(f"Loaded model from {weight_path}")
        
        
        # Evaluate the model on the test set
        model.eval()
        pbar = tqdm(test_dataloader) # The last sample was added to maintain
        for idx, test_subjects in enumerate(pbar):
            pbar.set_description(f"Evaluating {test_subjects['original'][0]['name']}")
            with torch.no_grad():
                for train_batch_idx, batch in enumerate(test_subjects['queue']):
                    with torch.autocast(device_type="cuda"):
                        source = batch['source']['data'].to(DEVICE)
                        mask = batch['mask']['data'].to(DEVICE)
                        if pre_post_norm:
                            normalize_per_sample_masked(source, mask, method=pre_post_norm_type)
                        # if block_augments:
                        #     source, aug_mask = augment(x = source, config=block_config, gin=gin_net)
                        #     fake_output = model(source)*aug_mask
                        # else:
                        if tta_flip_enabled:
                            fake_output = tta_forward(model, source * mask)
                        else:    
                            fake_output = model(source*mask)
                    test_subjects['aggregator'].add_batch((fake_output.float()), batch['location'])
                output = (test_subjects['aggregator'].get_output_tensor().to(DEVICE))*(test_subjects['original'][0]['mask']['data'].to(DEVICE))
                output = (output+1)/2
                output = torch.clip(output, min=0.0, max=1.0)
                
                # print(f"Output shape: {output.shape}")
                # print(f"Original shape: {orig[0]['target']['data'].shape}")
                # input()
                if block_augments:
                    test_metrics = iqa_metrics.compute(output.unsqueeze(1).float(), test_subjects['original'][0]['target']['data'].to(DEVICE).unsqueeze(1),test_subjects['original'][0]['mask']['data'].to(DEVICE))
                else:
                    test_metrics = iqa_metrics.compute(output.unsqueeze(1).float(), test_subjects['original'][0]['target']['data'].to(DEVICE).unsqueeze(1),test_subjects['original'][0]['mask']['data'].to(DEVICE))
                # affine = batch['source']['affine'][0]

                affine = test_subjects['original'][0]['target']['affine']
                
                image = tio.ScalarImage(tensor=output.float().cpu(), affine=affine)
                image.save(f"logs/{exp_name}/generations/{sample_index}_{test_subjects['original'][0]['name']}_{weight_prefix}_fake.nii.gz")
                test_subjects['original'][0]['target'].save(f"logs/{exp_name}/generations/{sample_index}_{test_subjects['original'][0]['name']}_orig.nii.gz")
                
        metrics = iqa_metrics.value()
        iqa_metrics.reset()
        
        print(f"Test Metrics for sample {sample_index}: PSNR: {metrics['PSNR']}, SSIM_3D: {metrics['SSIM_3D']},  SSIM_2D: {metrics['SSIM_2D']}, MAE: {metrics['MAE']}, MSE: {metrics['MSE']}")

        sample_test_metics = {
            'sample_index': sample_index,
            'weight_prefix': weight_prefix,
            'test_psnr': metrics['PSNR'],
            'test_ssim_3D': metrics['SSIM_3D'],
            'test_ssim_2D': metrics['SSIM_2D'],
            
            'test_mae': metrics['MAE'],
            'test_mse': metrics['MSE']
        }
        if wand_boolean:
            wandb.log(
                sample_test_metics
            )
        final_results.append(sample_test_metics)
        csv_logger_testing.log(sample_test_metics)

        
    del model    
    if experiment['training']['splits']['single_pass']:
        exit()