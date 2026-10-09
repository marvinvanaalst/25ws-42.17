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
import nibabel as nib
from utils_custom.tensor_operations import restore_original_batch, load_and_normalize_slices
from torch.optim.lr_scheduler import StepLR
from pl_models.models.pixGAN import NLayerDiscriminator as Discrimnator
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

avg_weighting = False
if avg_weighting:
    from merge_model_weights import average_model_weights

import kornia.augmentation as K
  
with open(experimnet_config, 'r') as f:
    experiment = json.load(f)

single_channel_model = experiment['model_info'].get('single_channel_model', False)


from utils_custom.eval_metrics import VolumeIQAMetrics

if experiment['model_info']['modelType'] == 'vol2vol':
    pass
elif experiment['model_info']['modelType'] == 'patch2patch' or experiment['model_info']['modelType'] == "pre_computed_patch2patch":
    from utils_custom.utils_opus import model_fetcher
    model_initializer, model_loader_params = model_fetcher(experiment).fetch_model()
    # model = patch2patch(experiment)
elif experiment['model_info']['modelType'] == 'pre_computed_slice2slice':
    from utils_custom.utils_opus import model_fetcher
    model_initializer, model_loader_params = model_fetcher(experiment).fetch_model()
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


if experiment['fine_tunning'].get('pretrain_evaluate_only', None) == True and experiment['fine_tunning'].get('pretrain_boolean', None)== True:
    experiment['training']['epochs'] = 0 

print("Total Number of samples", len(experiment['data']['training']['subjects']))
print("Total Splits", (len(experiment['data']['training']['subjects'])//experiment['training']['splits']['num_test_samples']))
import torch.nn as nn
for sdx in range(len(experiment['data']['training']['subjects'])//experiment['training']['splits']['num_test_samples']):
    sample_index = sdx
    if experiment["resume_from_sample"]["do"] and sample_index < experiment["resume_from_sample"]["index"]:
        print(f"Skipping sample {sample_index} as per resume configuration.")
        continue
    csv_logger_train = CSVLogger(log_dir=f'logs/{exp_name}/metrics', filename=f'{sample_index}_metircs_train_.csv')
    csv_logger_testing = CSVLogger(log_dir=f'logs/{exp_name}/metrics', filename=f'{sample_index}_metircs_testing_.csv')
    
    
    # Verify if the subject exists in the data directory
    # Create the data directory if it does not exist
    # Loading dataloaders
    if experiment['model_info']['modelType'] == 'patch2patch':
        from data_modules.opus_dataloader import all_data_handler
        train_daloader_fetcher, val_dataloader_fetcher, test_dataloader_fetcher = all_data_handler(experiment['not_corrected_for_compute'], experiment, idx=sample_index)
    elif experiment['model_info']['modelType'] == 'pre_computed_patch2patch':
        from data_modules.opus_CachingPatchDataset import cache_patch_dataloader
        from patchify.patchify_tio import create_patchify_dataset
        create_patchify_dataset(experiment)
        cache_dl = cache_patch_dataloader(experiment, DEVICE, idx=sample_index).fetch_allloaders()
        train_dataloader, val_dataloader, test_dataloader = cache_dl['train'], cache_dl['validation'], cache_dl['test']
    elif experiment['model_info']['modelType'] == 'pre_computed_slice2slice':
        from data_modules.opus_CachingSliceDataset import cache_slice_dataloader
        from patchify.slicify.slice_checker import create_slicify_dataset
        create_slicify_dataset(experiment)
        # Pinned: this script feeds batch['source'] to the model without a
        # .to(DEVICE), so the slices have to be cached on the GPU.
        cache_dl = cache_slice_dataloader(experiment, DEVICE, idx=sample_index, cache_mode='same_gpu')
        temp = cache_dl.fetch_TrainValLoaders()
        train_dataloader, val_dataloader, number_test_samples = temp['train'], temp['validation'], temp['number_test_samples']
        # print(f"Test subject name: {test_subject_name}")
        pass
    else:
        raise ValueError(f"Unsupported model type: {experiment.model_info.model_type}. Expected types are {experiment.model_info.possibleTypes}.")




    checkpoint_callback = LightningStyleCheckpoint(
        save_dir=f'logs/{exp_name}/checkpoints/{sample_index}',
        monitor='val_ssim',
        mode='max',
        top_k=experiment['checkpoint']['top_k'],
        save_every_n_epochs=experiment['checkpoint']['save_every_n_epochs'],
        save_weights_only=True,)
     
    
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
    checkpoint_callback_3 = LightningStyleCheckpoint(
        save_dir=f'logs/{exp_name}/checkpoints/{sample_index}',
        monitor=['val_loss','train_loss'],
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
    scalerG = GradScaler()
    scalerD = GradScaler()

    loss_fn = compile_loss_fn(experiment)
    
    criterionGAN = nn.BCEWithLogitsLoss()
    # print(f"Evaluating sample {sample_index} {orig[0]['name']}")
    print(model_loader_params)
    if len(model_loader_params.keys()) == 0:
        model = model_initializer().to(DEVICE)
    else:
        model = model_initializer(**model_loader_params).to(DEVICE)
    from monai.networks.nets import PatchDiscriminator

# Example configuration for 3D MRI (3T/7T)
# in_channels=2 if you are concatenating the source and target scans
# spatial_dims=3 for volumetric data
    # discimnator_model = PatchDiscriminator(
    #     spatial_dims=1,
    #     in_channels=2, 
    #     num_channels=64, 
    #     num_layers=3
    # ).to(DEVICE)
    
    discimnator_model = PatchDiscriminator(
    spatial_dims=2,
    in_channels=1,
    out_channels=1, 
     
    channels=64, 
    num_layers_d=3
    ).to(DEVICE)
    
        # 1. Define hyperparameters
    lr = 2e-4
    epochs_constant = 50
    epochs_decay = experiment['training']['epochs'] - 50  # Decay over the remaining 50 epochs
    total_epochs = epochs_constant + epochs_decay


    lr_lambda = lambda epoch: 1.0 if epoch < epochs_constant else \
                            max(0, 1.0 - (epoch - epochs_constant) / float(epochs_decay))
                            

    opt_g = torch.optim.Adam(
        model.parameters(), 
        lr=lr, 
        betas=(0.5, 0.999) # Standard for GANs
    )
    opt_d = torch.optim.Adam(
        discimnator_model.parameters(), 
        lr=lr, 
        betas=(0.5, 0.999) # Standard for GANs
    )
    # 3. Define the Lambda function
    # This returns a multiplier for the initial learning rate
    def lr_lambda(epoch):
        if epoch < epochs_constant:
            return 1.0
        else:
            # Linearly decay to 0
            return 1.0 - (epoch - epochs_constant) / float(epochs_decay + 1)
    from torch.optim.lr_scheduler import LambdaLR                  
    sched_g = LambdaLR(opt_g, lr_lambda=lr_lambda)
    sched_d = LambdaLR(opt_d, lr_lambda=lr_lambda)
    # 2. Setup Optimizer
    # 4. Setup Scheduler
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    # optimizer = torch.optim.Adam(model.parameters(), lr=experiment['training']['learning_rate'], weight_decay=experiment['training']['weight_decay'])
    # discimnator_model = Discrimnator().to(DEVICE)
    generator_model = model
    
    
    # opt_g = torch.optim.RMSprop(generator_model.parameters(), lr=experiment['training']['learning_rate'], weight_decay=experiment['training']['weight_decay'])
    # opt_d = torch.optim.RMSprop(discimnator_model.parameters(), lr=experiment['training']['learning_rate'])
    
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    if experiment['fine_tunning'].get('pretrain_boolean', None) == True:
            if experiment['fine_tunning'].get('pretrained_model_weight', None) is not None:
                ckpt = torch.load(experiment['fine_tunning'].get('pretrained_model_weight',""), weights_only=False)
                if "state_dict" not in ckpt:
                    model.load_state_dict(ckpt)
                else:
                    model.load_state_dict(ckpt["state_dict"])
    
    epoch = 0
    # augmentation_boolean = experiment['augmentations'].get('apply', False)


    for epoch_idx, _ in enumerate(tqdm(range(experiment['training']['epochs']) , desc=f"Epoch {epoch}")):
        loss_value = 0.0
        num_samples = 0
        model = model.float()
        discimnator_model.train()
        generator_model.train()

        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Training", leave=False)):
            source = batch['source'] * batch['mask']
            target = batch['target'] * batch['mask']

            # =========== Train Discriminator ===========
            opt_d.zero_grad() # Fixed: Was opt_g.zero_grad()
            with autocast():
                # 1. Real Loss
                pred_real = discimnator_model(target)
                # Using [-1] because PatchDiscriminator often returns a list of feature maps
                loss_D_real = criterionGAN(pred_real[-1], torch.ones_like(pred_real[-1]))
                
                # 2. Fake Loss
                # Fixed: Generator should take SOURCE, not TARGET
                fake_images = generator_model(source).detach() 
                pred_fake = discimnator_model(fake_images)
                loss_D_fake = criterionGAN(pred_fake[-1], torch.zeros_like(pred_fake[-1]))
                
                d_loss = (loss_D_real + loss_D_fake) * 0.5

            scalerD.scale(d_loss).backward()
            scalerD.step(opt_d)
            scalerD.update()
            
            # =========== Train Generator ===========
            opt_g.zero_grad()
            with autocast():
                # Re-generate fake images (without detach) so gradients flow to G
                output = generator_model(source) 
                pred_fake_g = discimnator_model(output)
                
                # Adversarial: G wants D to think fake is Real (target 1)
                adv_loss = criterionGAN(pred_fake_g[-1], torch.ones_like(pred_fake_g[-1]))
                
                # L1 Reconstruction
                recon_loss = nn.functional.l1_loss(output, target)
                
                # Total Loss (Lambda 100 on L1 is standard for Pix2Pix/ResViT)
                total_g_loss = 100 * recon_loss + adv_loss

            scalerG.scale(total_g_loss).backward()
            scalerG.step(opt_g)
            scalerG.update()
            
            loss_value += total_g_loss.item() * source.size(0)
            num_samples += source.size(0)
        sched_g.step()
        sched_d.step()        

        loss_value /= num_samples
        # scheduler.step()
        # print("validation")
        model.eval()
        val_error, num_samples = 0.0, 0.0
        val_loss = 0
        val_tqdm_loop = tqdm(val_dataloader, leave=False)
        # if epoch_idx > experiment['training']['no_val_till']:
        discimnator_model.eval()
        generator_model.eval()
        if len(val_dataloader):
            with torch.no_grad():
                for val_batch_idx, val_batch in enumerate(val_tqdm_loop):
                    with torch.cuda.amp.autocast():  # Enable FP16-safe inference
                        if single_channel_model:
                            val_output = model(val_batch['source'].squeeze(1)).unsqueeze(1)  # Remove channel dimension if single channel
                        else:
                            val_output = model(val_batch['source'])
                            
                        val_loss = 100* loss_fn(val_output.float(), val_batch['target'].float())
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
    
    original_size = None
    
    for wdx, weight_path in enumerate(weight_tqdm_loop):
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
        testing_loop_pwe_weight = tqdm(range(number_test_samples), desc=f"Testing {weight_prefix}", leave=False)
        for idxx, i in enumerate(testing_loop_pwe_weight):
            sample_index_w = i
            test_dataloader = cache_dl.fetch_testLoader(idx=i)
            test_subject_name = test_dataloader['test_name']
            
            testing_loop_pwe_weight.set_description(f"Testing {weight_prefix} Scan: {test_subject_name}")
            
            test_dataloader = test_dataloader['test']
            
            for train_batch_idx, batch in enumerate(test_dataloader):
                original_size = batch['original_size'][0]
                original_path_scan = batch['path_scan'][0]
                original_path_mask = batch['path_mask'][0]
                break
            
            original_path_scan = original_path_scan.replace(experiment['data']['direction']['source'], experiment['data']['direction']['target'])
            # input(f"Original path scan: {original_path_scan}")
            scan_orig_full = nib.load(original_path_scan)
            mask_orig = torch.from_numpy(nib.load(original_path_mask).get_fdata(dtype=np.float32)).float().to(DEVICE)
            scan_orig = load_and_normalize_slices(experiment, original_path_scan, dtype=torch.float32 ).to(DEVICE)
            scan_orig_div = (scan_orig* mask_orig)
            
            new_img = nib.Nifti1Image(scan_orig_div.squeeze(0).cpu().numpy(), affine=scan_orig_full.affine)
            nib.save(new_img, f"logs/{exp_name}/generations/Split{sdx}_{sample_index_w}_{test_subject_name}_original.nii.gz")
            # input('check the original scan created')
        
            pad_k = experiment['slicify']['pad_to']
            resize_k = experiment['slicify']['resize_to']
            with torch.no_grad():
                train_target_fake = []
                for train_batch_idx, batch in enumerate(test_dataloader):
                    with torch.cuda.amp.autocast():
                        if single_channel_model:
                            fake_output = model(batch['source'].squeeze(1)).unsqueeze(1)  # Remove channel dimension if single channel
                        else:
                            fake_output = model(batch['source'])
                        
                    fake_output = restore_original_batch(fake_output, batch['padding'], batch['original_size'], pad_k, resize_k)
                    # print(f"Fake output shape: {fake_output.shape}")
                    train_target_fake.append(fake_output.squeeze(1))  # Remove channel dimension if single channel
                    # fake_output = fake_output*(batch['mask']['data'].to(DEVICE))
                
                train_target_fake = torch.cat(train_target_fake, dim=0).permute(1, 2, 0).contiguous()
                # print(mask_orig.shape, train_target_fake.shape, batch['original_size'])
                # breakpoint()
                
                train_target_fake = train_target_fake * mask_orig
                train_target_fake = torch.clip(train_target_fake, min=0.0, max=1.0)
                            
                # print(f"Output shape: {output.shape}")
                # print(f"Original shape: {orig[0]['target']['data'].shape}")
                # input()
                # print(f"Evaluating sample {sample_index} {test_subject_name} with weight {weight_prefix}")
                # print(f"Output shape: {train_target_fake.shape}, Target shape: {train_real_target.shape}, Mask shape: {test_mask.shape}")
                # print("train_target_fake", train_target_fake.shape, "scan_orig_div::", scan_orig_div.shape)
                test_metrics = iqa_metrics.compute(train_target_fake, scan_orig_div)
                
                affine = batch['affine'][0].cpu().numpy().astype(np.float32)
                output_img = nib.Nifti1Image(train_target_fake.squeeze(0).cpu().numpy().astype(np.float32), affine)
                nib.save(output_img, f"logs/{exp_name}/generations/Split{sdx}_{sample_index_w}_{test_subject_name}_{weight_prefix}_fake.nii.gz")
                
        metrics = iqa_metrics.value()
        iqa_metrics.reset() 
        print(f"Test Metrics for sample {sample_index_w}: PSNR: {metrics['PSNR']}, SSIM_3D: {metrics['SSIM_3D']},  SSIM_2D: {metrics['SSIM_2D']}, MAE: {metrics['MAE']}, MSE: {metrics['MSE']}")
        sample_test_metics = {
            'sample_index': sample_index_w,
            'weight_prefix': weight_prefix,
            'test_psnr': metrics['PSNR'],
            'test_ssim_3D': metrics['SSIM_3D'],
            'test_ssim_2D': metrics['SSIM_2D'],
            
            'test_mae': metrics['MAE'],
            'test_mse': metrics['MSE']
        }
        
        final_results.append(sample_test_metics)
        csv_logger_testing.log(sample_test_metics)


            
        
    del model
    print("Training and evaluation completed for sample index:", sample_index)
    if experiment['training']['splits']['single_pass']:
        exit()