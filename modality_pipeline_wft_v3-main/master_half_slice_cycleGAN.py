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
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import save_image
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
    checkpoint_callback_fid = LightningStyleCheckpoint(
        save_dir=f'logs/{exp_name}/checkpoints/{sample_index}',
        monitor='fid',
        mode='min',
        top_k=experiment['checkpoint']['top_k'],
        save_every_n_epochs=10000,
        # save_every_n_epochs=experiment['checkpoint']['save_every_n_epochs'],
        
        save_weights_only=True,
        filename_template='{epoch:02d}-{monitor}={score:.4f}.pt'
    )
    checkpoint_callback_genLead = LightningStyleCheckpoint(
        save_dir=f'logs/{exp_name}/checkpoints/{sample_index}',
        monitor='loss_2MR',
        mode='min',
        top_k=experiment['checkpoint']['top_k'],
        save_every_n_epochs=10000,
        # save_every_n_epochs=experiment['checkpoint']['save_every_n_epochs'],
        
        save_weights_only=True,
        filename_template='{epoch:02d}-{monitor}={score:.4f}.pt'
    )
    checkpoint_callback_dis = LightningStyleCheckpoint(
        save_dir=f'logs/{exp_name}/m2/checkpoints/{sample_index}',
        monitor='loss_2CT',
        mode='min',
        top_k=experiment['checkpoint']['top_k'],
        save_every_n_epochs=10000,
        # save_every_n_epochs=experiment['checkpoint']['save_every_n_epochs'],
        
        save_weights_only=True,
        filename_template='{epoch:02d}-{monitor}={score:.4f}.pt'
    )
    checkpoint_callback_ssim = LightningStyleCheckpoint(
        save_dir=f'logs/{exp_name}/checkpoints/{sample_index}',
        monitor='ssim',
        mode='max',
        top_k=experiment['checkpoint']['top_k'],
        save_every_n_epochs=10000,
        # save_every_n_epochs=experiment['checkpoint']['save_every_n_epochs'],
        
        save_weights_only=True,
        filename_template='{epoch:02d}-{monitor}={score:.4f}.pt'
    )
    checkpoint_callback_psnr = LightningStyleCheckpoint(
        save_dir=f'logs/{exp_name}/checkpoints/{sample_index}',
        monitor='psnr',
        mode='max',
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
    print(model_loader_params)
    if len(model_loader_params.keys()) == 0:
        model = model_initializer().to(DEVICE)
        model_2 = model_initializer().to(DEVICE)

    else:
        model = model_initializer(**model_loader_params).to(DEVICE)
        model_2 = model_initializer(**model_loader_params).to(DEVICE)
        
    from pl_models.models.PatchGANDiscrimnator import Discriminator
    
    dis_CT2MR = Discriminator().to(DEVICE)
    dis_MR2CT = Discriminator().to(DEVICE)
        
    print("\n\n\n\n\n\n Set cycle GAN LRs.")
    optimizer_gen_CT2MR = torch.optim.Adam(model.parameters(), lr=experiment['training']['learning_rate'], weight_decay=experiment['training']['weight_decay'])
    optimizer_gen_MR2CT = torch.optim.Adam(model_2.parameters(), lr=experiment['training']['learning_rate'], weight_decay=experiment['training']['weight_decay'])
    
    optimizer_dis_CT2MR = torch.optim.Adam(dis_CT2MR.parameters(), lr=experiment['training']['learning_rate'], weight_decay=experiment['training']['weight_decay'])
    optimizer_dis_MR2CT = torch.optim.Adam(dis_MR2CT.parameters(), lr=experiment['training']['learning_rate'], weight_decay=experiment['training']['weight_decay'])


    
    def lambda_rule(epoch):
        start_decay = 20   # epoch to start decay
        total_epochs = 150
        if epoch < start_decay:
            return 1.0
        else:
            return 1.0 - (epoch - start_decay) / (total_epochs - start_decay)

    
    def lambda_rule(epoch):
        return 0.1 ** (epoch // 10)

    scheduler_gen_CT2MR = LambdaLR(optimizer_gen_CT2MR, lr_lambda=lambda_rule)
    scheduler_gen_MR2CT = LambdaLR(optimizer_gen_MR2CT, lr_lambda=lambda_rule)
    scheduler_dis_CT2MR = LambdaLR(optimizer_dis_CT2MR, lr_lambda=lambda_rule)
    scheduler_dis_MR2CT = LambdaLR(optimizer_dis_MR2CT, lr_lambda=lambda_rule)
    
        
    scaler_gen = GradScaler()
    scaler_dis_CT2MR = GradScaler()
    scaler_dis_MR2CT = GradScaler()
    
    l2_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid_metric = FrechetInceptionDistance(feature=2048).to(DEVICE)
    if experiment['fine_tunning'].get('pretrain_boolean', None) == True:
            if experiment['fine_tunning'].get('pretrained_model_weight', None) is not None:
                ckpt = torch.load(experiment['fine_tunning'].get('pretrained_model_weight',""), weights_only=False)
                if "state_dict" not in ckpt:
                    model.load_state_dict(ckpt)
                else:
                    model.load_state_dict(ckpt["state_dict"])
    
    epoch = 0
    for epoch_idx, _ in enumerate(tqdm(range(experiment['training']['epochs']) , desc=f"Epoch {epoch}")):
        loss_2MR, loss_2CT, loss_D2MR, loss_D2CT = 0.0, 0.0, 0.0, 0.0
        num_samples = 0
        model = model.float()
        model.train()
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Training", leave=False)):
            with autocast():

                optimizer_dis_CT2MR.zero_grad()
                optimizer_dis_MR2CT.zero_grad()
                
                
                
                fake_mri = model(batch['source'])
                D_MRI_real = dis_CT2MR(batch['target'])
                D_MRI_fake = dis_CT2MR(fake_mri.detach())
                # H_reals += D_MRI_real.mean().item()
                # H_fakes += D_MRI_fake.mean().item()
                D_MRI_real_loss = l2_loss(D_MRI_real, torch.ones_like(D_MRI_real))
                D_MRI_fake_loss = l2_loss(D_MRI_fake, torch.zeros_like(D_MRI_fake))
                D_CT2MR_loss = 0.5 *(D_MRI_real_loss + D_MRI_fake_loss)

                fake_ct = model_2(batch['target'])
                D_CT_real = dis_MR2CT(batch['source'])
                D_CT_fake = dis_MR2CT(fake_ct.detach())
                D_CT_real_loss = l2_loss(D_CT_real, torch.ones_like(D_CT_real))
                D_CT_fake_loss = l2_loss(D_CT_fake, torch.zeros_like(D_CT_fake))
                D_MR2CT_loss = 0.5 *(D_CT_fake_loss + D_CT_real_loss)
                    

                
            scaler_dis_MR2CT.scale(D_MR2CT_loss).backward()
            scaler_dis_MR2CT.step(optimizer_dis_MR2CT)
            scaler_dis_MR2CT.update()
            
            scaler_dis_CT2MR.scale(D_CT2MR_loss).backward()
            scaler_dis_CT2MR.step(optimizer_dis_CT2MR)
            scaler_dis_CT2MR.update()
            
            loss_D2MR += D_CT2MR_loss.item() * batch['source'].shape[0]
            loss_D2CT += D_MR2CT_loss.item() * batch['source'].shape[0]
            num_samples += batch['source'].shape[0]
                
                # Train Generators H and Z
                # with torch.cuda.amp.autocast():
                    # adversarial loss for both generators
                    
                    # D_CT_real = dis_MR2CT(batch['source']) means discrimnating with CT as ground truth. In M1 2 M2 Notation, M2 is the modality of which it is a discrimnator.
            
            with autocast():
                  
                optimizer_gen_MR2CT.zero_grad()
                optimizer_gen_CT2MR.zero_grad()
                
                # Adversarial
                fake_mri = model(batch['source'])
                fake_ct = model_2(batch['target'])
                
                
                loss_G_MRI = l2_loss(dis_CT2MR(fake_mri), torch.ones_like(D_MRI_fake))
                loss_G_CT = l2_loss(dis_MR2CT(fake_ct), torch.ones_like(D_CT_fake))
                # print("CT_GEN_LOSS", loss_G_CT, "MRI_GEN_LOSS", loss_G_MRI)

                # cycle loss
                cycle_ct = model_2(fake_mri)
                cycle_mri = model(fake_ct)
                cycle_ct_loss = l1_loss(batch['source'], cycle_ct)
                cycle_mri_loss = l1_loss(batch['target'], cycle_mri)

                # identity loss (remove these for efficiency if you set lambda_identity=0)
                identity_ct = model_2(batch['source'])
                identity_mri = model(batch['target'])
                identity_ct_loss = l1_loss(batch['source'], identity_ct)
                identity_mri_loss = l1_loss(batch['target'], identity_mri)
                # print("IdentityCTLOSS", identity_ct_loss, "IdentityMRILoss", identity_mri_loss)

            # add all togethor
            G_CT_loss = (
                loss_G_CT * experiment['Cycle_GAN']['loss_G_1']
                + cycle_ct_loss * experiment['Cycle_GAN']['LAMBDA_CYCLE']
                + identity_ct_loss * experiment['Cycle_GAN']['LAMBDA_IDENTITY']
            )
            G_MR_Loss = (
                loss_G_MRI * experiment['Cycle_GAN']['loss_G_2']
                + cycle_mri_loss * experiment['Cycle_GAN']['LAMBDA_CYCLE']
                + identity_mri_loss * experiment['Cycle_GAN']['LAMBDA_IDENTITY']
            )
            
            G_loss = G_CT_loss + G_MR_Loss

            
            scaler_gen.scale(G_loss).backward()
            
            scaler_gen.step(optimizer_gen_CT2MR)
            scaler_gen.step(optimizer_gen_MR2CT)
            
            scaler_gen.update()
            
            loss_2MR += G_MR_Loss.item() * batch['source'].shape[0]
            loss_2CT += G_CT_loss.item() * batch['source'].shape[0]
            
        # scheduler_gen_CT2MR.step()
        # scheduler_gen_MR2CT.step()
        # scheduler_dis_CT2MR.step()
        # scheduler_dis_MR2CT.step()   
            
        loss_2MR, loss_2CT, loss_D2MR, lossD2CT = loss_2MR/num_samples, loss_2CT/num_samples, loss_D2MR/num_samples, loss_D2CT/num_samples 

        # print("validation")
        model.eval()
        val_error, num_samples = 0.0, 0.0
        fid = 0
        val_tqdm_loop = tqdm(val_dataloader, leave=False)
        # if epoch_idx > experiment['training']['no_val_till']:
        with torch.no_grad():
            for val_batch_idx, val_batch in enumerate(val_tqdm_loop):
                with torch.cuda.amp.autocast():  # Enable FP16-safe inference
                    if single_channel_model:
                        val_output = model(val_batch['source'].squeeze(1)).unsqueeze(1)  # Remove channel dimension if single channel
                    else:
                        val_output = model(val_batch['source'])
                        
                    val_loss = l2_loss(val_output.float(), val_batch['target'].float())
                    fid_metric.update((val_batch['target']*255).clamp(0,255).to(torch.uint8).repeat(1,3,1,1), real=True)
                    fid_metric.update((val_output*255).clamp(0,255).to(torch.uint8).repeat(1,3,1,1), real=False)
                # psnr, ssim = iqa_metrics.compute(val_output, val_batch['target'], val_batch['mask'])
                val_error += val_loss.item() * batch['source'].shape[0]
                num_samples += val_batch['source'].shape[0]
            val_loss = val_error / num_samples
            fid_score = fid_metric.compute().item()
            fid_metric.reset()
            # metrics = iqa_metrics.value()
            # iqa_metrics.reset()
            val_tqdm_loop.set_description(f"Val Loss={val_loss:.4f}")
        epoch += 1
        os.makedirs(f"logs/{exp_name}/weight_generations/",exist_ok=True)
        save_image(val_output, f"logs/{exp_name}/weight_generations/{epoch_idx}.png", normalize=True, nrow=8)
        metrics = {
            'epoch': epoch_idx + 1,
            # 'train_loss': loss_value,
            'val_loss': val_loss,
            'loss_2MR' : loss_2MR,
            'loss_2CT' : loss_2CT,
            'loss_D2MR' : loss_D2MR,
            'loss_D2CT' : loss_D2CT,
            'val_loss_mse' : val_loss,
            'fid': fid_score
            # 'val_psnr': metrics['PSNR'],
            # 'val_ssim': metrics['SSIM'],
            # 'val_mae': metrics['MAE'],
            # 'val_mse': metrics['MSE'],
            # 'val_mmsim': metrics['MaskedSSIM']
        }
            
        # if wand_boolean:
        #     wandb.log(metrics)
        csv_logger_train.log(metrics)
        # checkpoint_callback.save(model, epoch, metrics)
        checkpoint_callback_2.save(model, epoch, metrics)
        checkpoint_callback_fid.save(model, epoch, metrics)
        checkpoint_callback_dis.save(model_2, epoch, metrics)
        checkpoint_callback_genLead.save(model, epoch, metrics)
        # checkpoint_callback_ssim.save(model, epoch, metrics)
        # checkpoint_callback_psnr.save(model, epoch, metrics)
        
        if avg_weighting:
            avg_annealing_checkpoint_callback.save(model, epoch, metrics)
    
    if avg_weighting:
        average_model_weights(f'logs/{exp_name}/checkpoints_averaging/{sample_index}', f'logs/{exp_name}/checkpoints/{sample_index}', model_loader_params, model_initializer, f"{experiment['training']['epochs']}_{experiment['checkpoint']['save_every_n_epochs']}")
    all_weights = os.listdir(f"logs/{exp_name}/checkpoints/{sample_index}")
    all_weights = [os.path.join(f"logs/{exp_name}/checkpoints/{sample_index}", path) for path in all_weights]

    weight_tqdm_loop = tqdm(all_weights, desc="Evaluating Weights", leave=False)
    
    original_size = None
    
    # input("Filter weights before evaluating 1/4")
    # input("Filter weights before evaluating 2/4")
    # input("Filter weights before evaluating 3/4")
    # input("Filter weights before evaluating 4/4")
    
    
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
            sample_index = i
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
            scan_orig = load_and_normalize_slices(original_path_scan, axis=2, dtype=torch.float32).to(DEVICE)
            scan_orig_div = (scan_orig* mask_orig)
            
            new_img = nib.Nifti1Image(scan_orig_div.squeeze(0).cpu().numpy(), affine=scan_orig_full.affine)
            nib.save(new_img, f"logs/{exp_name}/generations/{sdx}_{sample_index}_{test_subject_name}_original.nii.gz")
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
                nib.save(output_img, f"logs/{exp_name}/generations/{sdx}_{sample_index}_{test_subject_name}_{weight_prefix}_fake.nii.gz")
                
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
        
        
        final_results.append(sample_test_metics)
        csv_logger_testing.log(sample_test_metics)


            
        
    del model
    print("Training and evaluation completed for sample index:", sample_index)