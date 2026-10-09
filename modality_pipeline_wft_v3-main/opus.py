from utils_custom.utils_opus import model_fetcher

import torch
import json
# from box import Box
import sys
import csv
import torchio as tio
import warnings
warnings.simplefilter("default")  # or "always"
import os
from tqdm import tqdm
wand_boolean = False
print("WAND DB BOOLEAN", wand_boolean)
from utils_custom.csv_logger import CSVLogger
from utils_custom.LightningStyleCheckpoint import LightningStyleCheckpoint
argv = sys.argv
if len(argv) < 2:
    raise ValueError("Expected argv config file. Using default configuration.")
    
else:
    experimnet_config = argv[1]

with open('/Drive4T/inam/3T_7T_Synthesis/data.json', 'r') as f:
    params_data = json.load(f)    
    
with open(experimnet_config, 'r') as f:
    experiment = json.load(f)
    
# input(experiment)

from utils_custom.eval_metrics import VolumeIQAMetrics

# Dataloader 

##??
if experiment['model_info']['modelType'] == 'vol2vol':
    pass
elif experiment['model_info']['modelType'] == 'patch2patch':
    from utils_custom.utils_opus import model_fetcher
    model_initializer, model_loader_params = model_fetcher(experiment).fetch_model()
    # model = patch2patch(experiment)
else:
    raise ValueError(f"Unsupported model type: {experiment.model.type}. Expected types are '3DUNET' or 'VNet'.")


# from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger
# from pytorch_lightning.loggers import CSVLogger, WandbLogger

# from pytorch_lightning import Callback

# Init DataModule and Model
# Callbacks
exp_name=f'{experiment["experiment_name"]}'
# csv_logger = CSVLogger(save_dir='logs',  name=f'{exp_name}/csv_logs')

final_results = []


dir_path = "/path/to/directory"

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

DEVICE = experiment['device']['devices'][0]
print(f"Using device: {DEVICE}")

iqa_metrics = VolumeIQAMetrics(data_range=1.0, device = DEVICE)

if experiment['resume_from_sample']['do']:
    csv_logger_average = CSVLogger(log_dir=f'logs/{exp_name}/metrics', filename=f'average_testing_.csv', resume=True)
else:
    csv_logger_average = CSVLogger(log_dir=f'logs/{exp_name}/metrics', filename=f'average_testing_.csv', resume=False)
    
for sdx in range(len(params_data['data']['training']['subjects'])):
    sample_index = sdx
    if experiment["resume_from_sample"]["do"] and sample_index < experiment["resume_from_sample"]["index"]:
        print(f"Skipping sample {sample_index} as per resume configuration.")
        continue
    csv_logger_train = CSVLogger(log_dir=f'logs/{exp_name}/metrics', filename=f'{sample_index}_metircs_train_.csv')
    csv_logger_testing = CSVLogger(log_dir=f'logs/{exp_name}/metrics', filename=f'{sample_index}_metircs_testing_.csv')
    
    
    if experiment['model_info']['modelType'] == 'patch2patch':
        from data_modules.opus_dataloader import all_data_handler
        train_daloader_fetcher, val_dataloader_fetcher, test_dataloader_fetcher = all_data_handler(params_data, experiment, idx=sample_index)
    elif experiment['model_info']['modelType'] == 'patch2patch_caching':
        from data_modules.opus_CachingPatchDataset import all_data_handler
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
        save_every_n_epochs=experiment['checkpoint']['save_every_n_epochs'],
        save_weights_only=True,
        filename_template='{epoch:02d}-{monitor}={score:.4f}.pt'
    )
    if experiment['loss']['name'] == 'L1':
        loss_fn = torch.nn.L1Loss()
    
    train_daloader = train_daloader_fetcher.fetch_loader()
    val_daloader = val_dataloader_fetcher.fetch_loader()
    test_dataloader, aggregator, orig = test_dataloader_fetcher.fetch_loader_and_sampler()
    
    print(f"Evaluating sample {sample_index} {orig[0]['name']}")
    
    model = model_initializer(model_loader_params).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=experiment['training']['learning_rate'])
    
    
    epoch = 0
    for epoch_idx, _ in enumerate(tqdm(range(experiment['training']['epochs']) , desc=f"Epoch {epoch}")):
        loss_value = 0.0
        num_samples = 0
        model.train()
        for batch_idx, batch in enumerate(tqdm(train_daloader, desc=f"Training", leave=False)):
            output = model(batch['source']['data'].to(DEVICE))
            loss = loss_fn(output, batch['target']['data'].to(DEVICE))
            loss.backward()
            loss_value += loss.item()
            num_samples += batch['source']['data'].shape[0]
            optimizer.step()
            optimizer.zero_grad()
        loss_value /= num_samples

        # print("validation")
        model.eval()
        val_error, num_samples = 0.0, 0.0
        val_tqdm_loop = tqdm(val_daloader, leave=False)
        for val_batch_idx, val_batch in enumerate(val_tqdm_loop):
            with torch.no_grad():
                val_output = model(val_batch['source']['data'].to(DEVICE))
                val_loss = loss_fn(val_output, val_batch['target']['data'].to(DEVICE))
                real_target = val_batch['target']['data'].to(DEVICE)
                psnr, ssim = iqa_metrics.compute(val_output, real_target)
                val_error += val_loss.item()
                num_samples += val_batch['source']['data'].shape[0]
            val_loss = val_error / num_samples
            metrics = iqa_metrics.value()
            iqa_metrics.reset()
            val_tqdm_loop.set_description(f"Val PSNR={metrics['PSNR']:.4f} SSIM={metrics['SSIM']:.4f}")
        epoch += 1

        metrics = {
            'epoch': epoch_idx + 1,
            'train_loss': loss_value,
            'val_loss': val_loss,
            'val_psnr': metrics['PSNR'],
            'val_ssim': metrics['SSIM'],
            'val_mae': metrics['MAE'],
            'val_mse': metrics['MSE']
        }
        wandb.log({
            "epoch": epoch_idx + 1,
            f"train/loss_{sample_index}": loss_value,
            f"val/loss_{sample_index}": val_loss,
            f"val/psnr_{sample_index}":  metrics['PSNR'],
            f"val/ssim_{sample_index}":  metrics['SSIM'],
            f"val/mae_{sample_index}":  metrics['MAE'],
            f"val/mse_{sample_index}":  metrics['MSE']
        })
        csv_logger_train.log(metrics)
        checkpoint_callback.save(model, epoch, metrics)
        checkpoint_callback_2.save(model, epoch, metrics)
        
    all_weights = os.listdir(f"logs/{exp_name}/checkpoints/{sample_index}")
    all_weights = [os.path.join(f"logs/{exp_name}/checkpoints/{sample_index}", path) for path in all_weights]

    weight_tqdm_loop = tqdm(all_weights, desc="Evaluating Weights", leave=False)
    for wdx, weight_path in enumerate(weight_tqdm_loop):
        
        # print(weight_path)
        # input()
        for ele in weight_path.split('/'):
            if 'epoch' in ele:
                epoch = int(ele.split('=')[1])
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
            for train_batch_idx, batch in enumerate(test_dataloader):
                fake_output = model(batch['source']['data'].to(DEVICE))
                aggregator.add_batch(fake_output, batch['location'])
            output = aggregator.get_output_tensor().to(DEVICE)*orig[0]['mask']['data'].to(DEVICE)
            output = torch.clip(output, min=0.0, max=1.0)
            
            # print(f"Output shape: {output.shape}")
            # print(f"Original shape: {orig[0]['target']['data'].shape}")
            # input()
            test_metrics = iqa_metrics.compute(output.unsqueeze(1), orig[0]['target']['data'].to(DEVICE).unsqueeze(1))
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
            image.save(f"logs/{exp_name}/generations/{sample_index}_{orig[0]['name']}_{weight_prefix}_fake.nii.gz")
            orig[0]['target'].save(f"logs/{exp_name}/generations/{sample_index}_{orig[0]['name']}_orig.nii.gz")
            
        
    del model
#         # Save the output
#         output_tensor = aggregator.get_output_tensor()
#         affine = train_batch['source']['affine'][0]
#         image = tio.ScalarImage(tensor=output_tensor, affine=affine)
#         file_name = train_batch['source']['file_name'][0]
#         image.save(f"logs/{exp_name}/test_generation/{sample_index}_{file_name}_Synthetic_{params_data['direction']['target']}.nii.gz")
#         orig[0]['target'].save(f"logs/{exp_name}/test_generation/{sample_index}_{file_name}_Real_{params_data['direction']['target']}.nii.gz")
#         orig[0]['source'].save(f"logs/{exp_name}/test_generation/{sample_index}_{file_name}_Real_{params_data['direction']['source']}.nii.gz")

#     if experiment['model_info']['modelType'] == 'patch2patch':
#         sampler, aggregator = dataModule.fetch_sampler_aggregator_sampler()
#         model = model_fn(experiment, sampler=sampler, aggregator= aggregator)
#     else:
#         model = model_fn(experiment)

#     # Trainer
    

#     trainer = Trainer(max_epochs=experiment['training']['epochs'], 
#                     callbacks=[save_checkpoint], 
#                     logger=loggers,
#                     accelerator="auto", 
#                     devices=experiment['device']['devices'],
#                     benchmark=experiment['device']['benchmark'],
#                     strategy=experiment['device']['strategy'],
#                     # num_sanity_val_steps=0
#                     #   profiler="simple",)
#             )

#     print("Trainer not runnung yet")
#     trainer.fit(model, datamodule=dataModule)
#     # print("Asynchronous apparently")


#     weights_path = f"logs/{exp_name}/checkpoints/{sample_index}"
#     weights = os.listdir(weights_path)
#     max_stratergy = 'ssim'

#     max_contender = weights[0]
#     print("Using arbirary strategy to select best model")
#     # for element in weights:
#     #     temp = element.split('-')
#     #     for edx, temp_element in enumerate(temp):
#     #         if max_stratergy in temp_element:
#     #             print('akr', float(temp_element.split('=')[1]))
#     #             print('akr',float(max_contender.split('-')[edx].split('=')[1]))
#     #             input()
#     #             if float(temp_element.split('=')[1]) > float(max_contender.split('-')[edx].split('=')[1]):
#     #                 max_contender = element

#     # print(f"Best model based on {max_stratergy} is: {max_contender}")

#     ckpt_path = os.path.join(weights_path, max_contender)
#     # model.load_from_checkpoint(ckpt_path)
#     ckpt = torch.load(ckpt_path, weights_only=False)
#     print("Loading model from checkpoint:", ckpt_path)
#     model.load_state_dict(ckpt["state_dict"])
#     print("Model loaded successfully from checkpoint.")
#     predictions = trainer.predict(model, dataloaders=dataModule)
#     print("Predictions made successfully.")
#     final_results.append(*predictions['metrics'])
#     output = predictions['model_prediction']
#     file_name = output["file_name"]
#     output['predicted'].save(f"logs/{exp_name}/test_generation/{sample_index}_{file_name}_Synthetic_{params_data['direction']['target']}.nii.gz")
#     output['target'].save(f"logs/{exp_name}/test_generation/{sample_index}_{file_name}_Real_{params_data['direction']['target']}.nii.gz")
#     output['source'].save(f"logs/{exp_name}/test_generation/{sample_index}_{file_name}_Real_{params_data['direction']['source']}.nii.gz")
    
    
    
    
#     if wand_boolean:
#         wandb_logger.experiment.finish()
    
#     del trainer
    
    
# with open(f"logs/{exp_name}/final_metrics.csv", mode="w", newline="") as f:
#     writer = csv.DictWriter(f, fieldnames=final_results[0].keys())  # keys become columns
#     writer.writeheader()  # writes: epoch,train_loss,val_loss
#     writer.writerows(final_results)  # writes each dictionary as a row

# import wandb
# run = wandb.init(project="MR2MR", name=f'{exp_name}_results')
# for element in final_results:
#     wandb.log({
#         "test_psnr": element['test_psnr'],
#         "test_ssim": element['test_ssim'],
#         "test_mae": element['test_mae'],
#         "test_mse": element['test_mse']
#     })
# num_samples = len(final_results)
# psnr, ssim, mae, mse = 0, 0, 0, 0
# for result in final_results:
#     psnr += result['test_psnr']
#     ssim += result['test_ssim']
#     mae += result['test_mae']
#     mse += result['test_mse']
# psnr /= num_samples
# ssim /= num_samples
# mae /= num_samples
# mse /= num_samples


# final_data = {
#     "average_test_psnr": psnr,
#     "average_test_ssim": ssim,
#     "average_test_mae": mae,
#     "average_test_mse": mse,
#     "epoch": experiment['training']['epochs'],
# }
# if wand_boolean:
#     wandb.log(final_data)
# print(f"Average PSNR: {psnr}, SSIM: {ssim}, MAE: {mae}, MSE: {mse}")
# # Save the final results
# with open(f"logs/{exp_name}/final_metrics_avg.csv", mode="w", newline="") as f:
#     writer = csv.DictWriter(f, fieldnames=final_data.keys())
#     writer.writeheader()
#     writer.writerow(final_data)