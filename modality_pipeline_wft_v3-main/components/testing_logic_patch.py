from tqdm import tqdm
import torch
from augmentations.tta import *
from utils_custom.eval_metrics import VolumeIQAMetrics
from augmentations.pre_post_norm import normalize_per_sample_masked, denormalize
from utils_custom.csv_logger import CSVLogger
import os
import torchio as tio

class test_on_sample:
    def __init__(self, config):
        self.DEVICE = config['device']['devices'][0]
        self.tta_flip_enabled = config.get("tta_flips", {}).get("enable", False)
        self.pre_post_norm = config.get("pretrain_augments", {}).get("pre_post_norm", {}).get("apply", False)
        self.block_augments = config.get("pretrain_augments",{}).get("block_augments",{}).get("apply", False)
        self.block_config = config.get("pretrain_augments",{}).get("block_augments",{}).get("config", {})
        self.exp_name=f'{config["experiment_name"]}'


        
        
        
    def test_sample(self,model, sample_idx, epoch, test_dataloader,dir_name="training_eval"):
        self.csv_logger_testing = CSVLogger(log_dir=f'logs/{self.exp_name}/training_eval/', filename=f"{sample_idx}_metrics.csv",resume=True)
        model.eval()
        self.sample_index = sample_idx
        self.epoch = epoch
        iqa_metrics = VolumeIQAMetrics(data_range=1.0, device = self.DEVICE)
        pbar = tqdm(test_dataloader)
        os.makedirs(f'logs/{self.exp_name}/{dir_name}', exist_ok=True)
        os.makedirs(f'logs/{self.exp_name}/{dir_name}/{sample_idx}', exist_ok=True)
        base_dir = f'logs/{self.exp_name}/{dir_name}/{sample_idx}'
        # os.makedirs(f'logs/{self.exp_name}/training_eval/generations', exist_ok=True)
        
        with torch.no_grad():
            for idx, test_subjects in enumerate(pbar):
                        pbar.set_description(f"Evaluating {test_subjects['original'][0]['name']} for epoch {self.epoch}")
                        
                        with torch.no_grad():
                            for train_batch_idx, batch in enumerate(test_subjects['queue']):
                                with torch.autocast(device_type="cuda"):
                                    source = batch['source']['data'].to(self.DEVICE)
                                    mask = batch['mask']['data'].to(self.DEVICE)
                                    if self.pre_post_norm:
                                        normalize_per_sample_masked(source, mask, method=self.pre_post_norm_type)
                                    # if block_augments:
                                    #     source, aug_mask = augment(x = source, config=block_config, gin=gin_net)
                                    #     fake_output = model(source)*aug_mask
                                    # else:
                                    if self.tta_flip_enabled:
                                        fake_output = tta_forward(model, source * mask)
                                    else:    
                                        fake_output = model(source*mask)
                                test_subjects['aggregator'].add_batch((fake_output.float()+1)/2, batch['location'])
                            output = (test_subjects['aggregator'].get_output_tensor().to(self.DEVICE))*(test_subjects['original'][0]['mask']['data'].to(self.DEVICE))
                            output = torch.clip(output, min=0.0, max=1.0)
                            
                            # print(f"Output shape: {output.shape}")
                            # print(f"Original shape: {orig[0]['target']['data'].shape}")
                            # input()
                            if self.block_augments:
                                test_metrics = iqa_metrics.compute(output.unsqueeze(1).float(), test_subjects['original'][0]['target']['data'].to(self.DEVICE).unsqueeze(1),test_subjects['original'][0]['mask']['data'].to(self.DEVICE))
                            else:
                                test_metrics = iqa_metrics.compute(output.unsqueeze(1).float(), test_subjects['original'][0]['target']['data'].to(self.DEVICE).unsqueeze(1),test_subjects['original'][0]['mask']['data'].to(self.DEVICE))
                            
                            # affine = batch['source']['affine'][0]

                            affine = test_subjects['original'][0]['target']['affine']
                            
                            image = tio.ScalarImage(tensor=output.float().cpu(), affine=affine)
                            image.save(f"logs/{self.exp_name}/training_eval/{self.sample_index}_{self.epoch}_{test_subjects['original'][0]['name']}_fake.nii.gz")
                            test_subjects['original'][0]['target'].save(f"logs/{self.exp_name}/training_eval/{self.sample_index}_000_{test_subjects['original'][0]['name']}_orig.nii.gz")
                            
                            metrics = iqa_metrics.value()
                            iqa_metrics.reset()
                    
                        print(f"Validation Metrics for sample {epoch}: PSNR: {metrics['PSNR']}, SSIM_3D: {metrics['SSIM_3D']},  SSIM_2D: {metrics['SSIM_2D']}, MAE: {metrics['MAE']}, MSE: {metrics['MSE']}")

                        sample_test_metics = {
                            'epoch': self.epoch,
                            'sample_index': self.sample_index,
                            'test_psnr': metrics['PSNR'],
                            'test_ssim_3D': metrics['SSIM_3D'],
                            'test_ssim_2D': metrics['SSIM_2D'],
                            
                            'test_mae': metrics['MAE'],
                            'test_mse': metrics['MSE']
                        }
                        self.csv_logger_testing.log(sample_test_metics)
        model.train()   