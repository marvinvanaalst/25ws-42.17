import torch
import torch.nn as nn
from pl_models.models.synthseg.SynthSeg_loss import LSynthSeg
from pl_models.models.synthseg.SynthSeg_parts import SynthSegModel
from utils_custom.wavelet_loss import WaveletL1Loss, WaveletL1Loss2D
import lpips
from utils_custom.peeyush_loss import CombinedLoss
# from utils.losses_2 import laplacian_loss
from monai.losses import DiffusionLoss, BendingEnergyLoss
from pl_models.impact import IMPACTReg as IMPACT
from utils_custom.monai_losses import MRIPerceptualLoss
# diffusion = DiffusionLoss(normalize=True)     # 1st order, L2, single field
# bending   = BendingEnergyLoss(normalize=True) #

# L2 total variation: penalises squared first derivatives of a field
# tv = DiffusionLoss(normalize=True)
# loss = tv(x_hat)

def compile_loss_fn(loss_config):
    # Map of all supported losses
    active_losses = loss_config['loss']['types']
    
    losses_dict = {}
    for name, weight, *loss_contructor in active_losses:
        if name not in loss_config['loss']['available']:
            raise ValueError(f"Unknown loss: {name}")
        else:
            if name == "L1":
                losses_dict[name] = (nn.L1Loss().to(loss_config['device']['devices'][0]), weight)
            elif name == "L2":
                losses_dict[name] = (nn.MSELoss().to(loss_config['device']['devices'][0]), weight)
            elif name == "SynthSeg":
                # Placeholder for SynthSeg loss, replace with actual implementation
                SynthSeg_model = SynthSegModel().to(loss_config['device']['devices'][0])
                SynthSeg_model.load_state_dict(torch.load(loss_config['loss']['synth_seg_path']))
                SynthSeg_loss =  LSynthSeg(4, 6, SynthSeg_model, device=loss_config['device']['devices'][0])
                losses_dict[name] = (SynthSeg_loss, weight)
            elif name == "lpips_alex_2d_lpips_true":
                lpips_loss_fn = lpips.LPIPS(net='vgg', lpips=True).to(loss_config['device']['devices'][0])
                losses_dict[name] = (lpips_loss_fn, weight)
            elif name == "lpips_alex_2d_lpips_false":
                lpips_loss_fn = lpips.LPIPS(net='vgg').to(loss_config['device']['devices'][0])
                losses_dict[name] = (lpips_loss_fn, weight)
            # elif name == "3d_laplassian":
            #     laplacian_loss_fn = laplacian_loss(loss_config['device']['devices'][0])
            #     losses_dict[name] = (laplacian_loss_fn, weight)
            elif name == 'wavelet3D':
                wavelet_loss_fn = WaveletL1Loss(**loss_contructor[0]).to(loss_config['device']['devices'][0])
                losses_dict[name] = (wavelet_loss_fn,weight)
            elif name == 'wavelet2D':
                wavelet_loss_fn = WaveletL1Loss2D(**loss_contructor[0]).to(loss_config['device']['devices'][0])
                losses_dict[name] = (wavelet_loss_fn,weight)
            elif name == "DiffusionLoss":
                lapplacian_loss_fn = DiffusionLoss(normalize=True).to(loss_config['device']['devices'][0])
                losses_dict[name] = (lapplacian_loss_fn, weight)
            elif name == "Resnet50_medicalNet_Perceptual":
                medical_perceptual_loss = MRIPerceptualLoss().to(loss_config['device']['devices'][0])
                losses_dict[name] = (medical_perceptual_loss, weight)
                

            # elif name == "peeyush_loss":
            #     p_loss_fn = CombinedLoss(l1_weight=1.0, perceptual_weight=0.1, device=loss_config['device']['devices'][0])
            #     p_loss_fn.eval()
            #     losses_dict[name] = (p_loss_fn, weight)
                
                
            # elif name == "mssim":
            #     mssim_loss_fn = pytorch_msssim.MS_SSIM(data_range=1.0, size_average=True, channel=1)
            #     losses_dict[name] = (mssim_loss_fn, weight)

    print(f"[Loss] Active losses: {list(losses_dict.keys())}")
    print(f"[Loss] Weights: {[w for _, w in losses_dict.values()]}")

    # Build a static switch-style loss function
    def total_loss(pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        # This is like a static switch: only includes selected blocks
        if "L1" in losses_dict:
            fn, w = losses_dict["L1"]
            loss += w * fn(pred, target)
        if "L2" in losses_dict:
            fn, w = losses_dict["L2"]
            loss += w * fn(pred, target)
        if "SynthSeg" in losses_dict:
            fn, w = losses_dict["SynthSeg"]
            loss += w * fn(pred, target)
        if "lpips_alex_2d_lpips_true" in losses_dict:
            fn, w = losses_dict["lpips_alex_2d_lpips_true"]
            # LPIPS expects images in [-1, 1]
            pred = pred * 2 - 1
            target = target * 2 - 1
            loss += w * fn(pred.repeat(1,3,1,1), target.repeat(1,3,1,1)).mean()
        if "lpips_alex_2d_lpips_false" in losses_dict:
            fn, w, wavelet_composition = losses_dict["lpips_alex_2d_lpips_false"]
            # LPIPS expects images in [-1, 1]
            pred = pred * 2 - 1
            target = target * 2 - 1
            loss += w * fn(pred.repeat(1,3,1,1), target.repeat(1,3,1,1)).mean()
        if "wavelet3D" in losses_dict:
            fn, w, *kwargs = losses_dict['wavelet3D']
            loss += w* fn(pred, target)
        if "wavelet2D" in losses_dict:
            fn, w, *kwargs = losses_dict['wavelet2D']
            loss += w* fn(pred, target)
        if "DiffusionLoss" in losses_dict:
            fn, w = losses_dict['DiffusionLoss']
            loss += w* fn((target - pred).expand(-1,3,-1,-1,-1))

        return loss

    return total_loss
