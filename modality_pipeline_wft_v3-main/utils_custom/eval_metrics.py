
# import torch
# from torchmetrics import MeanAbsoluteError, MeanSquaredError
# from torchmetrics.image import PeakSignalNoiseRatio
# from monai.metrics import SSIMMetric

# class VolumeIQAMetrics:
#     def __init__(self, data_range=1.0, device='cpu'):
#         """
#         data_range: Max intensity range of the input data. Use 1.0 if inputs are normalized.
#         """
#         self.metrics = {
#             'PSNR': 0,
#             'SSIM':0,
#             'MAE': 0,
#             'MSE':0,
#             'samples': 0
#         }
#         self.psnr = PeakSignalNoiseRatio(data_range=data_range).to(device)
#         self.mae = MeanAbsoluteError().to(device)
#         self.mse = MeanSquaredError().to(device)
#         self.ssim = SSIMMetric(
#             spatial_dims=3,
#             data_range=data_range,
#             kernel_type='gaussian',
#             win_size=11,
#         )

#     def compute(self, preds: torch.Tensor, targets: torch.Tensor) -> dict:
#         """
#         preds, targets: tensors of shape [B, C, H, W, D], normalized to [0, 1]
#         Returns a dictionary with PSNR, SSIM, MAE, MSE
#         """
#         # Ensure inputs are on same device and dtype
#         # preds = preds.to(dtype=torch.float32)
#         # targets = targets.to(dtype=torch.float32)

#         # Compute metrics
#         self.metrics['PSNR']+=self.psnr(preds, targets).item()
#         self.metrics['SSIM']+=self.ssim(preds, targets).mean().item()
#         self.metrics['MAE']+=self.mae(preds, targets).item()
#         self.metrics['MSE']+=self.mse(preds, targets).item()
#         self.metrics['samples'] += preds.shape[0]
#         return self.metrics['PSNR'], self.metrics['SSIM']
        
#     def value(self):
#         """
#         Returns the average values of the metrics computed so far.
#         """
#         if self.metrics['samples'] == 0:
#             raise ValueError("No samples processed. Call compute() before value().")
        
#         return {k: v / self.metrics['samples'] for k, v in self.metrics.items() if k != 'samples'}
    
#     def reset(self):
#         """
#         Resets the metrics to zero.
#         """
#         for key in self.metrics:
#                 self.metrics[key] = 0


import torch
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from torchmetrics.image import PeakSignalNoiseRatio
from monai.metrics import SSIMMetric


class VolumeIQAMetrics:
    """_summary_
    Made to porcess one volume at a time.
    
    
    """
    def __init__(self, data_range=1.0, device='cpu', skip_zero_masked=True):
        """
        data_range: Max intensity range of the input data. Use 1.0 if inputs are normalized.
        skip_zero_masked: If True, skip SSIM on patches with all-zero mask.
        """
        self.device = device
        self.skip_zero_masked = skip_zero_masked

        # Metric accumulators
        self.metrics = {
            'PSNR': 0.0,
            'SSIM_3D': 0.0,
            'SSIM_2D': 0.0,
            'MAE': 0.0,
            'MSE': 0.0,
            'MaskedSSIM': 0.0,
            'samples': 0,
            'masked_samples': 0,
        }

        # TorchMetrics & MONAI metrics
        self.psnr = PeakSignalNoiseRatio(data_range=data_range).to(device)
        self.mae = MeanAbsoluteError().to(device)
        self.mse = MeanSquaredError().to(device)
        self.ssim_3d = SSIMMetric(
            spatial_dims=3,
            data_range=data_range,
            kernel_type='gaussian',
            win_size=11,
        )
        self.ssim_2d = SSIMMetric(
            spatial_dims=2,        # <--- now it computes 2D SSIM
            data_range=1.0,        # set according to your image normalization
            kernel_type='gaussian',# Gaussian window
            win_size=11,           # 11x11 patch per pixel neighborhood
        )

    def _ensure_batch_dim(self, tensor: torch.Tensor):
        """
        Ensures input has shape [B, C, D, H, W]
        """
        if tensor.ndim == 3:       # [D, H, W]
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 4:     # [C, D, H, W]
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 5:     # [B, C, D, H, W] is expected
            raise ValueError(f"Expected 5D tensor, got {tensor.shape}")
        return tensor

    def compute(self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        """
        Compute and accumulate metrics for a batch or single volume.

        Args:
            preds: [B, C, D, H, W] or compatible (will be reshaped)
            targets: same shape as preds
            mask: optional [B, C, D, H, W] binary mask (1 for foreground, 0 for background)

        Returns:
            Tuple of (PSNR, SSIM) for immediate access.
        """
        preds = self._ensure_batch_dim(preds)
        targets = self._ensure_batch_dim(targets)

        if mask is not None:
            mask = self._ensure_batch_dim(mask)
            preds = preds*mask
            targets = targets*mask
        
        self.metrics['PSNR'] += self.psnr(preds, targets).item()
        self.metrics['MAE'] += self.mae(preds.reshape(-1), targets.reshape(-1)).item()
        self.metrics['MSE'] += self.mse(preds.reshape(-1), targets.reshape(-1)).item()
        self.metrics['SSIM_3D'] += self.ssim_3d(preds, targets).item()
        
        ssim_scores = []
        # iterate over slices along depth
        for d in range(preds.shape[4]): 
            pred_slice = preds[:, :, :, :, d]     # torch.Size([1, 1, 512, 512, 55])
            target_slice = targets[:, :, :, :, d]# (B, C, H, W)

            score = self.ssim_2d(pred_slice, target_slice)  # (B,)
            ssim_scores.append(score)

        # stack and average
        ssim_scores = torch.stack(ssim_scores, dim=0)  # (D, B)
        mean_ssim = ssim_scores.mean()                 # scalar
        self.metrics['SSIM_2D'] += mean_ssim.item()
        
        self.metrics['samples'] += preds.shape[0]

        # Compute masked SSIM if mask is given
        if mask is not None:
            for i in range(preds.shape[0]):
                p = preds[i:i+1]
                t = targets[i:i+1]
                m = mask[i:i+1]

                if self.skip_zero_masked and torch.all(m == 0):
                    continue  # skip background-only volume

                p_masked = p * m
                t_masked = t * m
                ssim_val = self.ssim_3d(p_masked, t_masked).item()
                self.metrics['MaskedSSIM'] += ssim_val
                self.metrics['masked_samples'] += 1

        return self.metrics['PSNR'], self.metrics['SSIM_3D']

    def value(self):
        """
        Return the average of all computed metrics.
        """
        if self.metrics['samples'] == 0:
            raise ValueError("No samples processed. Call compute() before value().")

        result = {
            'PSNR': self.metrics['PSNR'] / self.metrics['samples'],
            'SSIM_3D': self.metrics['SSIM_3D'] / self.metrics['samples'],
            'SSIM_2D' :  self.metrics['SSIM_2D'] / self.metrics['samples'],
            'MAE': self.metrics['MAE'] / self.metrics['samples'],
            'MSE': self.metrics['MSE'] / self.metrics['samples'],
        }

        if self.metrics['masked_samples'] > 0:
            result['MaskedSSIM'] = self.metrics['MaskedSSIM'] / self.metrics['masked_samples']
        else:
            print("No Masked SSIM")
            result['MaskedSSIM'] = None  # or 0.0 if preferred

        return result

    def reset(self):
        """
        Resets all metrics to zero.
        """
        for key in self.metrics:
            self.metrics[key] = 0.0
