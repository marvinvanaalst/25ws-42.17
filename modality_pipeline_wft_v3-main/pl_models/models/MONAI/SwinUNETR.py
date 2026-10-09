import torch
# from monai.networks.nets import SwinUNETR
from pl_models.monai_edited.swin_unetr import SwinUNETR

# Lightweight configuration to stay under 10M parameters
# Standard ROI size is often (96, 96, 96) for MRI patches

def fetchSwinUNETR(
    # img_size=(96, 96, 96),
    in_channels=1,         # 1 for 3T MRI
    out_channels=1,        # 1 for 7T MRI synthesis (Regression)
    feature_size=12,       # Keep this low for parameter efficiency
    depths=(2, 2, 2, 2),   # Number of Swin blocks per stage
    num_heads=(3, 6, 12, 24),
    norm_name="instance",
    spatial_dims=3,
    interpolation=None):
    return SwinUNETR(
        # img_size=img_size,
        in_channels=in_channels,         # 1 for 3T MRI
        out_channels=out_channels,        # 1 for 7T MRI synthesis (Regression)
        feature_size=feature_size,       # Keep this low for parameter efficiency
        depths=depths,   # Number of Swin blocks per stage
        num_heads=num_heads,
        norm_name=norm_name,
        spatial_dims=spatial_dims,
        interpolation=interpolation #Need to copy from previous conda implemetation
        
    )
