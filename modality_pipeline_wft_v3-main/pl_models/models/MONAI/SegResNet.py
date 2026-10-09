import torch
from monai.networks.nets import SegResNet

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fetch_SegResNet(
    spatial_dims=3,
    init_filters=32,      
    in_channels=1,         
    out_channels=1,        # 1 for 7T MRI synthesis (Regression)
    dropout_prob=0.2,
    

    blocks_down=[1, 2, 2, 4], 
    blocks_up=[1, 1, 1],
    

    act="RELU",            
    norm="GROUP",        
    upsample_mode="deconv" 
):
    return SegResNet(
        spatial_dims,
        init_filters,       
        in_channels,         
        out_channels,        
        dropout_prob,

        blocks_down, 
        blocks_up,
        

        act,            
        norm,         
        upsample_mode
        )