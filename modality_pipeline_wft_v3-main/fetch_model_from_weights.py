
import torch
from pl_models.models.ESAU_3D_transformer_like import ESAU_3D

from pl_models.monai_edited.swin_unetr import SwinUNETR
from pl_models.monai_edited.unetr import UNETR

from pl_models.models.VNET.VNet import VNet
from pl_models.models.MONAI.SegResNet import SegResNet
from pl_models.models.UNET_3D.A3dUNET import ResidualUNetSE3D
from pl_models.models.UNET_3D.A3dUNET import UNet3D

import json
path = "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/ISBI_Paper_1_logs/2026_01_20_UNETR_32_trilinear/config.json"
# path = '/storage/an_inam/MR2MR/modality_pipeline_wft_v3/ISBI_Paper_1_logs/2026_01_18_SwinUNETR_Max_Conv/config.json'
with open(path, "r") as f:
    data = json.load(f)
model_obj = data['model']
print(model_obj)
model = UNETR(**model_obj).to('cuda:0')
a = torch.randn(8,1,64,64,64).to('cuda:0')
op = model(a)
print(op.shape)


weight = "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/ISBI_Paper_1_logs/2026_01_20_UNETR_32_trilinear/checkpoints/1/saveEvery_epoch_epoch=300.pt"
state = torch.load(weight, map_location="cpu")
for k, v in state.items():
    print(k, list(v.shape) if hasattr(v, "shape") else type(v))


for name, p in model.named_parameters():
    print(name, p.numel())
    
model.load_state_dict(state, strict=True)  # 2. load parameters
# model.eval()    

total_params = sum(p.numel() for p in model.parameters())
print(total_params)