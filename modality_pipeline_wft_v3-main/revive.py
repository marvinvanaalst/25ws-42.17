import torch

state = torch.load("/storage/an_inam/MR2MR/modality_pipeline_wft_v3/isbi_logs_C/30_ESAU_3D_L1_32_nearest/checkpoints/0/saveEvery_epoch_epoch=150.pt", map_location="cuda:3")

# if it's nested inside "state_dict"
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]

total_params = 0
for k, v in state.items():
    if hasattr(v, "numel"):
        total_params += v.numel()

print("Total parameters:", total_params)

for k, v in state.items():
    print(k, list(v.shape) if hasattr(v, "shape") else type(v))
    
from pl_models.models.ESAU_3D_transformer_like import ESAU_3D
model = ESAU_3D(
        in_channels = 1,
        out_channels=1,
        n_channels = 32,
        num_heads=[1,2,4,8],
        res = True,
        activation =True,
        interpolation = "nearest"
)

model.load_state_dict(state)