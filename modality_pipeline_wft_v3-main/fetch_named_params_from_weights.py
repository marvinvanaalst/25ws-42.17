import torch


# path = "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/ISBI_Paper_1_logs/2026_01_18_SwinUNETR_Max_Conv/checkpoints/0/saveEvery_epoch_epoch=300.pt"
path = "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/ISBI_Paper_1_logs/2026_01_18_SwinUNETR_Max_Nearest/checkpoints/0/saveEvery_epoch_epoch=300.pt"
path = '/storage/an_inam/MR2MR/modality_pipeline_wft_v3/ISBI_Paper_1_logs/2026_01_18_SwinUNETR_Max_trilinear/checkpoints/0/saveEvery_epoch_epoch=300.pt'
print(path)
model = torch.load(path, map_location="cpu")

# for name, module in model.named_modules():
#     print(name, "->", module)
    
for key in model.keys():
    print(key)
    # print(key.shape)