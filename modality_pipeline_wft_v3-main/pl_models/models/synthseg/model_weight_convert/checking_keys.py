import torch
from box import Box
from pprint import pprint

# Tried to read keys form the weights given in 7TAcquisitions paper. It only contains the model weights, and no weights for discriminator or perceptual loss model.
path = '/Drive4T/inam/3T_7T_Synthesis/pl_models/models/synthseg/total_weights.ckpt'
poses = torch.load(path, weights_only=False)

for keys, ww in poses.items():
    print(keys)

print("\n\n\n\n\n\n")

# exit()
pprint(poses['callbacks'])
# for keys, ww in poses['pytorch-lightning_version']:
