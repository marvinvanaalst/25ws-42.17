import torch
import torch.nn as nn
import sys
import os
from pl_models.models.VNET.VNet_parts import InputTransition, DownTransition, UpTransition


class VNet(nn.Module):
    def __init__(
            self, place_holer=None
    ):
        super().__init__()
        self.in_tr = InputTransition(16)
        self.down_tr32 = DownTransition(16, 2)
        self.down_tr64 = DownTransition(32, 3)
        self.down_tr128 = DownTransition(64, 3)
        self.down_tr256 = DownTransition(128, 3)
        self.up_tr256 = UpTransition(256, 256, 3)
        self.up_tr128 = UpTransition(256, 128, 3)
        self.up_tr64 = UpTransition(128, 64, 2)
        self.up_tr32 = UpTransition(64, 32, 1)
        self.out_tr = nn.Conv3d(32, 1, kernel_size=1)     

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        x = self.up_tr256(out256, out128)
        x = self.up_tr128(x, out64)
        x = self.up_tr64(x, out32)
        x = self.up_tr32(x, out16)
        x = self.out_tr(x)
        return x

if __name__ == "__main__":
    model = VNet()
    x = torch.randn(1,1, 64, 64, 64)  # Example input tensor
    output = model(x)
    print(output.shape)  # Should be (1, 1, 96, 96, 96) if input is (1, 1, 96, 96, 96)
    
    img_channels = 1
    img_size = 64
    x = torch.randn((4, img_channels, img_size, img_size, img_size))
    print(x.shape)
    output = model(x)
    print(output.shape)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
