import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """
    3D Conv block: Conv3d -> (Norm) -> LeakyReLU
    First block usually without normalization.
    """
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels,
                      kernel_size=4, stride=2, padding=1, bias=not normalize)
        ]
        if normalize:
            layers.append(nn.BatchNorm3d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Pix2PixDiscriminator3D(nn.Module):
    """
    Conditional PatchGAN Discriminator for 3D volumes.

    Input: (img, cond), each (B, C, D, H, W)
    Output: patch-level logits (B, 1, D', H', W')
    """
    def __init__(self,
                 in_channels_img: int = 1,
                 in_channels_cond: int = 1,
                 base_channels: int = 64,
                 num_layers: int = 3):
        super().__init__()

        # Total input channels = condition + target
        in_channels_total = in_channels_img + in_channels_cond

        layers = [
            ConvBlock3D(in_channels_total, base_channels, normalize=False)
        ]

        # Next PatchGAN layers: C128, C256, C512...
        curr = base_channels
        for i in range(1, num_layers + 1):
            out = min(base_channels * 2**i, 512)
            layers.append(ConvBlock3D(curr, out, normalize=True))
            curr = out

        self.feature_extractor = nn.Sequential(*layers)

        # Final conv (stride 1) → keeps spatial resolution for patch output
        self.final_conv = nn.Conv3d(curr, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, img, cond):
        x = torch.cat([img, cond], dim=1)   # concat along channels
        x = self.feature_extractor(x)
        return self.final_conv(x)
