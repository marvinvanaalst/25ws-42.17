import numpy as np
# import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.modules.utils import _triple
# import copy
from torch.nn.parameter import Parameter
# import numbers
from einops import rearrange
from torch.nn import init
import os
# import util.util as util
# import ipdb

class Tanh01(nn.Module):
    def forward(self, x):
        return 0.5 * (torch.tanh(x) + 1)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Attention3d(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention3d, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for rotary embeddings"

        # scale factor for attention logits (standard ViT)
        self.scale = self.head_dim ** -0.5

        # learned per-head temperature (your original idea)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkv projection
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=1, bias=bias)

        # depthwise conv refinement over qkv (kept from your design)
        self.qkv_dwconv = nn.Conv3d(
            dim * 3, dim * 3,
            kernel_size=3, stride=1, padding=1,
            groups=dim * 3, bias=bias
        )

        # output projection
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)

        # gated v-flow: per-position, per-head gating over channels
        self.v_gate = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim, bias=True),
            nn.Sigmoid()
        )

    def _rotary_embedding(self, q, k, D, H, W):
        """
        Apply RoPE on flattened 3D positions.

        q, k: (B, heads, C, L) with C = head_dim, L = D*H*W
        """
        b, h, c, L = q.shape
        device = q.device
        dtype = q.dtype

        half_dim = c // 2  # we rotate pairs
        # inverse frequencies for RoPE (standard 1/10000^(2i/d))
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, 1, device=device, dtype=dtype) / half_dim))
        # position index over flattened volume
        pos = torch.arange(L, device=device, dtype=dtype)  # (L,)
        freqs = torch.einsum('i,j->ij', inv_freq, pos)     # (half_dim, L)

        sin = freqs.sin()[None, None, :, :]  # (1,1,half_dim,L)
        cos = freqs.cos()[None, None, :, :]  # (1,1,half_dim,L)

        # split channels into two halves for complex-like rotation
        q1, q2 = q[:, :, :half_dim, :], q[:, :, half_dim:2*half_dim, :]
        k1, k2 = k[:, :, :half_dim, :], k[:, :, half_dim:2*half_dim, :]

        q_rot_1 = q1 * cos - q2 * sin
        q_rot_2 = q1 * sin + q2 * cos
        k_rot_1 = k1 * cos - k2 * sin
        k_rot_2 = k1 * sin + k2 * cos

        q_rot = torch.cat([q_rot_1, q_rot_2], dim=2)
        k_rot = torch.cat([k_rot_1, k_rot_2], dim=2)

        return q_rot, k_rot

    def forward(self, x):
        b, c, D, H, W = x.shape

        # qkv projection + depthwise refinement
        qkv = self.qkv_dwconv(self.qkv(x))     # (B, 3C, D, H, W)
        q, k, v = qkv.chunk(3, dim=1)          # each (B, C, D, H, W)

        # reshape to (B, heads, head_dim, L) with L = D*H*W
        q = rearrange(q, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
        L = D * H * W

        # 1) Apply 3D rotary embeddings (on flattened spatial index)
        q, k = self._rotary_embedding(q, k, D, H, W)

        # 2) Normalize q, k after rotation (stabilizes attention)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 3) Gated v-flow: per-position, per-head gating over channels
        #    v: (B, heads, C, L) → (B, heads, L, C)
        v_perm = v.permute(0, 1, 3, 2)                     # (B, H, L, C)
        gate = self.v_gate(v_perm)                         # (B, H, L, C)
        gate = gate.permute(0, 1, 3, 2)                    # (B, H, C, L)
        v = v * gate                                       # gated value flow

        # 4) Attention
        # attn: (B, heads, L, L)
        attn = (q @ k.transpose(-2, -1)) * self.scale * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v                                     # (B, heads, C, L)

        # reshape back to (B, C, D, H, W)
        out = rearrange(out, 'b head c (d h w) -> b (head c) d h w',
                        head=self.num_heads, d=D, h=H, w=W)

        out = self.project_out(out)
        return out
    

class Attention_Block(nn.Module):
    def __init__(self,input_channel,output_channel,num_heads=8):
        super(Attention_Block,self).__init__()
        self.input_channel=input_channel
        self.output_channel=output_channel
        self.attention_s=Attention3d(dim=input_channel, num_heads=num_heads, bias=False)
        self.batch_norm_out = nn.BatchNorm3d(output_channel)
    def forward(self, inputs):

        attn_s=self.attention_s(inputs)

        inputs_attn=inputs+attn_s

        return  self.batch_norm_out(inputs_attn)

class Conv_FFN(nn.Module):
    def __init__(
        self,
        input_channel,
        middle_channel,
        output_channel,
        res=True,
        use_se=True,            # optional squeeze-and-excitation
        groups=1,               # grouped conv capability
        act_layer=nn.SiLU,      # better activation than LeakyReLU
        norm_layer=nn.BatchNorm3d
    ):
        super().__init__()

        self.res = res
        self.input_channel = input_channel
        self.output_channel = output_channel

        # Conv-Norm-Act (pre-activation style residual block)
        self.conv_1 = nn.Conv3d(input_channel, middle_channel, 3, padding=1, bias=False, groups=groups)
        self.norm_1 = norm_layer(middle_channel)
        self.act_1 = act_layer(inplace=True)

        # FFN channel expansion + projection
        self.conv_2 = nn.Conv3d(middle_channel, output_channel, 3, padding=1, bias=False, groups=groups)
        self.norm_2 = norm_layer(output_channel)
        self.act_2 = act_layer(inplace=True)

        # Optional SE attention block for channel reweighting
        if use_se:
            squeeze = max(1, output_channel // 8)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(output_channel, squeeze, 1),
                nn.SiLU(inplace=True),
                nn.Conv3d(squeeze, output_channel, 1),
                nn.Sigmoid()
            )
        else:
            self.se = None

        # Residual projection if needed
        if input_channel != output_channel:
            self.shortcut = nn.Sequential(
                nn.Conv3d(input_channel, output_channel, 1, bias=False),
                norm_layer(output_channel)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv_1(x)
        out = self.norm_1(out)
        out = self.act_1(out)

        out = self.conv_2(out)
        out = self.norm_2(out)

        if self.se is not None:
            out = out * self.se(out)

        out = self.act_2(out)

        if self.res:
            out = out + identity

        return out

class ESAU_Block(nn.Module):
    def __init__(self,in_channels,out_channels,num_heads=8,res=True):
        super(ESAU_Block,self).__init__()
        self.esaublock=nn.Sequential(
            Attention_Block(in_channels,in_channels,num_heads=num_heads),
            Conv_FFN(in_channels,in_channels,out_channels,res=res),
        )
    def forward(self,x):
        return self.esaublock(x)
      
class Down(nn.Module):

    def __init__(self, in_channels, out_channels, num_heads=8, res=True,
                 use_norm=True, norm_layer=nn.BatchNorm3d):
        super(Down, self).__init__()

        layers = [
            nn.MaxPool3d(kernel_size=2, stride=2)  # standard downsample
        ]

        if use_norm:
            # Norm before ESAU_block helps stabilize feature distribution
            layers.append(norm_layer(in_channels))

        # Leave ESAU_Block untouched internally
        layers.append(ESAU_Block(in_channels, out_channels, num_heads=num_heads, res=res))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

    
class LastDown(nn.Module):

    def __init__(self, in_channels, out_channels, num_heads=8, res=True,
                 use_norm=True, norm_layer=nn.BatchNorm3d):
        super(LastDown, self).__init__()

        layers = [
            nn.MaxPool3d(kernel_size=2, stride=2)  # stable downsampling
        ]

        # IMPORTANT: normalization after a pooling drop helps stabilize the distribution
        if use_norm:
            layers.append(norm_layer(in_channels))

        # Global/local attention refinement stage
        layers.append(Attention_Block(in_channels, in_channels, num_heads=num_heads))

        # FFN mixing + residual update
        layers.append(Conv_FFN(in_channels, 2 * in_channels, out_channels, res=res))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)



class Up(nn.Module):
    def __init__(self, in_channels, out_channels,res_unet=True, num_heads=8,res=True, interpolation_type='trilinear'):
        super(Up,self).__init__()
        self.res_unet=res_unet
        if interpolation_type == "trilinear" or interpolation_type == "nearest":
            self.up = nn.Upsample(scale_factor=(2,2,2), mode=interpolation_type, align_corners=None)
        elif interpolation_type == "conv" or interpolation_type == "Conv":
            self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            raise ValueError(interpolation_type)
        
        self.conv = ESAU_Block(in_channels, out_channels, num_heads=num_heads,res=res)

    def forward(self, x1, x2):

        x1 = self.up(x1)
        
        if self.res_unet:
            x=x1+x2
        else:
            x = torch.cat([x2, x1], dim=1)

        return self.conv(x)



class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels,decouple=None,bn=True,res=True,activation='tanh'):
        super(SingleConv,self).__init__()
        self.act=activation
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu = nn.SiLU(inplace=True)
        self.tanh = Tanh01()
        

    def forward(self, x):
        x=self.conv(x)
        if self.act=='Tanh01':
            x=self.tanh(x)
        elif self.act == 'SiLU':
            x=self.lrelu(x)
        return x
        


class ESAU_3D(nn.Module):
    def __init__(self,in_channels=1,out_channels=1,n_channels=64,num_heads=[1,2,4,8],res=True,activation=False,interpolation='trilinear'):
        super(ESAU_3D,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_channels = n_channels
        self.out_acttivation = activation
        self.up_scale_interpolation = interpolation
        
        self.firstconv=SingleConv(in_channels, n_channels//2,res=res,activation='SiLU')
        self.enc1 = ESAU_Block(n_channels//2, n_channels,num_heads=num_heads[0],res=res) 
        
        self.enc2 = Down(n_channels, 2 * n_channels,num_heads=num_heads[1],res=res)
        
        self.enc3 = Down(2 * n_channels, 4 * n_channels,num_heads=num_heads[2],res=res)
        
        self.enc4 = LastDown(4 * n_channels, 4 * n_channels,num_heads=num_heads[3],res=res)
        
        self.dec1 = Up(4 * n_channels, 2 * n_channels,num_heads=num_heads[2],res=res,interpolation_type='trilinear')
        
        self.dec2 = Up(2 * n_channels, 1 * n_channels,num_heads=num_heads[1],res=res, interpolation_type='trilinear')
        
        self.dec3 = Up(1 * n_channels, n_channels//2,num_heads=num_heads[0],res=res, interpolation_type='trilinear')

        self.out1 = SingleConv(n_channels//2,n_channels//2,res=res,activation=' ')
        
        self.out2 = SingleConv(n_channels//2,out_channels,res=res,activation='Tanh01')


    
    def forward(self, x):
        b, c, h, w, d = x.size()

        x =self.firstconv(x)
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        x4 = self.enc4(x3)

        output = self.dec1(x4, x3)
        output = self.dec2(output, x2)
        output = self.dec3(output, x1)
        output = self.out1(output)

        output = self.out2(output)

        return output
    

if __name__ == "__main__":
    firstU = ESAU_3D(in_channels=1,n_channels=32,out_channels=1)
    img_channels = 1
    img_size = 64
    x = torch.randn((4, img_channels, img_size, img_size, img_size))
    print(x.shape)
    output = firstU(x)
    print(output.shape)
    total_params = sum(p.numel() for p in firstU.parameters())
    trainable_params = sum(p.numel() for p in firstU.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    
