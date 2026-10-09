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
from ptflops import get_model_complexity_info

class Tanh01(nn.Module):
    def forward(self, x):
        return 0.5 * (torch.tanh(x) + 1)

class LoRAConv3d(nn.Module):
    def __init__(self, dim, dim_out, kernel_size, bias, stride=1, padding=0, rank=2, alpha=1.0):        
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # LoRA decomposition (down -> up)
        in_c, out_c = dim, dim_out
        ksize = kernel_size

        # Down-project
        self.lora_A = nn.Conv3d(in_c, rank, kernel_size=ksize, 
                                stride=stride, padding=padding, bias=bias)

        # Up-project
        self.lora_B = nn.Conv3d(rank, out_c, kernel_size=1, bias=bias)


    def forward(self, x):
        return self.lora_B(self.lora_A(x))
    
class Attention3d(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention3d, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # self.qkv = nn.Conv3d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv = LoRAConv3d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = LoRAConv3d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, d, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (d h w) -> b (head c) d h w', head=self.num_heads, d=d, h=h, w=w)

        out = self.project_out(out)
        return out
    

class Attention_Block(nn.Module):
    def __init__(self,input_channel,output_channel,num_heads=8):
        super(Attention_Block,self).__init__()
        self.input_channel=input_channel
        self.output_channel=output_channel
        self.attention_s=Attention3d(dim=input_channel, num_heads=num_heads, bias=False)

    def forward(self, inputs):

        attn_s=self.attention_s(inputs)

        inputs_attn=inputs+attn_s

        return inputs_attn

class Conv_FFN(nn.Module):
    def __init__(self,input_channel,middle_channel,output_channel,res=True):
        super(Conv_FFN,self).__init__()
        self.input_channel=input_channel
        self.output_channel=output_channel
        self.conv_1 = LoRAConv3d(input_channel, middle_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_2 = LoRAConv3d(middle_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False)
        if self.input_channel != self.output_channel:
            self.shortcut = LoRAConv3d(input_channel,output_channel,kernel_size=1,padding=0,stride=1, bias=False)
            
        self.res=res
        self.act=nn.LeakyReLU(inplace=True)

    def forward(self, inputs):
        conv_S=self.act(self.conv_1(inputs))
        conv_S=self.act(self.conv_2(conv_S))

        if self.input_channel == self.output_channel:
            identity_out=inputs
        else:
            identity_out=self.shortcut(inputs)

        if self.res:
            output=conv_S+identity_out
        else:
            output=conv_S

        return output


class ESAU_Block(nn.Module):
    def __init__(self,in_channels,out_channels,num_heads=8,res=True):
        super(ESAU_Block,self).__init__()
        self.esaublock=nn.Sequential(
            Attention_Block(in_channels,in_channels,num_heads=num_heads),
            Conv_FFN(in_channels,in_channels,out_channels,res=res),
            # nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            
        )
    def forward(self,x):
        return self.esaublock(x)
      
               
class Down(nn.Module):

    def __init__(self, in_channels, out_channels,num_heads=8,res=True):
        super(Down,self).__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d((2,2,2), (2,2,2)),
            ESAU_Block(in_channels, out_channels, num_heads=num_heads, res=res),
            nn.BatchNorm3d(out_channels),
        )
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            
    def forward(self, x):
        return self.encoder(x)

    
class LastDown(nn.Module):

    def __init__(self, in_channels, out_channels,num_heads=8,res=True):
        super(LastDown,self).__init__()
        # num_groups = min(16, out_channels)

        self.encoder = nn.Sequential(
            nn.MaxPool3d((2,2,2)),
            Attention_Block(in_channels,in_channels,num_heads=num_heads),
            Conv_FFN(in_channels,2*in_channels,out_channels,res=res),
            # nn.BatchNorm3d(out_channels),

            )
    def forward(self, x):
        return self.encoder(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels,res_unet=True, num_heads=8,res=True, interpolation_type='trilinear'):
        super(Up,self).__init__()
        self.res_unet=res_unet
        if interpolation_type != "Conv":
            self.up = nn.Upsample(scale_factor=(2,2,2), mode=interpolation_type, align_corners=None)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        
        self.conv = ESAU_Block(in_channels, out_channels, num_heads=num_heads,res=res)

    def forward(self, x1, x2):

        x1 = self.up(x1)
        
        if self.res_unet:
            x=x1+x2
        else:
            x = torch.cat([x2, x1], dim=1)

        return self.conv(x)



class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels,decouple=None,bn=True,res=True,activation=False):
        super(SingleConv,self).__init__()
        self.act=activation
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.activation = nn.LeakyReLU(inplace=True)
        self.activation2 = Tanh01()
        

    def forward(self, x):
        x=self.conv(x)
        if self.act=='Tanh01':
            x=self.activation2(x)
        elif self.act == 'LeakyReLU':
            x=self.activation(x)
        return x
        


class ESAU_3D(nn.Module):
    def __init__(self,in_channels=1,out_channels=1,n_channels=64,num_heads=[1,2,4,8],res=True,activation=False,interpolation='trilinear'):
        super(ESAU_3D,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_channels = n_channels
        self.out_acttivation = activation
        self.up_scale_interpolation = interpolation
        
        self.firstconv=SingleConv(in_channels, n_channels//2,res=res,activation='Tanh01')
        self.enc1 = ESAU_Block(n_channels//2, n_channels,num_heads=num_heads[0],res=res) 
        
        self.enc2 = Down(n_channels, 2 * n_channels,num_heads=num_heads[1],res=res)
        
        self.enc3 = Down(2 * n_channels, 4 * n_channels,num_heads=num_heads[2],res=res)
        
        self.enc4 = LastDown(4 * n_channels, 4 * n_channels,num_heads=num_heads[3],res=res)
        
        self.dec1 = Up(4 * n_channels, 2 * n_channels,num_heads=num_heads[2],res=res,interpolation_type='trilinear')
        
        self.dec2 = Up(2 * n_channels, 1 * n_channels,num_heads=num_heads[1],res=res, interpolation_type='trilinear')
        
        self.dec3 = Up(1 * n_channels, n_channels//2,num_heads=num_heads[0],res=res, interpolation_type='trilinear')

        self.out1 = SingleConv(n_channels//2,n_channels//2,res=res,activation='LeakyReLU')
        
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
    
def count_true_flops(model, input_tensor):
    from torchtnt.utils.flops import FlopTensorDispatchMode
    with FlopTensorDispatchMode(model) as mode:
        _ = model(input_tensor)
    # return mode.get_total_flops(), mode.get_module_flops()
    print(f"Total FLOPs: {mode.get_total_flops()/1e6:.3f} MFLOPs")
    
if __name__ == "__main__":
    firstU = ESAU_3D(in_channels=1,n_channels=32,out_channels=1, interpolation='Conv')
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


    macs, params = get_model_complexity_info(firstU, (img_channels, img_size, img_size, img_size))
    print(macs, params)

    # x = torch.randn((1, img_channels, img_size, img_size, img_size))
    # count_true_flops(model=firstU, input_tensor=x)