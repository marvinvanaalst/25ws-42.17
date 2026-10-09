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
class RMSNorm3D(nn.Module):
    def __init__(self, num_channels, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1, 1))

    def forward(self, x):
        # Normalize across channels
        rms = torch.sqrt((x * x).mean(dim=1, keepdim=True) + self.eps)
        return (x / rms) * self.weight
    
class Tanh01(nn.Module):
    def forward(self, x):
        return 0.5 * (torch.tanh(x) + 1)

class Attention3d(nn.Module):
    def __init__(self, dim, num_heads, bias, a_map_detach):
        super(Attention3d, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        assert dim%2==0, f"Dim(={dim}) should be divisible by two"
        dim_r = dim//2
        self.dim_r =dim_r
        self.qkv = nn.Conv3d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)
        
        self.qkv = nn.Conv3d(dim, dim_r*4, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim_r*4, dim_r*4, kernel_size=3, stride=1, padding=1, groups=dim_r*4, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)
        
        self.norm_complement_layer = None

        
        self.a_map_detach = a_map_detach
        # self.fuse_conv = nn.Conv3d(2*dim, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        b, c, d, h, w = x.shape
        qkv = self.qkv(x)
        qkv = self.qkv_dwconv(qkv)
        q, k, v, C_ = qkv.chunk(4, dim=1)

        q = rearrange(q, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)

        C_ = rearrange(C_, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)


        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        # print("attention_map_shape, ",attn.shape)
        if self.a_map_detach:
            compementary_attn_map = 1- attn.detach()
            
        else:
            compementary_attn_map = 1- attn
            
            
        out_2= compementary_attn_map@C_
        out_2 = rearrange(out_2, 'b head c (d h w) -> b (head c) d h w', head=self.num_heads, d=d, h=h, w=w)
        # out_2 =  (out_2 -  out_2.mean(dim=1, keepdim=True)) / torch.sqrt(out_2.var(dim=1, unbiased=False, keepdim=True))
        if self.norm_complement_layer == None:
            self.norm_complement_layer = RMSNorm3D(self.dim_r)
        out_2 = self.norm_complement_layer(out_2)
        # if self.norm_c ==None:    
        #     self.norm_c = nn.LayerNorm(out_2.shape)
        #     print(out_2.shape)
        # out_2 = self.norm_c(out_2)
        
        out = (attn @ v)
        out = rearrange(out, 'b head c (d h w) -> b (head c) d h w', head=self.num_heads, d=d, h=h, w=w)


        
        merged_out = torch.cat([out, out_2], dim=1)
        # merged_out = self.fuse_conv(merged_out)
        merged_out = self.project_out(merged_out)
        return merged_out

    

class Attention_Block(nn.Module):
    def __init__(self,input_channel,output_channel,a_map_detach, num_heads=8):
        super(Attention_Block,self).__init__()
        self.input_channel=input_channel
        self.output_channel=output_channel
        self.a_map_detach = a_map_detach
        self.attention_s=Attention3d(dim=input_channel, a_map_detach =self.a_map_detach, num_heads=num_heads, bias=False)
        self.batch_norm_out = nn.BatchNorm3d(output_channel)
    def forward(self, inputs):

        attn_s=self.attention_s(inputs)

        inputs_attn=inputs+attn_s

        return  self.batch_norm_out(inputs_attn)

class Conv_FFN(nn.Module):
    def __init__(self,input_channel,middle_channel,output_channel,res=True,):
        super(Conv_FFN,self).__init__()
        self.input_channel=input_channel
        self.output_channel=output_channel
        self.conv_1 = nn.Conv3d(input_channel, middle_channel, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn_middle =  nn.BatchNorm3d(middle_channel)
        self.conv_2 = nn.Conv3d(middle_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_out =  nn.BatchNorm3d(output_channel)
        
        if self.input_channel != self.output_channel:
            self.shortcut = nn.Conv3d(in_channels=input_channel,out_channels=output_channel,kernel_size=1,padding=0,stride=1,groups=1,bias=False)
            
        self.res=res
        self.act=nn.LeakyReLU(inplace=True)
    # def _initialize_weights(self):
        # GAussian--> no emperical evidence

    def forward(self, inputs):
        conv_S=self.act(self.conv_1(inputs))
        conv_S=self.act(self.bn_out(self.conv_2(conv_S)))

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
    def __init__(self,in_channels,out_channels,a_map_detach, num_heads=8,res=True):
        super(ESAU_Block,self).__init__()
        self.esaublock=nn.Sequential(
            Attention_Block(in_channels,in_channels,a_map_detach, num_heads=num_heads),
            Conv_FFN(in_channels,in_channels,out_channels,res=res),
        )
    def forward(self,x):
        return self.esaublock(x)
      
               
class Down(nn.Module):

    def __init__(self, in_channels, out_channels,a_map_detach = True, num_heads=8,res=True):
        super(Down,self).__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d((2,2,2), (2,2,2)),
            ESAU_Block(in_channels, out_channels, a_map_detach, num_heads=num_heads, res=res),
            # nn.BatchNorm3d(out_channels),
        )
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #             init.normal_(m.weight, mean=0.0, std=0.02)
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm3d):
    #             init.constant_(m.weight, 1)
    #             init.constant_(m.bias, 0)
            
    def forward(self, x):
        return self.encoder(x)

    
class LastDown(nn.Module):

    def __init__(self, in_channels, out_channels,a_map_detach, num_heads=8,res=True):
        super(LastDown,self).__init__()

        self.encoder = nn.Sequential(
            nn.MaxPool3d((2,2,2)),
            Attention_Block(in_channels,in_channels,a_map_detach, num_heads=num_heads),
            Conv_FFN(in_channels,2*in_channels,out_channels,res=res),
            )
    def forward(self, x):
        return self.encoder(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels,res_unet=True, num_heads=8,res=True, interpolation_type='trilinear',a_map_detach=True):
        super(Up,self).__init__()
        self.res_unet=res_unet
        self.aa_map_detach = a_map_detach
        if interpolation_type == "trilinear" or interpolation_type == "nearest":
            self.up = nn.Upsample(scale_factor=(2,2,2), mode=interpolation_type, align_corners=None)
        elif interpolation_type == "conv" or interpolation_type == "Conv":
            self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            raise ValueError(interpolation_type)
        
        self.conv = ESAU_Block(in_channels, out_channels, a_map_detach=self.aa_map_detach, num_heads=num_heads,res=res)

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
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.tanh = Tanh01()
        

    def forward(self, x):
        x=self.conv(x)
        if self.act=='Tanh01':
            x=self.tanh(x)
        elif self.act == 'LeakyReLU':
            x=self.lrelu(x)
        return x
        


class ESAU_3D(nn.Module):
    def __init__(self,in_channels=1,out_channels=1,n_channels=64,num_heads=[1,2,4,8],res=True,activation=False,interpolation='trilinear', a_map_detach=True):
        super(ESAU_3D,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_channels = n_channels
        self.out_acttivation = activation
        self.up_scale_interpolation = interpolation
        
        self.firstconv=SingleConv(in_channels, n_channels//2,res=res,activation='LeakyReLU')
        self.enc1 = ESAU_Block(n_channels//2, n_channels,a_map_detach, num_heads=num_heads[0],res=res, ) 
        
        self.enc2 = Down(n_channels, 2 * n_channels,a_map_detach, num_heads=num_heads[1],res=res)
        
        self.enc3 = Down(2 * n_channels, 4 * n_channels,a_map_detach, num_heads=num_heads[2],res=res)
        
        self.enc4 = LastDown(4 * n_channels, 4 * n_channels,a_map_detach, num_heads=num_heads[3],res=res)
        
        self.dec1 = Up(4 * n_channels, 2 * n_channels, num_heads=num_heads[2],res=res,interpolation_type='trilinear', a_map_detach=True)
        
        self.dec2 = Up(2 * n_channels, 1 * n_channels,num_heads=num_heads[1],res=res, interpolation_type='trilinear', a_map_detach=True)
        
        self.dec3 = Up(1 * n_channels, n_channels//2,num_heads=num_heads[0],res=res, interpolation_type='trilinear', a_map_detach=True)

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
    firstU = ESAU_3D(in_channels=1,n_channels=32,out_channels=1, a_map_detach=False)
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

    
