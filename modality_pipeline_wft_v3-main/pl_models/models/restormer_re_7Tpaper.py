"""
Source: unofficial Restormer implementation (leftthomas/Restormer style).

Adapted for the modality pipeline:
  * `inp_channels` / `out_channels` are constructor arguments instead of a
    hardcoded 3, so the 1-channel MRI slice pipeline can use it.
  * the network is residual (`self.output(fr) + x`), which is unbounded. The
    slice pipeline trains and scores in [0, 1] (slicify min-max normalises both
    source and target), so the forward ends in a range clamp - matching the
    `nn.Hardtanh(0, 1)` tail of pl_models/models/restormer.py.
  * `channels` must double at every level (e.g. [16, 32, 64]); the decoder
    channel bookkeeping below assumes channels[i+1] == 2 * channels[i].
  * this is the **3-level** variant (two down/up samples). The decoder, the
    refinement stack and the output conv are indexed against a 3-entry
    `channels`, so len(channels) == 3 is enforced in the constructor.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))
        return x


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Restormer(nn.Module):
    def __init__(self, num_blocks=[1, 2, 2], num_heads=[1, 2, 4], channels=[48, 96, 192], num_refinement=4,
                 expansion_factor=2.66, inp_channels=3, out_channels=3, out_activation='hardtanh'):
        super(Restormer, self).__init__()

        # `zip` in the encoder list would silently truncate to the shortest of
        # the three, so a stale 4-entry num_blocks would drop a level unnoticed.
        if not (len(num_blocks) == len(num_heads) == len(channels) == 3):
            raise ValueError(
                f"this is the 3-level Restormer: num_blocks, num_heads and channels must all "
                f"have 3 entries (got {len(num_blocks)}, {len(num_heads)}, {len(channels)})."
            )
        for i in range(len(channels) - 1):
            if channels[i + 1] != 2 * channels[i]:
                raise ValueError(
                    f"channels must double at every level (got {channels}); the decoder "
                    f"concatenations below assume channels[i+1] == 2 * channels[i]."
                )
        if inp_channels != out_channels:
            # the forward ends in `self.output(fr) + x`, so the two must match
            raise ValueError(
                f"inp_channels ({inp_channels}) must equal out_channels ({out_channels}) "
                f"for the residual connection."
            )

        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.embed_conv = nn.Conv2d(inp_channels, channels[0], kernel_size=3, padding=1, bias=False)

        self.encoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(
            num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in
                                       zip(num_blocks, num_heads, channels)])
        # the number of down sample or up sample == the number of encoder - 1
        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])
        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])
        # the number of reduce block == the number of decoder - 1
        self.reduces = nn.ModuleList([nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
                                      for i in reversed(range(2, len(channels)))])
        # the number of decoder == the number of encoder - 1, so 3 levels get 2:
        # level 3 is the bottleneck and is encoder-only. (The 4-level original
        # opened this list with a channels[2]-wide stack; at 3 levels that block
        # has nothing to decode and would sit here unused.)
        self.decoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor)
                                                       for _ in range(num_blocks[1])])])
        # the channel of last one is not change
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                             for _ in range(num_blocks[0])]))

        self.refinement = nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                          for _ in range(num_refinement)])
        self.output = nn.Conv2d(channels[1], out_channels, kernel_size=3, padding=1, bias=False)

        # The pipeline normalises source and target to [0, 1] and scores with
        # data_range=1.0, so an unbounded residual output is out of domain.
        if out_activation == 'hardtanh':
            self.out_act = nn.Hardtanh(min_val=0.0, max_val=1.0)
        elif out_activation == 'sigmoid':
            self.out_act = nn.Sigmoid()
        elif out_activation == 'tanh':
            # pair with model_info.output_range_01 = false
            self.out_act = nn.Tanh()
        elif out_activation in (None, 'none'):
            self.out_act = nn.Identity()
        else:
            raise ValueError(f"Unknown out_activation={out_activation!r}; "
                             f"expected 'hardtanh', 'sigmoid', 'tanh' or 'none'.")

    def forward(self, x):
        fo = self.embed_conv(x)
        out_enc1 = self.encoders[0](fo)
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))  # bottleneck

        # ups[0] is UpSample(channels[2]), reduces[0] is channels[2] -> channels[1]
        out_dec2 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](out_enc3), out_enc2], dim=1)))
        # last level keeps channels[1]: the concat of ups[1] (-> channels[0]) and
        # out_enc1 (channels[0]) is already channels[1] wide, so no reduce.
        fd = self.decoders[1](torch.cat([self.ups[1](out_dec2), out_enc1], dim=1))
        fr = self.refinement(fd)
        out = self.output(fr) + x
        return self.out_act(out)
    
if __name__ == "__main__":
    a = torch.rand(8,3,256,256)
    b = Restormer()
    c = b(a)
    print(a.shape, c.shape)