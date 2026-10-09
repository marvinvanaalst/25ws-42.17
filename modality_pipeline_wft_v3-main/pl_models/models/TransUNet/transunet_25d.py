"""
TransUNet25D — TransUNet (Chen et al. 2021) adapted for 5-channel 2.5D MRI synthesis.

R50-ViT-B_16 backbone:
  - ResNet-50 V2 encoder (bottleneck, 3 stages, no Layer4) -> CNN skip connections
  - ViT-B/16 transformer encoder (12 layers, 768-dim, 12-head) on 14x14 feature map
  - Cascaded Upsampler (CUP) decoder with 3 skip connections
  - Conv2d(->1) + Tanh output head, range [-1, 1]

Input adaptation:
  - conv1: 3->5 channels. Channels 0-2 = pretrained. Channels 3-4 = mean(pretrained).
  - Center-crops 256x256 input to 224x224; pads output back to 256x256.
  - Crop safety: verified 35px margin to brain tissue on all sides (empirically).

Usage:
    from pl_models.models.TransUNet.transunet_25d import TransUNet25D
    model = TransUNet25D(n_channels=5, img_size=224,
                         pretrained_path='model_weights/R50+ViT-B_16.npz')
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ViT configuration (fixed: ViT-B/16)
# ---------------------------------------------------------------------------
VIT_HIDDEN_SIZE = 768
VIT_MLP_DIM     = 3072
VIT_NUM_HEADS   = 12
VIT_NUM_LAYERS  = 12
VIT_DROPOUT     = 0.1
RESNET_GRID     = 14    # feature map side at 1/16 of 224
NUM_PATCHES     = RESNET_GRID * RESNET_GRID   # 196


# ===========================================================================
# Building blocks
# ===========================================================================

class StdConv2d(nn.Conv2d):
    """Weight-standardised convolution (ResNet V2 style)."""
    def forward(self, x):
        w = self.weight
        v = w.var(dim=[1, 2, 3], keepdim=True, unbiased=False)
        m = w.mean(dim=[1, 2, 3], keepdim=True)
        w = (w - m) / (v + 1e-10).sqrt()
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def _conv3x3(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias)


def _conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation bottleneck block (ResNet V2)."""
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1   = nn.GroupNorm(32, cin,  eps=1e-6)
        self.conv1 = _conv1x1(cin, cmid)
        self.gn2   = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = _conv3x3(cmid, cmid, stride=stride)
        self.gn3   = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv3 = _conv1x1(cmid, cout)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = _conv1x1(cin, cout, stride=stride) if (stride != 1 or cin != cout) else None

    def forward(self, x):
        residual = x
        y = self.relu(self.gn1(x))
        if self.downsample is not None:
            residual = self.downsample(y)
        y = self.conv1(y)
        y = self.conv2(self.relu(self.gn2(y)))
        y = self.conv3(self.relu(self.gn3(y)))
        return y + residual


class ResNetV2(nn.Module):
    """
    ResNet-50 V2 backbone — 3 stages only (no layer4).
    Returns (skip1 at 1/4, skip2 at 1/8, skip3 at 1/16).
    """
    def __init__(self, n_channels: int = 5):
        super().__init__()
        # 7x7 conv stem (stride=2): input -> 112x112
        self.conv_root = nn.Conv2d(n_channels, 64, kernel_size=7,
                                   stride=2, padding=3, bias=False)
        self.gn_root   = nn.GroupNorm(32, 64, eps=1e-6)
        self.relu      = nn.ReLU(inplace=True)
        self.pool      = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 1: 64->256ch, stride=1 -> 56x56
        self.layer1 = nn.Sequential(
            PreActBottleneck( 64, cout=256, stride=1),
            PreActBottleneck(256, cout=256, stride=1),
            PreActBottleneck(256, cout=256, stride=1),
        )
        # Stage 2: 256->512ch, stride=2 -> 28x28
        self.layer2 = nn.Sequential(
            PreActBottleneck(256, cout=512, stride=2),
            PreActBottleneck(512, cout=512, stride=1),
            PreActBottleneck(512, cout=512, stride=1),
            PreActBottleneck(512, cout=512, stride=1),
        )
        # Stage 3: 512->1024ch, stride=2 -> 14x14 (feeds ViT)
        self.layer3 = nn.Sequential(*[
            PreActBottleneck(512 if i == 0 else 1024,
                             cout=1024, stride=(2 if i == 0 else 1))
            for i in range(9)
        ])

    def forward(self, x):
        x = self.relu(self.gn_root(self.conv_root(x)))  # [B, 64, 112, 112]
        skip0 = x                                       # [B, 64, 112, 112]
        x = self.pool(x)                                # [B, 64,  56,  56]
        skip1 = self.layer1(x)                          # [B, 256, 56,  56]
        skip2 = self.layer2(skip1)                      # [B, 512, 28,  28]
        feat14 = self.layer3(skip2)                     # [B, 1024, 14, 14]
        return feat14, [skip2, skip1, skip0]


# ===========================================================================
# ViT-B/16 Encoder
# ===========================================================================

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads  = VIT_NUM_HEADS
        self.head_size  = VIT_HIDDEN_SIZE // VIT_NUM_HEADS
        self.scale      = self.head_size ** -0.5
        self.query      = nn.Linear(VIT_HIDDEN_SIZE, VIT_HIDDEN_SIZE)
        self.key        = nn.Linear(VIT_HIDDEN_SIZE, VIT_HIDDEN_SIZE)
        self.value      = nn.Linear(VIT_HIDDEN_SIZE, VIT_HIDDEN_SIZE)
        self.out        = nn.Linear(VIT_HIDDEN_SIZE, VIT_HIDDEN_SIZE)
        self.attn_drop  = nn.Dropout(VIT_DROPOUT)
        self.proj_drop  = nn.Dropout(VIT_DROPOUT)

    def forward(self, x):
        B, N, C = x.shape
        H, D = self.num_heads, self.head_size
        q = self.query(x).reshape(B, N, H, D).permute(0, 2, 1, 3)
        k = self.key(x).reshape(B, N, H, D).permute(0, 2, 1, 3)
        v = self.value(x).reshape(B, N, H, D).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        return self.proj_drop(self.out(x))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(VIT_HIDDEN_SIZE, VIT_MLP_DIM)
        self.fc2  = nn.Linear(VIT_MLP_DIM, VIT_HIDDEN_SIZE)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(VIT_DROPOUT)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(VIT_HIDDEN_SIZE, eps=1e-6)
        self.attn = Attention()
        self.ln2  = nn.LayerNorm(VIT_HIDDEN_SIZE, eps=1e-6)
        self.mlp  = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ViTEncoder(nn.Module):
    """
    ViT-B encoder on top of the 14x14 ResNet feature map.
    Patch embedding: 1x1 conv (1024->768) + flatten.
    """
    def __init__(self, grid_size: int = RESNET_GRID):
        super().__init__()
        self.num_patches = grid_size * grid_size   # 196
        self.patch_embed = nn.Conv2d(1024, VIT_HIDDEN_SIZE, kernel_size=1)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, VIT_HIDDEN_SIZE))
        self.pos_embed   = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, VIT_HIDDEN_SIZE))
        self.emb_drop    = nn.Dropout(VIT_DROPOUT)
        self.blocks      = nn.Sequential(*[TransformerBlock() for _ in range(VIT_NUM_LAYERS)])
        self.ln          = nn.LayerNorm(VIT_HIDDEN_SIZE, eps=1e-6)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, feature_map):
        """feature_map: [B, 1024, 14, 14] -> [B, 196, 768]"""
        B = feature_map.shape[0]
        x = self.patch_embed(feature_map)           # [B, 768, 14, 14]
        x = x.flatten(2).transpose(1, 2)            # [B, 196, 768]
        cls = self.cls_token.expand(B, -1, -1)      # [B, 1, 768]
        x = self.emb_drop(torch.cat([cls, x], dim=1) + self.pos_embed)  # [B, 197, 768]
        x = self.blocks(x)
        x = self.ln(x)
        return x[:, 1:]    # drop CLS -> [B, 196, 768]


# ===========================================================================
# CUP Decoder (Cascaded Upsampler with skip connections)
# ===========================================================================

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class CUPDecoder(nn.Module):
    """
    Transformer out [B,768,14,14] -> [B,16,224,224] via 4 upsample stages.
    Skips:
      dec0: 14->28,   in=512, skip=512 (skip2, 28x28),   out=256
      dec1: 28->56,   in=256, skip=256 (skip1, 56x56),   out=128
      dec2: 56->112,  in=128, skip=64  (skip0, 112x112), out=64
      dec3: 112->224, in=64,  skip=0,                    out=16
    """
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(VIT_HIDDEN_SIZE, 512, kernel_size=3, padding=1, bias=False)
        self.dec0 = DecoderBlock(512, 512, 256)   # 14->28,  +skip2 (512ch)
        self.dec1 = DecoderBlock(256, 256, 128)   # 28->56,  +skip1 (256ch)
        self.dec2 = DecoderBlock(128,  64,  64)   # 56->112, +skip0 (64ch)
        self.dec3 = DecoderBlock( 64,   0,  16)   # 112->224, no skip

    def forward(self, tokens, features):
        B = tokens.shape[0]
        x = tokens.transpose(1, 2).reshape(B, VIT_HIDDEN_SIZE, RESNET_GRID, RESNET_GRID)
        x = self.proj(x)
        skip2, skip1, skip0 = features
        x = self.dec0(x, skip2)   # [B, 256, 28, 28]
        x = self.dec1(x, skip1)   # [B, 128, 56, 56]
        x = self.dec2(x, skip0)   # [B, 64, 112, 112]
        x = self.dec3(x, None)    # [B, 16, 224, 224]
        return x   # [B, 16, 224, 224]


# ===========================================================================
# Main model
# ===========================================================================

class TransUNet25D(nn.Module):
    """
    TransUNet adapted for 5-channel 2.5D MRI synthesis.

    Config params (passed from JSON via model_fetcher):
        n_channels     : int  = 5
        img_size       : int  = 224
        pretrained_path: str  = 'model_weights/R50+ViT-B_16.npz'  (or None)
    """
    CROP_SAFE_MARGIN_PX = 35  # verified margin between crop edge and brain

    def __init__(self, n_channels: int = 5, img_size: int = 224,
                 pretrained_path: str = None, **kwargs):
        super().__init__()
        self.n_channels = n_channels
        self.img_size   = img_size

        self.encoder = ResNetV2(n_channels=n_channels)
        self.vit      = ViTEncoder(grid_size=img_size // 16)
        self.decoder  = CUPDecoder()
        self.head     = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Tanh()   # output in [-1, 1]; set output_range_01=false in config
        )

        self._init_decoder()

        if pretrained_path:
            self.load_pretrained(pretrained_path)
        else:
            print("[TransUNet25D] No pretrained_path — training from scratch.")

    def _init_decoder(self):
        for m in list(self.decoder.modules()) + list(self.head.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def _center_crop(self, x):
        h, w = x.shape[-2], x.shape[-1]
        t = (h - self.img_size) // 2
        l = (w - self.img_size) // 2
        return x[..., t: t + self.img_size, l: l + self.img_size]

    def _pad_back(self, x, orig_h, orig_w):
        if x.shape[-2] == orig_h and x.shape[-1] == orig_w:
            return x
        pt = (orig_h - self.img_size) // 2
        pl = (orig_w - self.img_size) // 2
        pb = orig_h - self.img_size - pt
        pr = orig_w - self.img_size - pl
        return F.pad(x, (pl, pr, pt, pb), value=0.0)

    def forward(self, x):
        """x: [B, 5, H, W] -> [B, 1, H, W]  (H,W=256 from pipeline, crops/pads internally)"""
        orig_h, orig_w = x.shape[-2], x.shape[-1]
        if orig_h != self.img_size or orig_w != self.img_size:
            x = self._center_crop(x)                  # [B,5,224,224]
        feat14, features = self.encoder(x)            # CNN features
        tokens = self.vit(feat14)                     # [B,196,768]
        feat   = self.decoder(tokens, features)       # [B,16,224,224]
        out    = self.head(feat)                       # [B,1,224,224]
        if orig_h != self.img_size or orig_w != self.img_size:
            out = self._pad_back(out, orig_h, orig_w) # [B,1,256,256]
        return out

    # -----------------------------------------------------------------------
    # Pretrained weight loading from R50+ViT-B_16.npz
    # -----------------------------------------------------------------------
    def load_pretrained(self, npz_path: str):
        import os
        if not os.path.exists(npz_path):
            print(f"[TransUNet25D] WARNING: checkpoint not found: {npz_path}")
            print("  -> Continuing with random init.")
            return

        print(f"[TransUNet25D] Loading pretrained weights: {npz_path}")
        weights = np.load(npz_path)
        print(f"  Checkpoint keys ({len(weights.files)}): "
              f"{', '.join(list(weights.files)[:8])} ...")

        msd     = self.state_dict()
        new_sd  = {k: v.clone() for k, v in msd.items()}
        loaded  = skipped = 0

        def _pt(arr, ndim_expect=None):
            """Numpy array -> float32 tensor, transposed if needed."""
            if arr.ndim == 4:          # conv: (H,W,Cin,Cout)->(Cout,Cin,H,W)
                arr = np.transpose(arr, (3, 2, 0, 1))
            elif arr.ndim == 2:        # dense: (Cin,Cout)->(Cout,Cin)
                arr = arr.T
            return torch.from_numpy(arr.copy()).float()

        def _copy(tf_key, pt_key):
            nonlocal loaded, skipped
            if tf_key not in weights or pt_key not in new_sd:
                return
            t = _pt(weights[tf_key])
            if new_sd[pt_key].shape == t.shape:
                new_sd[pt_key] = t; loaded += 1
            else:
                skipped += 1

        # ---------- ResNet V2 backbone ----------
        # GroupNorm root
        _copy('resnet50/group_norm/gamma', 'encoder.gn_root.weight')
        _copy('resnet50/group_norm/beta',  'encoder.gn_root.bias')

        stage_cfg = [
            ('resnet50/block1', 'encoder.layer1', [3,3,3]),
            ('resnet50/block2', 'encoder.layer2', [3,3,3,3]),
            ('resnet50/block3', 'encoder.layer3', [3,3,3,3,3,3,3,3,3]),
        ]
        sub_names = [
            ('a', 'conv1', 'gn1'),
            ('b', 'conv2', 'gn2'),
            ('c', 'conv3', 'gn3'),
        ]
        for tf_stage, pt_stage, blk_counts in stage_cfg:
            for i, _ in enumerate(blk_counts):
                tf_blk = f'{tf_stage}/unit{i+1:02d}'
                pt_blk = f'{pt_stage}.{i}'
                for sub, conv_pt, gn_pt in sub_names:
                    _copy(f'{tf_blk}/{sub}/standardized_conv2d/kernel',
                          f'{pt_blk}.{conv_pt}.weight')
                    _copy(f'{tf_blk}/{sub}/group_norm/gamma',
                          f'{pt_blk}.{gn_pt}.weight')
                    _copy(f'{tf_blk}/{sub}/group_norm/beta',
                          f'{pt_blk}.{gn_pt}.bias')
                # projection / downsample
                _copy(f'{tf_blk}/a/proj/standardized_conv2d/kernel',
                      f'{pt_blk}.downsample.weight')

        # ---------- 5-channel conv1 special init ----------
        key_c1 = 'resnet50/root_block/standardized_conv2d/kernel'
        if key_c1 in weights:
            w3 = _pt(weights[key_c1])           # (64,3,7,7)
            pt_k = 'encoder.conv_root.weight'
            if pt_k in new_sd:
                w5 = new_sd[pt_k].clone()       # (64,5,7,7) zeros
                w5[:, :3] = w3
                avg = w3.mean(dim=1, keepdim=True)
                w5[:, 3:5] = avg.expand(-1, 2, -1, -1)
                new_sd[pt_k] = w5
                loaded += 1
                print("  [conv1] ch[0-2]=pretrained, ch[3-4]=mean(pretrained)")
        else:
            print(f"  [WARNING] conv1 key not found: {key_c1}")

        # ---------- ViT encoder ----------
        # CLS token
        for k in ('cls', 'cls_token'):
            if k in weights:
                v = torch.from_numpy(weights[k].copy()).float()
                if v.dim() == 2:
                    v = v.unsqueeze(0)
                new_sd['vit.cls_token'] = v; loaded += 1; break

        # Positional embedding
        for k in ('Transformer/posembed_input/pos_embedding',
                  'posembed_input/pos_embedding'):
            if k in weights:
                new_sd['vit.pos_embed'] = torch.from_numpy(
                    weights[k].copy()).float(); loaded += 1; break

        # Patch embedding (hybrid backbone uses the ResNet feature map)
        for k in ('embedding/kernel', 'Transformer/embedding/kernel',
                  'hybrid/embedding/kernel', 'hybrid_model/embedding/kernel'):
            if k in weights:
                new_sd['vit.patch_embed.weight'] = _pt(weights[k]); loaded += 1; break
        for k in ('embedding/bias', 'Transformer/embedding/bias',
                  'hybrid/embedding/bias', 'hybrid_model/embedding/bias'):
            if k in weights:
                new_sd['vit.patch_embed.bias'] = torch.from_numpy(
                    weights[k].copy()).float(); loaded += 1; break

        # Transformer blocks
        for i in range(VIT_NUM_LAYERS):
            bp = f'Transformer/encoderblock_{i}'
            pp = f'vit.blocks.{i}'
            _copy(f'{bp}/LayerNorm_0/scale', f'{pp}.ln1.weight')
            _copy(f'{bp}/LayerNorm_0/bias',  f'{pp}.ln1.bias')
            _copy(f'{bp}/LayerNorm_2/scale', f'{pp}.ln2.weight')
            _copy(f'{bp}/LayerNorm_2/bias',  f'{pp}.ln2.bias')
            _copy(f'{bp}/MlpBlock_3/Dense_0/kernel', f'{pp}.mlp.fc1.weight')
            _copy(f'{bp}/MlpBlock_3/Dense_0/bias',   f'{pp}.mlp.fc1.bias')
            _copy(f'{bp}/MlpBlock_3/Dense_1/kernel', f'{pp}.mlp.fc2.weight')
            _copy(f'{bp}/MlpBlock_3/Dense_1/bias',   f'{pp}.mlp.fc2.bias')

            attn_bp = f'{bp}/MultiHeadDotProductAttention_1'
            for attn_name, pt_attn in [('query', 'query'), ('key', 'key'),
                                        ('value', 'value'), ('out', 'out')]:
                kk = f'{attn_bp}/{attn_name}/kernel'
                bk = f'{attn_bp}/{attn_name}/bias'
                if kk in weights:
                    w = weights[kk]
                    w = w.reshape(VIT_HIDDEN_SIZE, VIT_HIDDEN_SIZE).T
                    t = torch.from_numpy(w.copy()).float()
                    pk = f'{pp}.attn.{pt_attn}.weight'
                    if pk in new_sd and new_sd[pk].shape == t.shape:
                        new_sd[pk] = t; loaded += 1
                    else:
                        skipped += 1
                if bk in weights:
                    b = weights[bk].reshape(-1)
                    t = torch.from_numpy(b.copy()).float()
                    pk = f'{pp}.attn.{pt_attn}.bias'
                    if pk in new_sd and new_sd[pk].shape == t.shape:
                        new_sd[pk] = t; loaded += 1
                    else:
                        skipped += 1

        # Final LayerNorm
        _copy('Transformer/encoder_norm/scale', 'vit.ln.weight')
        _copy('Transformer/encoder_norm/bias',  'vit.ln.bias')

        # ---------- Load (strict=True uses our updated new_sd) ----------
        self.load_state_dict(new_sd, strict=True)
        print(f"  Done. Loaded: {loaded} | Skipped (shape mismatch): {skipped}")
        print("  Decoder + output head: random init (by design).")
