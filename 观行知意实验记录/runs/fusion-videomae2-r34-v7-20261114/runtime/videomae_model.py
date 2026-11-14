"""VideoMAE V2 蒸馏 ViT-S/16（K710 微调，公开权重 OpenGVLab/VideoMAE2 distill/vit_s_k710_dl_from_giant.pth，Apache-2.0）的四通道适配。

构造时不下载；预训练文件只由训练入口传入。输入为缓存的16帧192四通道（伪彩深度RGB+IR），
内部双线性上采样到预训练的224网格（VideoMAE 使用按展平序号的一维正弦位置编码，不能直接换网格）。
"""
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from candidate_models import four_channels


def sinusoid_table(n, dim):
    pos = np.arange(n)[:, None]
    i = np.arange(dim)[None, :]
    angle = pos / np.power(10000, 2 * (i // 2) / dim)
    angle[:, 0::2] = np.sin(angle[:, 0::2])
    angle[:, 1::2] = np.cos(angle[:, 1::2])
    return torch.tensor(angle, dtype=torch.float32).unsqueeze(0)


class DropPath(nn.Module):
    def __init__(self, p=0.):
        super().__init__(); self.p = p
    def forward(self, x):
        if self.p == 0. or not self.training: return x
        keep = 1 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        b, n, c = x.shape
        bias = torch.cat([self.q_bias, torch.zeros_like(self.v_bias), self.v_bias])
        qkv = F.linear(x, self.qkv.weight, bias).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__(); self.fc1 = nn.Linear(dim, hidden); self.act = nn.GELU(); self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x): return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, heads, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6); self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6); self.mlp = Mlp(dim, dim * 4)
        self.drop_path = DropPath(drop_path)
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.proj = nn.Conv3d(3, dim, (2, 16, 16), (2, 16, 16))
    def forward(self, x): return self.proj(x).flatten(2).transpose(1, 2)


class VideoMAEEncoder(nn.Module):
    def __init__(self, dim=384, depth=12, heads=6, frames=16, size=224, drop_path=.1):
        super().__init__()
        self.patch_embed = PatchEmbed(dim)
        n = (frames // 2) * (size // 16) ** 2
        self.register_buffer('pos_embed', sinusoid_table(n, dim), persistent=False)
        rates = [float(v) for v in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([Block(dim, heads, r) for r in rates])
        self.fc_norm = nn.LayerNorm(dim, eps=1e-6)
        self.num_features = dim
    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed.to(x.dtype)
        for blk in self.blocks: x = blk(x)
        return self.fc_norm(x.mean(1))


class VideoMAEClassifier(nn.Module):
    name = 'videomae2_s'
    def __init__(self, pretrained=None, dropout=.5, drop_path=.1):
        super().__init__()
        net = VideoMAEEncoder(drop_path=drop_path)
        if pretrained:
            state = torch.load(Path(pretrained), map_location='cpu', weights_only=False)
            state = state.get('module', state.get('model', state))
            state = {k: v.float() for k, v in state.items() if not k.startswith('head.')}
            missing, unexpected = net.load_state_dict(state, strict=False)
            if missing or unexpected: raise RuntimeError(f'VideoMAE weights mismatch: {missing} {unexpected}')
        net.patch_embed.proj = four_channels(net.patch_embed.proj)
        self.encoder = net
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.num_features, 40))
        nn.init.normal_(self.head[1].weight, std=.01); nn.init.zeros_(self.head[1].bias)
        mean, std = [.485, .456, .406], [.229, .224, .225]
        self.register_buffer('mean', torch.tensor(mean + [sum(mean) / 3]).view(1, 4, 1, 1, 1))
        self.register_buffer('std', torch.tensor(std + [sum(std) / 3]).view(1, 4, 1, 1, 1))
    def forward(self, x):
        if x.shape[-2:] != (224, 224):
            b, c, t, h, w = x.shape
            x = F.interpolate(x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w), size=(224, 224), mode='bilinear', align_corners=False)
            x = x.reshape(b, t, c, 224, 224).permute(0, 2, 1, 3, 4)
        return self.head(self.encoder(((x - self.mean) / self.std).contiguous()))

    def layer_groups(self, lr, decay, head_multiplier, weight_decay):
        """逐层学习率衰减：越靠近输入，学习率越小。返回 AdamW 参数组。"""
        depth = len(self.encoder.blocks)
        groups = {}
        for name, p in self.named_parameters():
            if name.startswith('head.'): layer, scale = depth + 1, head_multiplier
            elif name.startswith('encoder.blocks.'):
                layer = int(name.split('.')[2]) + 1; scale = decay ** (depth + 1 - layer)
            elif name.startswith('encoder.fc_norm'): layer, scale = depth + 1, 1.0
            else: layer, scale = 0, decay ** (depth + 1)
            no_wd = p.ndim == 1 or name.endswith('_bias')
            key = (layer, no_wd, scale)
            if key not in groups:
                groups[key] = {'params': [], 'lr': lr * scale, 'lr_scale': scale, 'weight_decay': 0. if no_wd else weight_decay}
            groups[key]['params'].append(p)
        return list(groups.values())