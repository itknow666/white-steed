"""TorchVision Video Swin-T：公开Kinetics400预训练迁移到深度+IR四通道。

模型实现来自torchvision（BSD-3-Clause）；这里只适配输入和40类分类头。
推理时构造weights=None，不发起网络请求。
"""
from pathlib import Path
import torch
from torch import nn
from torchvision.models.video import swin3d_t


class VideoSwin(nn.Module):
    def __init__(self, pretrained=None, dropout=.5):
        super().__init__()
        net = swin3d_t(weights=None)
        if pretrained:
            net.load_state_dict(torch.load(Path(pretrained), map_location='cpu', weights_only=True), strict=True)
        old = net.patch_embed.proj
        patch = nn.Conv3d(4, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None)
        with torch.no_grad():
            patch.weight[:, :3].copy_(old.weight*.75)
            patch.weight[:, 3:].copy_(old.weight.mean(1, keepdim=True)*.75)
            if old.bias is not None:
                patch.bias.copy_(old.bias)
        net.patch_embed.proj = patch
        dim = net.head.in_features
        net.head = nn.Identity()
        self.encoder = net
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 40))
        nn.init.normal_(self.head[1].weight, std=.01)
        nn.init.zeros_(self.head[1].bias)
        mean, std = [.485,.456,.406], [.229,.224,.225]
        self.register_buffer('mean', torch.tensor(mean+[sum(mean)/3]).view(1,4,1,1,1))
        self.register_buffer('std', torch.tensor(std+[sum(std)/3]).view(1,4,1,1,1))

    def forward(self, x):
        return self.head(self.encoder((x-self.mean)/self.std))
