"""四种视频骨干的四通道适配。构造时绝不下载，预训练文件只由训练入口传入。"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

NAMES = ['x3d_m', 'tsm_r50', 'mvit_v2_s', 'uniformer_s']


def four_channels(old):
    cls = nn.Conv2d if isinstance(old, nn.Conv2d) else nn.Conv3d
    new = cls(4, old.out_channels, old.kernel_size, old.stride, old.padding,
              dilation=old.dilation, groups=old.groups, bias=old.bias is not None)
    with torch.no_grad():
        new.weight[:, :3].copy_(old.weight*.75)
        new.weight[:, 3:].copy_(old.weight.mean(1,keepdim=True)*.75)
        if old.bias is not None: new.bias.copy_(old.bias)
    return new


class Candidate(nn.Module):
    def __init__(self, name, pretrained=None, dropout=.5):
        super().__init__()
        self.name = name
        state = torch.load(Path(pretrained), map_location='cpu', weights_only=True) if pretrained else None
        mean, std = [.45]*3, [.225]*3
        if name == 'x3d_m':
            from pytorchvideo.models.x3d import create_x3d
            net = create_x3d(input_clip_length=16,input_crop_size=192,head_activation=None)
            if state is not None: net.load_state_dict(state['model_state'],strict=True)
            net.blocks[0].conv.conv_t = four_channels(net.blocks[0].conv.conv_t)
            dim = net.blocks[-1].proj.in_features
            net.blocks[-1].proj = nn.Identity()
            net.blocks[-1].dropout = nn.Identity()
        elif name == 'tsm_r50':
            from torchvision.models import resnet50
            from temporal_shift_vendor import make_temporal_shift
            net = resnet50(weights=None,num_classes=400)
            make_temporal_shift(net,n_segment=8,n_div=8,place='blockres')
            if state is not None:
                converted = {}
                for key,value in state['state_dict'].items():
                    key = key.removeprefix('module.')
                    key = key.replace('base_model.','',1) if key.startswith('base_model.') else key.replace('new_fc.','fc.',1)
                    converted[key] = value
                net.load_state_dict(converted,strict=True)
            net.conv1 = four_channels(net.conv1)
            dim = net.fc.in_features
            net.fc = nn.Identity()
            mean,std = [.485,.456,.406],[.229,.224,.225]
        elif name == 'mvit_v2_s':
            from torchvision.models.video import mvit_v2_s
            net = mvit_v2_s(weights=None)
            if state is not None: net.load_state_dict(state,strict=True)
            net.conv_proj = four_channels(net.conv_proj)
            dim = net.head[1].in_features
            net.head = nn.Identity()
        elif name == 'uniformer_s':
            from uniformer_vendor import uniformer_small
            net = uniformer_small()
            if state is not None: net.load_state_dict(state,strict=True)
            net.patch_embed1.proj = four_channels(net.patch_embed1.proj)
            dim = net.head.in_features
            net.head = nn.Identity()
        else: raise ValueError(name)
        self.encoder = net
        self.head = nn.Sequential(nn.Dropout(dropout),nn.Linear(dim,40))
        nn.init.normal_(self.head[1].weight,std=.01)
        nn.init.zeros_(self.head[1].bias)
        self.register_buffer('mean',torch.tensor(mean+[sum(mean)/3]).view(1,4,1,1,1))
        self.register_buffer('std',torch.tensor(std+[sum(std)/3]).view(1,4,1,1,1))

    def forward(self,x):
        # 全部缓存来自同步16帧192；TSM固定取8帧，MViT内部上采样到其预训练224网格。
        if self.name == 'tsm_r50':
            if x.shape[2] != 8:
                ids = torch.linspace(0,x.shape[2]-1,8,device=x.device).long()
                x = x[:,:,ids]
            x = (x-self.mean)/self.std
            b,c,t,h,w = x.shape
            x = x.permute(0,2,1,3,4).reshape(b*t,c,h,w)
            feature = self.encoder(x).reshape(b,t,-1).mean(1)
        else:
            if self.name == 'mvit_v2_s' and x.shape[-2:] != (224,224):
                b,c,t,h,w = x.shape
                x = F.interpolate(x.permute(0,2,1,3,4).reshape(b*t,c,h,w),size=(224,224),mode='bilinear',align_corners=False)
                x = x.reshape(b,t,c,224,224).permute(0,2,1,3,4)
            feature = self.encoder((x-self.mean)/self.std)
        return self.head(feature)
