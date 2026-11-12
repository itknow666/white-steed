"""C3D 3D卷积骨干，按 OpenMMLab C3D 参数布局适配深度伪彩+IR。

输入[B,4,16,H,W]，内部缩放到112；输出固定40类logits。
预训练来源为 Sports1M 预训练后 UCF101 微调；推理构造不会联网。
参考 https://github.com/open-mmlab/mmaction2/blob/main/mmaction/models/backbones/c3d.py
该实现采用 PyTorch 独立层以避免推理时安装 MMCV；上游许可 Apache-2.0。
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


class Conv(nn.Module):
    def __init__(self,inc,outc):
        super().__init__()
        self.conv=nn.Conv3d(inc,outc,3,padding=1,bias=True)
        self.activate=nn.ReLU(inplace=True)
    def forward(self,x):return self.activate(self.conv(x))


class C3DBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1a=Conv(3,64);self.pool1=nn.MaxPool3d((1,2,2),stride=(1,2,2))
        self.conv2a=Conv(64,128);self.pool2=nn.MaxPool3d(2,2)
        self.conv3a=Conv(128,256);self.conv3b=Conv(256,256);self.pool3=nn.MaxPool3d(2,2)
        self.conv4a=Conv(256,512);self.conv4b=Conv(512,512);self.pool4=nn.MaxPool3d(2,2)
        self.conv5a=Conv(512,512);self.conv5b=Conv(512,512);self.pool5=nn.MaxPool3d(2,2,padding=(0,1,1))
        self.fc6=nn.Linear(8192,4096);self.fc7=nn.Linear(4096,4096)
        self.relu=nn.ReLU();self.dropout=nn.Dropout(.5)
    def forward(self,x):
        x=self.pool1(self.conv1a(x));x=self.pool2(self.conv2a(x))
        x=self.pool3(self.conv3b(self.conv3a(x)))
        x=self.pool4(self.conv4b(self.conv4a(x)))
        x=self.pool5(self.conv5b(self.conv5a(x)))
        return self.relu(self.fc7(self.dropout(self.relu(self.fc6(x.flatten(1))))))


class Candidate(nn.Module):
    def __init__(self,name='c3d',pretrained=None,dropout=.5):
        super().__init__()
        if name!='c3d':raise ValueError(name)
        self.encoder=C3DBackbone()
        if pretrained:
            state=torch.load(Path(pretrained),map_location='cpu',weights_only=True)
            state=state.get('state_dict',state)
            src={k.removeprefix('backbone.'):v for k,v in state.items() if k.startswith('backbone.')}
            if not src:raise ValueError('Checkpoint missing expected backbone keys')
            self.encoder.load_state_dict(src,strict=True)
        old=self.encoder.conv1a.conv
        new=nn.Conv3d(4,64,3,padding=1,bias=True)
        with torch.no_grad():
            new.weight[:,:3].copy_(old.weight*.75)
            new.weight[:,3:].copy_(old.weight.mean(1,keepdim=True)*.75)
            new.bias.copy_(old.bias)
        self.encoder.conv1a.conv=new
        self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(4096,40))
        nn.init.normal_(self.head[1].weight,std=.01);nn.init.zeros_(self.head[1].bias)
        self.register_buffer('mean',torch.tensor([104.,117.,128.,(104.+117.+128.)/3]).view(1,4,1,1,1))
    def forward(self,x):
        if x.shape[2]!=16:raise ValueError('C3D expects 16 frames')
        if x.shape[-2:]!=(112,112):
            b,c,t,h,w=x.shape
            x=F.interpolate(x.permute(0,2,1,3,4).reshape(b*t,c,h,w),size=(112,112),mode='bilinear',align_corners=False)
            x=x.reshape(b,t,c,112,112).permute(0,2,1,3,4)
        return self.head(self.encoder(x*255.-self.mean))
