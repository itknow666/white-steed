"""R(2+1)D-34 IG65M/K400, matching the public Depth_Color+IR baseline."""
from pathlib import Path
import torch
from torch import nn
from ig65m_backbone import r2plus1d_34

MEAN=(.43216,.394666,.37645)
STD=(.22803,.22145,.216989)

class RawModel(nn.Module):
    def __init__(self, channels=4, pretrained=None, dropout=.3):
        super().__init__()
        if channels not in (3,4): raise ValueError('channels must be 3 or 4')
        net=r2plus1d_34()
        # torch.hub upstream recipe sets Caffe2-compatible BN momentum to .9.
        for module in net.modules():
            if isinstance(module,nn.BatchNorm3d): module.momentum=.9
        if pretrained:
            state=torch.load(Path(pretrained),map_location='cpu',weights_only=True)
            net.load_state_dict(state,strict=True)
        old=net.stem[0]
        if channels==4:
            new=nn.Conv3d(4,old.out_channels,old.kernel_size,old.stride,old.padding,bias=False)
            with torch.no_grad():
                new.weight[:,:3].copy_(old.weight)
                new.weight[:,3:].copy_(old.weight.mean(1,keepdim=True))
            net.stem[0]=new
        dim=net.fc.in_features
        net.fc=nn.Identity()
        self.encoder=net
        self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(dim,40))
        mean=list(MEAN)+([sum(MEAN)/3] if channels==4 else [])
        std=list(STD)+([sum(STD)/3] if channels==4 else [])
        self.register_buffer('mean',torch.tensor(mean).view(1,channels,1,1,1))
        self.register_buffer('std',torch.tensor(std).view(1,channels,1,1,1))

    def forward(self,x):
        return self.head(self.encoder((x-self.mean)/self.std))
