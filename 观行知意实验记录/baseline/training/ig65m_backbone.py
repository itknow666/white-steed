"""IG65M-compatible R(2+1)D34 architecture.

Reference (MIT): https://github.com/moabitcoin/ig65m-pytorch/blob/master/ig65m/models.py
Compatibility convolution dimensions and BN epsilon match that release.
RawModel sets BN momentum to 0.9 during fine-tuning; the backbone factory
otherwise retains the torchvision default until RawModel configures it.
"""
from torch import nn
from torchvision.models.video.resnet import BasicBlock,Conv2Plus1D,R2Plus1dStem,VideoResNet

WEIGHT_URL='https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth'

def r2plus1d_34(num_classes=400):
    model=VideoResNet(block=BasicBlock,conv_makers=[Conv2Plus1D]*4,layers=[3,4,6,3],stem=R2Plus1dStem,num_classes=num_classes)
    for layer,channels,mid in [(model.layer2,128,288),(model.layer3,256,576),(model.layer4,512,1152)]:
        layer[0].conv2[0]=Conv2Plus1D(channels,channels,mid)
    for module in model.modules():
        if isinstance(module,nn.BatchNorm3d):module.eps=1e-3
    return model
