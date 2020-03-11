from copy import deepcopy
import torch
from torch import nn
from torch.nn import *
from torch.nn import functional as F
from torchvision import models
import pretrainedmodels
from typing import Optional
from .utils import *

class seresnext(nn.Module):

    def __init__(self, n, model_name='se_resnext101_32x4d'):
        super().__init__()
        self.backbone = get_cadene_model(model_name)
#         in_features = self.backbone.fc.in_features
        # for Resnet
        # print(self.backbone.layer4)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        arch = get_cadene_model(model_name)
        arch_layer0 = list(arch.layer0.children())
        arch_layer4 = list(arch.layer4.children())
        w = arch_layer0[0].weight
        # self.backbone.layer0.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.backbone.layer0.conv1.weight = nn.Parameter(torch.sum(w, dim=1, keepdim=True))
        self.backbone.layer0.relu1 = Mish()
        # nc = 2048 # 512 if res34
        # nc = self.backbone.layer4.conv3.weight.shape[0]
        nc = 2048
        
        # self.layer3_graph = self.backbone.layer3
        # self.layer3 = self.backbone.layer3

        self.layer4_1 = self.backbone.layer4
        self.layer4_2 = self.backbone.layer4
        self.layer4_3 = self.backbone.layer4
        
        self.head1 = Head(nc,n[0], activation='mish')
        self.head2 = Head(nc,n[1], activation='mish')
        self.head3 = Head(nc,n[2], activation='mish')
        
        to_Mish(self.backbone.layer1), to_Mish(self.backbone.layer2), to_Mish(self.backbone.layer3)
        to_Mish(self.backbone.layer4)
        

    def forward(self, x):

        x = self.backbone.layer0.conv1(x)
        x = self.backbone.layer0.bn1(x)
        x = self.backbone.layer0.relu1(x)
        x = self.backbone.layer0.pool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        
        # x1 = self.layer3_graph(x)
        x = self.backbone.layer3(x)
        # x3 = self.layer3(x)
        
        x = self.backbone.layer4(x)
        # x1 = self.layer4_1(x)
        # x2 = self.layer4_2(x)
        # x3 = self.layer4_3(x)
        
        x1 = self.head1(x)
        x2 = self.head2(x)
        x3 = self.head3(x)
        
        return x1,x2,x3