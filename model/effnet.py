## Code copied from Lukemelas github repository have a look
## https://github.com/lukemelas/EfficientNet-PyTorch/tree/master/efficientnet_pytorch
"""
This file contains helper functions for building the model and for loading model parameters.
These helper functions are built to mirror those in the official TensorFlow implementation.
"""

import re
import math
import collections
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo
from .utils import *
from efficientnet_pytorch import EfficientNet

class EfficientNetWrapper(nn.Module):
    def __init__(self, pretrained_model='efficientnet-b4'):
        super(EfficientNetWrapper, self).__init__()
        
        # Load imagenet pre-trained model 
        self.backbone = EfficientNet.from_pretrained(pretrained_model, in_channels=1).to('cuda:0')
        nc = 1792
        self.bn = nn.BatchNorm2d(nc, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        n = [168, 11, 7]
        self.head1 = Head(nc,n[0])
        self.head2 = Head(nc,n[1])
        self.head3 = Head(nc,n[2])
        
    def forward(self, X):
        output = self.backbone.extract_features(X)
        output = self.bn(output)
        out1 = self.head1(output)
        out2 = self.head2(output)
        out3 = self.head3(output)
        return out1, out2, out3
