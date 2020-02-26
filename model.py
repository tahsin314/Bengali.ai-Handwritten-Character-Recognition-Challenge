from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import pretrainedmodels

from fastai.vision import *

def get_cadene_model(model_name='se_resnext101_32x4d', pretrained=True):
    if pretrained:
        arch = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    else:
        arch = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=None)
    return arch

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class Head(nn.Module):
    def __init__(self, nc, n, ps=0.5, pool = 'gem'):
        super().__init__()
        if pool == 'gem':
            layers = [GeM(), Mish(), Flatten()] + \
            bn_drop_lin(nc*2, 512, True, ps, Mish()) + \
            bn_drop_lin(512, n, True, ps)
        else:
            layers = [AdaptiveConcatPool2d(), Mish(), Flatten()] + \
            bn_drop_lin(nc*2, 512, True, ps, Mish()) + \
            bn_drop_lin(512, n, True, ps)
        self.fc = nn.Sequential(*layers)
        self._init_weight()
        
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
        
    def forward(self, x):
        return self.fc(x)


class MishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.tanh(F.softplus(x))   # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        tanh_sp = torch.tanh(F.softplus(x)) 
        return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))

class Mish(nn.Module):
    def forward(self, x):
        return MishFunction.apply(x)

def to_Mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Mish())
        # if isinstance(child, utils.MemoryEfficientSwish):
        #     setattr(model, child_name, Mish())
        else:
            to_Mish(child)

class seresnext(nn.Module):

    def __init__(self, n, model_name='se_resnext101_32x4d'):
        super().__init__()
        self.backbone = get_cadene_model(model_name)
#         in_features = self.backbone.fc.in_features
        # for Resnet
        # self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        arch = get_cadene_model(model_name)
        # arch = list(arch.layer0.children())
        # w = arch[0].weight
        # self.backbone.layer0.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.backbone.layer0.conv1.weight = nn.Parameter(torch.mean(w, dim=1, keepdim=True))
        self.backbone.layer0.relu1 = Mish()
        nc = 1024 # 512 if res34
        
        # self.layer3_graph = self.backbone.layer3
        # self.layer3 = self.backbone.layer3

        self.layer4_1 = self.backbone.layer4
        self.layer4_2 = self.backbone.layer4
        self.layer4_3 = self.backbone.layer4
        
        self.head1 = Head(nc,n[0])
        self.head2 = Head(nc,n[1])
        self.head3 = Head(nc,n[2])
        
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