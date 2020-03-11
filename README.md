## Bengali.ai Handwritten Character Recognition Challenge
My scripts for the [Bengali.ai handwritten character Recognition challenge 2019](https://www.kaggle.com/c/bengaliai-cv19).

Trained using customized 1 channel input `densenet` and `se-resnext` model.

Used `OHEM Loss` for training.

So far my best score is obtained from se-resnext50. It scored 0.981 on validation and `0.9707` on LB.


**Update**: Another se-resnext50 model scores 0.9805 on val data but had lower loss. It achieved `0.9717` on LB.


**Things that works**:
 - SE-Resnext50
 - CutMix + MixUp
 - OHEM Loss
 - Gem (Did not compare with `AvgPool`)
 - Mish (Did not compare with `ReLU`)
 - Training for longer epochs and no early stopping
 - Rely on validation loss rather than recall score.

**Things that may work**:
- AugMix
- EfficientNet with customized last Convolutional Layer
- SE-Resnext101 (Teacher-Student model)
- Focal Loss with EfficientNet-B4

**Important Discussions**:
- Validation [Data](https://www.kaggle.com/haqishen/validation-with-unseen)
- [Focal Loss + OHEM](https://www.kaggle.com/c/bengaliai-cv19/discussion/128665): Note: Unstable
- [Discussion](https://www.kaggle.com/c/bengaliai-cv19/discussion/128911) on how to train with EfficientNet
- [MaxBlurPool2D](https://www.kaggle.com/c/bengaliai-cv19/discussion/125819)
- [Augmix](https://www.kaggle.com/c/bengaliai-cv19/discussion/129697) Implementation  

**Resources**:
- Half Precision GeM:

```
def gem(x, p=3, eps=1e-6):
    x = x.double() # x=x.to(torch.float32) # comment this during inference
    x = F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
    return x.half() # Comment this line in inference code use ## return x

class GeM(nn.Module):
    def init(self, p=3, eps=1e-6):
        super(GeM,self).init()
        # super().init()
        self.p = Parameter(torch.ones(1)*p)
        # print(self.p.dtype)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
    def repr(self):
        return self.class.name + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'
```


**More resources will be added soon.**