## Bengali.ai Handwritten Character Recognition Challenge: 100-ish solution(?)
My scripts for the [Bengali.ai handwritten character Recognition challenge 2019](https://www.kaggle.com/c/bengaliai-cv19). Although I could not submit the final trained models within the competition deadline, later I found out that it would get me a `100`-ish private Leadeboard position.

I added customized 1 channel input `seresext`, `densenet`, `efficientnet` and `ghostnet` models in this repo.

**My Method**:
 - Trained each model for around 100 epochs with `OHEM cross-entropy` loss.
 - Trained the first 30 epochs with `rate=1` for OHEM loss. Then gradually decreased the rate to 0.65 over the next 35 epochs. This approach significantly increased my *Grapheme* recall. 
- Applied *CutOut, CutMix and Mixup* augmentations.
- Used heavy augmentations during training `se-resnext101` model.

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
- [Half Precision GeM](https://www.kaggle.com/c/bengaliai-cv19/discussion/128911):

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

**How to run**
- Run `pip install -r requirements.txt`
- Run `train.py`

**Credits**

Thanks to the kaggle community for sharing a lot of resources and making the tasks easier for us. Here is a list of discussions from where I've borrowed codes and/or became inspired to write codes:

- [Augmix](https://www.kaggle.com/c/bengaliai-cv19/discussion/129697)
- [GridMask](https://www.kaggle.com/c/bengaliai-cv19/discussion/128161)
- [MixUp and CutMix](https://www.kaggle.com/c/bengaliai-cv19/discussion/126504)
- [OHEM Loss](https://www.kaggle.com/c/bengaliai-cv19/discussion/128637)

**More resources will be added soon.**