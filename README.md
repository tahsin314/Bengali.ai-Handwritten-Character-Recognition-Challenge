## Bengali.ai Handwritten Character Recognition Challenge
My scripts for the [Bengali.ai handwritten character Recognition challenge 2019](https://www.kaggle.com/c/bengaliai-cv19).

Trained using customized 1 channel input `densenet` and `se-resnext` model.

Used `OHEM Loss` for training.

So far my best score is obtained from se-resnext50. It scored 0.981 on validation and 0.9707 on LB.

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

**More resources will be added soon.**