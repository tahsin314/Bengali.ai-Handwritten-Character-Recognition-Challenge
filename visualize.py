
import os
from matplotlib import pyplot as plt
import pandas as pd
import cv2
from torchvision.utils import save_image
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    Resize,
    CenterCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,    
    Cutout,
    RandomGamma,
    ShiftScaleRotate ,
    GaussNoise,
    Blur,
    MotionBlur,   
    GaussianBlur,
    Normalize, 
)
from BanglaDataset import *
from utils import *

sz = 128
train_aug =Compose([
  ShiftScaleRotate(p=1,border_mode= cv2.BORDER_CONSTANT, value=0, scale_limit=0.25),
    # Cutout(p=0.25, max_h_size=sz//16, max_w_size=sz//16,num_holes=12),
    GridMask(num_grid=5, p=0.3),
    # OneOf([
    #     ElasticTransform(p=0.1, alpha=1, sigma=50, alpha_affine=30,border_mode=cv2.BORDER_CONSTANT,value =0),
    #     GridDistortion(distort_limit =0.05 ,border_mode=cv2.BORDER_CONSTANT,value =0, p=0.1),
    #     OpticalDistortion(p=0.1, distort_limit= 0.05, shift_limit=0.2,border_mode=cv2.BORDER_CONSTANT,value =0)                  
    #     ], p=0.3),
    OneOf([
        GaussNoise(var_limit=0.01),
        Blur(),
        GaussianBlur(blur_limit=3)
        ], p=1.0),
    RandomGamma(p=0.2)])

sz = 128

def visualize(original_image):
    fontsize = 18
    fig = plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(8,8,1)
    plt.axis('off')
    ax.imshow(original_image, cmap='gray')
    for i in range(63):
        augment = train_aug(image = image)
        aug_img = augment['image']
        ax = fig.add_subplot(8,8,i+2)
        plt.axis('off')
        ax.imshow(aug_img, cmap='gray')
    fig.savefig('aug.png')

train_df = pd.read_csv('data/train.csv')
dirname = 'data/128_numpy'
# dirname = 'data/train_128px'
train_ds = BanglaDataset(train_df, dirname, [i for i in range(64)], aug=train_aug)
train_loader = DataLoader(train_ds,batch_size=64, shuffle=True)
im, _, _, _ = iter(train_loader).next()
print(im.size(), torch.max(im))
save_image(im, 'images.png', nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
# image = cv2.imread(os.path.join(dirname, '{}.png'.format(train_df['image_id'][65453])), cv2.IMREAD_GRAYSCALE)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image= cv2.resize(image, (224, 224))

# augment = train_aug(image = image)
# aug_img = augment['image']
# visualize(image)