import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import gc
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
from random import choices
# Any results you write to the current directory are saved as output.
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
from tqdm import tqdm_notebook as tqdm
from utils import *
import warnings
warnings.filterwarnings('ignore')

HEIGHT = 137
WIDTH = 236

#check https://www.kaggle.com/iafoss/image-preprocessing-128x128

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size=128, pad=16):
    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return 255-cv2.resize(img,(size,size))

def Resize(df,size=128):
    resized = {} 
    df = df.set_index('image_id')
    for i in tqdm(range(df.shape[0])):
       # image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))
        image0 = 255 - df.loc[df.index[i]].values.reshape(137,236).astype(np.uint8)
    #normalize each image by its max val
        img = (image0*(255.0/image0.max())).astype(np.uint8)
        image = crop_resize(img)
        resized[df.index[i]] = image.reshape(-1)
    resized = pd.DataFrame(resized).T.reset_index()
    resized.columns = resized.columns.astype(str)
    resized.rename(columns={'index':'image_id'},inplace=True)
    return resized


class GraphemeDataset(Dataset):
    def __init__(self, df, data, ImageIdx, _type='train', aug = None, return_name=False):
        self.df = df
        self.data = data
        # self.label = self.df
        self.aug = aug
        self.dirname = dirname
        self.ImageIdx = ImageIdx
        self._type = _type
        self.return_name = return_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        label1 = self.label.vowel_diacritic.values[idx]
        label2 = self.label.grapheme_root.values[idx]
        label3 = self.label.consonant_diacritic.values[idx]
        #image = self.df.iloc[idx][1:].values.reshape(128,128).astype(np.float)
        image = self.data[idx, :].reshape(128,128).astype(np.float)
        if self.transform:
            augment = self.aug(image =image)
            image = augment['image']
            cutout = Cutout(32,0.5,True,1)
            image = cutout(image)
        norm = Normalize([0.0692],[0.2051])
        image = norm(image)

        return image,label1,label2,label3

class BanglaDataset(Dataset):
    def __init__(self, df, dirname, ImageIdx, _type='train', aug = None, return_name=False):
        self.df = df
        # self.label = self.df
        self.aug = aug
        self.dirname = dirname
        self.ImageIdx = ImageIdx
        self._type = _type
        self.return_name = return_name

    def __len__(self):
        return len(self.ImageIdx)

    def __getitem__(self,idx):

        # image = cv2.imread(os.path.join(self.dirname, '{}.png'.format(self.df['image_id'][self.ImageIdx[idx]])), cv2.IMREAD_UNCHANGED)
        # image = image.reshape(*image.shape, 1)
        # print(os.path.join(self.dirname, '{}.npy'.format(self.df['image_id'][self.ImageIdx[idx]])))
        # choice = choices(['normal', 'erode', 'dilate'], weights=[0.60, 0.25, 0.15])
        choice = choices(['normal', 'erode', 'dilate'], weights=[1.0, 0.0, 0.0])
        image = np.load(os.path.join(self.dirname, '{}.npy'.format(self.df['image_id'][self.ImageIdx[idx]])))
        # print(image.shape)
        image = cv2.cvtColor(image.reshape(*image.shape, 1), cv2.COLOR_GRAY2RGB).astype(np.float)
        image= cv2.resize(image, (224, 224))/255.
        # image = image.reshape(*image.shape, 1)
        # image = cv2.remap(image)
        # print(image.shape)
        if choice[0] == 'normal':
            pass
        elif choice[0] == 'erode':
            image = erode(image, 3)
        else:
            image = dilate(image, 3)
        
        label1 = self.df.grapheme_root.values[self.ImageIdx[idx]]
        label2 = self.df.vowel_diacritic.values[self.ImageIdx[idx]]
        label3 = self.df.consonant_diacritic.values[self.ImageIdx[idx]]
        # image = self.data[idx, :].reshape(128,128).astype(np.float)
        if self.aug is not None:
            # print(image.shape)
            augment = self.aug(image = image)
            # image = augment['image'].transpose(2, 0, 1)
            # image = augment['image'].reshape(1, *image.shape)
            image = augment['image'].transpose(2, 0, 1)

        else:
            # image = image.reshape(1, *image.shape)
            image = image.transpose(2, 0, 1)
        if self.return_name:
            return self.df['image_id'][self.ImageIdx[idx]], image,label1,label2,label3
        
        return image,label1,label2,label3
