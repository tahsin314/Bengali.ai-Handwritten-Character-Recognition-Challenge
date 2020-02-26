import os
import pandas as pd
import numpy as np
import PIL
import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont
import cv2
from tqdm import tqdm as T 
from p_tqdm import p_map

HEIGHT = 137
WIDTH = 236
SIZE = 128

dir_name = 'data/train_128px'

if os.path.exists(dir_name):
    print("Directory Already Exists.")
else:
    os.mkdir(dir_name)

def image_creator(idx):
    df = pd.read_parquet('data/train_image_data_{}.parquet'.format(idx))
    
    for i in range(len(df)):
        flattened_image = df.iloc[i].drop('image_id').values.astype(np.uint8).reshape(137, 236)
        img = cv2.cvtColor(flattened_image, cv2.COLOR_BGR2RGB)
        # img = np.pad(img, (59, 60, 10, 10), 'constant', 255)
        img = cv2.copyMakeBorder(img, 59, 60, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.imwrite(os.path.join(dir_name, df['image_id'][i]+'.png'), img)
        # unpacked_image.save(os.path.join(dir_name, df['image_id'][i]+'.png'))

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size=SIZE, pad=16):
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
    return cv2.resize(img,(size,size))

def image_128(i):
    df = pd.read_parquet('data/train_image_data_{}.parquet'.format(i))
    #the input is inverted
    data = 255 - df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)
    for idx in range(len(df)):
        name = df.iloc[idx, 0]
        #normalize each image by its max val
        img = (data[idx]*(255.0/data[idx].max())).astype(np.uint8)
        img = crop_resize(img)
        cv2.imwrite(os.path.join(dir_name, name+'.png'), img)


# os.mkdir('data/images')
# for j in range(3, 4):
#     df = pd.read_parquet('data/train_image_data_{}.parquet'.format(j))
#     dirname = 'data/images'
#     image_creator(df, dirname)
p_map(image_128, list(range(0, 4)))