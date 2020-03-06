import os
import pandas as pd 
import numpy as np 
from p_tqdm import p_map
from tqdm import tqdm as T
import cv2
img_dir='data/numpy_format'

try:
    os.mkdir(img_dir)
except:
    pass

# df = pd.read_parquet('train_images_data.parquet')
# l = len(df)

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
    return cv2.resize(img,(size,size))

def Resize(df,size=128):
    resized = {}
    char_data = {'id': list(df.iloc[:, 0]), 'data':np.array(df.iloc[:, 1:])}
    for i in T(range(len(char_data['id']))):
       # image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))
        # image0 = 255 - df.loc[df.index[i]].values.reshape(137,236).astype(np.uint8)
        image0 = 255 - np.copy(char_data['data'][i]).reshape(HEIGHT, WIDTH)
        name = char_data['id'][i]
    #normalize each image by its max val
        # print(image0.max())
        img = (image0*(255.0/image0.max())).astype(np.uint8)
        # print('Before crop: ', np.max(img))
        image = crop_resize(img)
        np.save(os.path.join(img_dir, name+'.npy'), image)

def save_as_npy(df):
    resized = {}
    char_data = {'id': list(df.iloc[:, 0]), 'data':np.array(df.iloc[:, 1:])}
    for i in T(range(len(char_data['id']))):
        image = np.copy(char_data['data'][i]).reshape(HEIGHT, WIDTH)
        name = char_data['id'][i]
        np.save(os.path.join(img_dir, name+'.npy'), image)

def col_to_numpy(idx):
    img = np.array(df.iloc[idx][1:]).reshape(128, 128)
    np.save(os.path.join(img_dir, df.iloc[idx][0]+'.npy'), img, allow_pickle=True)

# prqt = pd.read_parquet('data/train_image_data_{}.parquet'.format(0))
# df0 = Resize(prqt)
# prqt = pd.read_parquet('data/train_image_data_{}.parquet'.format(1))
# df1 = Resize(prqt)
# prqt = pd.read_parquet('data/train_image_data_{}.parquet'.format(2))
# df2 = Resize(prqt)
# prqt = pd.read_parquet('data/train_image_data_{}.parquet'.format(3))
# df3 = Resize(prqt)
# df = pd.concat([df0, df1, df2, df3], ignore_index=True)

# df.to_parquet('train_images_data.parquet', index=False)

# p_map(col_to_numpy, list(range(l)))
# p_map(Resize, ['data/train_image_data_{}.parquet'.format(i) for i in range(4)])
for i in T(range(4)):
    df = pd.read_parquet('data/train_image_data_{}.parquet'.format(i))
    # Resize(df)
    save_as_npy(df)
# x = np.load('data/128_numpy/Train_10.npy')
# print(df.head()