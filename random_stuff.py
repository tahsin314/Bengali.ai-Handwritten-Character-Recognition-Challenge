import pandas as pd 
import numpy as np 
from copy import deepcopy
from tqdm import tqdm as T
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from BanglaDataset import *

n_fold = 5
SEED = 24
fold = 0
train_df = pd.read_csv('data/train.csv')
nunique = list(train_df.nunique())[1:-1]
train_df['id'] = train_df['image_id'].apply(lambda x: int(x.split('_')[1]))
X, y = train_df[['id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values[:,0], train_df.values[:,1:]
train_df['fold'] = np.nan

#split data
mskf = MultilabelStratifiedKFold(n_splits=n_fold, random_state=SEED)
for i, (_, test_index) in enumerate(mskf.split(X, y)):
    train_df.iloc[test_index, -1] = i
    
# train_df['fold'] = train_df['fold'].astype('int')
# train_idx = []
# val_idx = []
# # writer.close()

# for i in T(range(len(train_df))):
#     if train_df.iloc[i]['fold'] == fold: val_idx.append(i)
#     else: train_idx.append(i)

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
    df = df.set_index('image_id')
    for i in T(range(df.shape[0])):
       # image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))
        image0 = 255 - df.loc[df.index[i]].values.reshape(137,236).astype(np.uint8)
    #normalize each image by its max val
        img = (image0*(255.0/image0.max())).astype(np.uint8)
        # print(img.max())
        image = crop_resize(img)
        # print(image.max())
        resized[df.index[i]] = image.reshape(-1)
    resized = pd.DataFrame(resized).T.reset_index()
    resized.columns = resized.columns.astype(str)
    resized.rename(columns={'index':'image_id'},inplace=True)
    return resized

# val_data = pd.DataFrame(columns=['image_id', *(str(i) for i in range(32332))])
# # val_data['id'] = val_idx

# for i in T(range(4)):
#     prqt = pd.read_parquet('data/train_image_data_{}.parquet'.format(i))
#     l = len(prqt)
#     tmp_idx = []
#     for fb in T(train_idx):
#         if i*l<=fb<(i+1)*l:
#             tmp_idx.append(fb%l)
#     prqt = prqt.drop(prqt.index[tmp_idx])
#     val_data = val_data.append(prqt, ignore_index = True)

# print(len(val_data))
# # val_data.iloc[:, 1:].values = val_data.iloc[:, 1:].values.astype(np.int8)
# for i in T(range(32332)):
#     val_data[str(i)]=val_data[str(i)].astype(np.int8) 

# print(val_data.head())
# val_data.to_parquet('val_data.parquet', index=False)

prqt = pd.read_parquet('data/train_image_data_{}.parquet'.format(0))
df0 = Resize(prqt)
prqt = pd.read_parquet('data/train_image_data_{}.parquet'.format(1))
df1 = Resize(prqt)
prqt = pd.read_parquet('data/train_image_data_{}.parquet'.format(2))
df2 = Resize(prqt)
prqt = pd.read_parquet('data/train_image_data_{}.parquet'.format(3))
df3 = Resize(prqt)
df = pd.concat([df0, df1, df2, df3], ignore_index=True)

df.to_parquet('train_images_data.parquet', index=False)
# for i in T(range(4)):
#     prqt = pd.read_parquet('data/train_image_data_{}.parquet'.format(i))
#     df = Resize(prqt)
# df = pd.read_parquet('train_images_data.parquet')

# for i in T(train_idx):
# df = df.drop(df.index[train_idx])

# df.to_csv('val_images_data.parquet', index=False)