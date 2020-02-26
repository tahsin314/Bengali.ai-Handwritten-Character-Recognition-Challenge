import os
import sys
import curses 
import gc
import time
from random import choices
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn
import cv2
from tqdm import tqdm as T
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from apex import amp
import torch, torchvision
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from BanglaDataset import BanglaDataset, GraphemeDataset
from utils import *
from metrics import *
from optimizers import Over9000
from model import seresnext
## This library is for augmentations .
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
n_fold = 5
fold = 0
SEED = 24
batch_size = 48
sz = 128
learning_rate = 1e-3
thr = 0.9
opts = ['normal', 'mixup', 'cutmix']
device = 'cuda:0'
apex = False
pretrained_model = 'se_resnext50_32x4d'
# model_name = '{}_trial_stage1_fold_{}_loss.pth'.format(pretrained_model, fold)
weight_file = 'model_weights_best_recall.pth'
load_model = True
results = pd.DataFrame()
pred_thr = pd.DataFrame()
n_epochs = 60
valid_recall = 0.0
best_valid_recall = 0.0

# val_aug = Compose([Normalize([0.0692], [0.2051])])
val_aug = Compose([Normalize()])

train_df = pd.read_csv('data/train.csv')
nunique = list(train_df.nunique())[1:-1]
train_df['id'] = train_df['image_id'].apply(lambda x: int(x.split('_')[1]))
train_df= train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
X, y = train_df[['id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values[:,0], train_df.values[:,1:]
train_df['fold'] = np.nan

#split data
mskf = MultilabelStratifiedKFold(n_splits=n_fold, random_state=SEED)
for i, (_, test_index) in enumerate(mskf.split(X, y)):
    train_df.iloc[test_index, -1] = i
idxs = [i for i in range(len(train_df))]
train_df['fold'] = train_df['fold'].astype('int')
train_idx = []
val_idx = []
model = seresnext(nunique, pretrained_model).to(device)

def idx_to_class(idx):
    cls_map = pd.read_csv('./data/class_map.csv')
    cls_map = cls_map[:168]
    return cls_map.iloc[idx]['component']

# for i in T(range(len(train_df))):
#     if train_df.iloc[i]['fold'] == fold: val_idx.append(i)
#     else: train_idx.append(i)
train_idx = idxs[:int((n_fold-1)*len(idxs)/(n_fold))]
val_idx = idxs[int((n_fold-1)*len(idxs)/(n_fold)):]

# valid_ds = GraphemeDataset('val_images_data.parquet', 'data/train.csv', val_aug) 
valid_ds = BanglaDataset(train_df, 'data/128_numpy', val_idx, _type='valid', aug=val_aug, return_name=True)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)



def evaluate(results, pred_thr):
   model.eval()
   total = 0.0
   running_loss = 0.0
   running_acc = 0.0
   grapheme_root_out=0.0
   vowel_diacritic_out=0.0
   consonant_diacritic_out=0.0
   running_acc = 0.0
   pred1= []
   pred2= []
   pred3 = []
   pred1_val= []
   pred2_val= []
   pred3_val = []
   lab1 = []
   lab2 = []
   lab3 = []
   names = []
   with torch.no_grad():
     for idx, (name, inputs, labels1, labels2, labels3) in T(enumerate(valid_loader),total=len(valid_loader)):
        inputs = inputs.to(device)
        labels1 = labels1.to(device)
        labels2 = labels2.to(device)
        labels3 = labels3.to(device)
        total += len(inputs)
        outputs1,outputs2,outputs3 = model(inputs.float())
        names.extend(name)
        pred1.extend(torch.argmax(outputs1, dim=1).cpu().numpy())
        pred2.extend(torch.argmax(outputs2, dim=1).cpu().numpy())
        pred3.extend(torch.argmax(outputs3, dim=1).cpu().numpy())
        
        pred1_val.extend(np.max(nn.functional.softmax(outputs1).cpu().numpy(), axis=1))
        pred2_val.extend(np.max(nn.functional.softmax(outputs2).cpu().numpy(), axis=1))
        pred3_val.extend(np.max(nn.functional.softmax(outputs3).cpu().numpy(), axis=1))

        lab1.extend(labels1.cpu().numpy())
        lab2.extend(labels2.cpu().numpy())
        lab3.extend(labels3.cpu().numpy())

        
        loss1 = 0.7*criterion(outputs1,labels1)
        loss2 = 0.20*criterion(outputs2,labels2)
        loss3 = 0.10*criterion(outputs3,labels3)
        running_loss += loss1.item()+loss2.item()+loss3.item()
        
        grapheme_root_out       += (outputs1.argmax(1)==labels1).float().mean()
        vowel_diacritic_out     += (outputs2.argmax(1)==labels2).float().mean()
        consonant_diacritic_out += (outputs3.argmax(1)==labels3).float().mean()

   recall_graph = sklearn.metrics.recall_score(pred1, lab1, average='macro')
   recall_vowel = sklearn.metrics.recall_score(pred2, lab2, average='macro')
   recall_consonant = sklearn.metrics.recall_score(pred3, lab3, average='macro')
   scores = [recall_graph, recall_vowel, recall_consonant]
   total_recall = np.average(scores, weights=[2, 1, 1])
   msg = 'Loss: {:.4f} \n Acc:     \t Root {:.4f}     \t Vowel {:.4f} \t Consonant {:.4f} \nRecall:  \t Root {:.4f}     \t Vowel {:.4f} \t Consonant {:.4f} Total {:.4f}\n'.format(running_loss/(len(valid_loader)), grapheme_root_out/(len(valid_loader)), vowel_diacritic_out/(len(valid_loader)), consonant_diacritic_out/(len(valid_loader)), recall_graph, recall_vowel, recall_consonant, total_recall)
   print(msg)
   l = 0
   for idx,(i, j) in enumerate(zip(lab1, pred1)):
       if i!=j:
            results.loc[l, 'ID'] = names[idx]
            results.loc[l, 'Graph_Actual'] = idx_to_class(i)
            results.loc[l, 'Graph_Pred'] = idx_to_class(j)
            l+=1
   pred_thr['ID'] = names
   pred_thr['Graph_confidence'] = pred1_val
   pred_thr['Vowel_confidence'] = pred2_val
   pred_thr['Consonant_confidence'] = pred3_val

   pred_thr['Graph_pred'] = pred1
   pred_thr['Vowel_pred'] = pred2
   pred_thr['Consonant_pred'] = pred3

   results.to_csv('results.csv', index=False)
   df = results.groupby('Graph_Actual').size().reset_index(name='Count').rename(columns={'Graph_Actual':'Graph_Actual_value'})
   df.to_csv('results_count.csv', index=False)

   
   return  total_recall, pred_thr

plist = [
        {'params': model.backbone.layer0.parameters(),  'lr': learning_rate/100},
        {'params': model.backbone.layer1.parameters(),  'lr': learning_rate/100},
        {'params': model.backbone.layer2.parameters(),  'lr': learning_rate/100},
        {'params': model.backbone.layer3.parameters(),  'lr': learning_rate/100},
    ]
# optimizer = optim.Adam(plist, lr=learning_rate)
optimizer = Over9000(plist, lr=learning_rate, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, learning_rate, total_steps=None, epochs=n_epochs, steps_per_epoch=5021, pct_start=0.0,
                                   anneal_strategy='cos', cycle_momentum=True,base_momentum=0.85, max_momentum=0.95,  div_factor=100.0)
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=learning_rate/100, max_lr=learning_rate, step_size_up=10000, step_size_down=None, mode='exp_range', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9)
criterion = nn.CrossEntropyLoss()

if load_model:
  tmp = torch.load(weight_file)
  model.load_state_dict(tmp)
  # optimizer.load_state_dict(tmp['optim'])
  print('Model Loaded!')

if apex:
    amp.initialize(model, optimizer, opt_level='O1')

torch.cuda.empty_cache()
print(gc.collect())
valid_recall, pred_thr = evaluate(results, pred_thr)
pred_thr.to_csv('pred_thr.csv', index=False)