import os
import shutil
import sys
import threading
import curses 
import gc
import time
from random import choices
from itertools import chain
import numpy as np
import pandas as pd
import sklearn
import cv2
from tqdm import tqdm as T
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from apex import amp
import torch, torchvision
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from BanglaDataset import BanglaDataset
from utils import *
from metrics import *
from optimizers import Over9000
from augmentations.augmix import RandomAugMix
from augmentations.gridmask import GridMask
from model.seresnext import seresnext
from model.effnet import EfficientNetWrapper
from model.densenet import *
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
n_fold = 20
fold = 0
SEED = 24
batch_size = 32
sz = 128
learning_rate = 7.5e-4
patience = 5
opts = ['normal', 'mixup', 'cutmix']
device = 'cuda:0'
apex = False
pretrained_model = 'se_resnext101_32x4d'
# pretrained_model = 'densenet121'
# pretrained_model = 'efficientnet-b4'
model_name = '{}_trial_stage1_fold_{}'.format(pretrained_model, fold)
model_dir = 'model_dir'
history_dir = 'history_dir'
tb_dir = 'runs_seresnext'
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
load_model = False
history = pd.DataFrame()
prev_epoch_num = 0
n_epochs = 210
valid_recall = 0.0
best_valid_recall = 0.0
best_valid_loss = np.inf
np.random.seed(SEED)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(history_dir, exist_ok=True)

if os.path.exists(tb_dir):
  try:
    shutil.rmtree(tb_dir)
  except OSError as e:
    print("Error: {} : {}".format(tb_dir, e.strerror))

def launchTensorBoard():
  os.system('tensorboard --logdir ./ --port 9999 --host 0.0.0.0')
  return 


try:
  t = threading.Thread(target=launchTensorBoard, args=([]))
  t.start()
except:
  pass

writer = SummaryWriter(tb_dir)

train_aug =Compose([
  ShiftScaleRotate(p=0.9,border_mode= cv2.BORDER_CONSTANT, value=[0, 0, 0], scale_limit=0.25),
    OneOf([
    Cutout(p=0.3, max_h_size=sz//16, max_w_size=sz//16, num_holes=10, fill_value=0),
    GridMask(num_grid=7, p=0.7, fill_value=0)
    ], p=0.20),
    RandomAugMix(severity=1, width=1, alpha=1., p=0.05),
    # OneOf([
    #     ElasticTransform(p=0.1, alpha=1, sigma=50, alpha_affine=30,border_mode=cv2.BORDER_CONSTANT,value =0),
    #     GridDistortion(distort_limit =0.05 ,border_mode=cv2.BORDER_CONSTANT,value =0, p=0.1),
    #     OpticalDistortion(p=0.1, distort_limit= 0.05, shift_limit=0.2,border_mode=cv2.BORDER_CONSTANT,value =0)                  
    #     ], p=0.3),
    OneOf([
        GaussNoise(var_limit=0.01),
        Blur(),
        GaussianBlur(blur_limit=3),
        RandomGamma(p=0.8),
        ], p=0.4)
    # Normalize()
    ]
      )
# Normalize([0.0692], [0.2051])]
# val_aug = Compose([Normalize([0.0692], [0.2051])])
val_aug = Compose([Normalize()])
train_df = pd.read_csv('data/train.csv')
# train_pseudo_df = pd.read_csv('data/train_and_pseudo.csv')
nunique = list(train_df.nunique())[1:-1]
train_df['id'] = train_df['image_id'].apply(lambda x: int(x.split('_')[1]))
# train_pseudo_df['id'] = train_pseudo_df['image_id'].apply(lambda x: int(x.split('_')[1]))
X, y = train_df[['id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values[:,0], train_df.values[:,1:]
train_df['fold'] = np.nan
train_df= train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
#split data
mskf = MultilabelStratifiedKFold(n_splits=n_fold, random_state=SEED)
for i, (_, test_index) in enumerate(mskf.split(X, y)):
    train_df.iloc[test_index, -1] = i
    
train_df['fold'] = train_df['fold'].astype('int')
idxs = [i for i in range(len(train_df))]
train_idx = []
val_idx = []
model = seresnext(nunique, pretrained_model).to(device)
# model = Dnet(nunique).to(device)
# model = EfficientNetWrapper(pretrained_model).to(device)
# print(summary(model, (3, 128, 128)))
writer.add_graph(model, torch.FloatTensor(np.random.randn(1, 1, 137, 236)).cuda())
# writer.close()

# For stratified split
for i in T(range(len(train_df))):
    if train_df.iloc[i]['fold'] == fold: val_idx.append(i)
    else: train_idx.append(i)

# train_idx = idxs[:int((n_fold-1)*len(idxs)/(n_fold))]
# train_idx = np.load('train_pseudo_idxs.npy')
# val_idx = idxs[int((n_fold-1)*len(idxs)/(n_fold)):]

train_ds = BanglaDataset(train_df, 'data/numpy_format', train_idx, aug=train_aug)
train_loader = DataLoader(train_ds,batch_size=batch_size, shuffle=True)

valid_ds = BanglaDataset(train_df, 'data/numpy_format', val_idx, aug=None)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)

writer = SummaryWriter(tb_dir)
## This function for train is copied from @hanjoonchoe
## We are going to train and track accuracy and then evaluate and track validation accuracy

def train(epoch,history):
  t1 = time.time()
  model.train()
  losses = []
  accs = []
  acc= 0.0
  total = 0.0
  running_loss = 0.0
  grapheme_root_out=0.0
  vowel_diacritic_out=0.0
  consonant_diacritic_out=0.0
  running_acc = 0.0
  running_recall = 0.0
  rate = 1
  
  if epoch<30:
    rate = 1
  elif epoch>=30 and rate>0.65:
    rate = np.exp(-(epoch-30)/60)
  else:
    rate = 0.65
  for idx, (inputs,labels1,labels2,labels3) in enumerate(train_loader):
    inputs = inputs.to(device)
    labels1 = labels1.to(device)
    labels2 = labels2.to(device)
    labels3 = labels3.to(device)
    total += len(inputs)
    choice = choices(opts, weights=[0.20, 0.30, 0.50])
    # print(torch.max())
    # denormalize = UnNormalize(*imagenet_stats)
    # print(torch.max(denormalize(inputs)))
    writer.add_images('my_image', inputs, 0)
    optimizer.zero_grad()
    if choice[0] == 'normal':
      outputs1,outputs2,outputs3 = model(inputs.float())
      loss1 = 0.7*criterion(outputs1,labels1)
      loss2 = 0.20* criterion(outputs2,labels2)
      loss3 = 0.10*criterion(outputs3,labels3)
      loss = loss1 + loss2 + loss3
      running_loss += loss.item()
    
    elif choice[0] == 'mixup':
      inputs, targets = mixup(inputs, labels1, labels2, labels3, np.random.uniform(0.8, 1.0))
      outputs1, outputs2, outputs3 = model(inputs.float())
      loss1, loss2, loss3 = mixup_criterion(outputs1,outputs2,outputs3, targets, rate=rate)
      loss = 0.7*loss1 + 0.20*loss2 + 0.10*loss3
      running_loss += loss.item()
    
    elif choice[0] == 'cutmix':
      inputs, targets = cutmix(inputs, labels1, labels2, labels3, np.random.uniform(0.8, 1.0))
      outputs1, outputs2, outputs3 = model(inputs.float())
      loss1, loss2, loss3 = cutmix_criterion(outputs1,outputs2,outputs3, targets, rate=rate)
      loss = 0.7*loss1 + 0.20*loss2 + 0.10*loss3
      running_loss += loss.item()

    grapheme_root_out       += (outputs1.argmax(1)==labels1).float().mean()
    vowel_diacritic_out     += (outputs2.argmax(1)==labels2).float().mean()
    consonant_diacritic_out += (outputs3.argmax(1)==labels3).float().mean()
  
    if apex:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
          loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    acc = running_acc/total
    # scheduler.step()
    elapsed = int(time.time() - t1)
    eta = int(elapsed / (idx+1) * (len(train_loader)-(idx+1)))
    lr = None
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    writer.add_scalar('Learning Rate', lr, epoch*len(train_loader)+idx)
    writer.add_scalar('OHEM Rate', rate, epoch)
    writer.add_scalar('Loss/train', running_loss/(idx+1), epoch*len(train_loader)+idx)
    writer.add_scalar('Train Accuracy/Root', grapheme_root_out/(idx+1), epoch*len(train_loader)+idx)
    writer.add_scalar('Train Accuracy/Vowel', vowel_diacritic_out/(idx+1), epoch*len(train_loader)+idx)
    writer.add_scalar('Train Accuracy/Consonant', consonant_diacritic_out/(idx+1), epoch*len(train_loader)+idx)
    if idx%1==0:
      msg = 'Epoch: {} \t Progress: {}/{} \t Loss: {:.4f} \t Time: {}s \t ETA: {}s'.format(epoch, 
      idx, len(train_loader), running_loss/(idx+1), elapsed, eta)
      print(msg, end='\r')
      # \nAcc:     \t Root {:.4f}     \t Vowel {:.4f} \t Consonant {:.4f}
      # , grapheme_root_out/(idx+1), vowel_diacritic_out/(idx+1), consonant_diacritic_out/(idx+1)
      # stdscr.addstr(0, 0, msg)
      # stdscr.refresh()
  
  losses.append(running_loss/len(train_loader))
  # accs.append(running_acc/(len(train_loader)*3))
  
  total_train_recall = running_recall/len(train_loader)
  torch.cuda.empty_cache()
  gc.collect()
  history.loc[epoch, 'train_loss'] = losses[0]
  history.loc[epoch, 'Time'] = elapsed
  
  return  total_train_recall


def evaluate(epoch,history):
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
   lab1 = []
   lab2 = []
   lab3 = []
   with torch.no_grad():
     for idx, (inputs,labels1,labels2,labels3) in T(enumerate(valid_loader),total=len(valid_loader)):
        inputs = inputs.to(device)
        labels1 = labels1.to(device)
        labels2 = labels2.to(device)
        labels3 = labels3.to(device)
        total += len(inputs)
        outputs1,outputs2,outputs3 = model(inputs.float())
        pred1.extend(torch.argmax(outputs1, dim=1).cpu().numpy())
        pred2.extend(torch.argmax(outputs2, dim=1).cpu().numpy())
        pred3.extend(torch.argmax(outputs3, dim=1).cpu().numpy())
        
        lab1.extend(labels1.cpu().numpy())
        lab2.extend(labels2.cpu().numpy())
        lab3.extend(labels3.cpu().numpy())

        
        loss1 = 0.70*criterion(outputs1,labels1)
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
   writer.add_scalar('Loss/val', running_loss/(len(valid_loader)), epoch)
   writer.add_scalar('Val Accuracy/Root', grapheme_root_out/(len(valid_loader)), epoch)
   writer.add_scalar('Val Accuracy/Vowel', vowel_diacritic_out/(len(valid_loader)), epoch)
   writer.add_scalar('Val Accuracy/Consonant', consonant_diacritic_out/(len(valid_loader)), epoch)

   writer.add_scalar('Val Recall/Root', recall_graph, epoch)
   writer.add_scalar('Val Recall/Vowel', recall_vowel, epoch)
   writer.add_scalar('Val Recall/Consonant', recall_consonant, epoch)
   writer.add_scalar('Val Recall/Total', total_recall, epoch)

   msg = 'Loss: {:.4f} \n Acc:     \t Root {:.4f}     \t Vowel {:.4f} \t Consonant {:.4f} \nRecall:  \t Root {:.4f}     \t Vowel {:.4f} \t Consonant {:.4f} Total {:.4f}\n'.format(running_loss/(len(valid_loader)), grapheme_root_out/(len(valid_loader)), vowel_diacritic_out/(len(valid_loader)), consonant_diacritic_out/(len(valid_loader)), recall_graph, recall_vowel, recall_consonant, total_recall)
   print(msg)
   lr_reduce_scheduler.step(running_loss)
   history.loc[epoch, 'valid_loss'] = running_loss/(len(valid_loader))
   history.loc[epoch, 'valid_grapheme_recall'] = recall_graph
   history.loc[epoch, 'valid_vowel_recall'] =  recall_vowel
   history.loc[epoch, 'valid_conso_recall'] = recall_consonant
   history.loc[epoch, 'valid_recall'] = total_recall
   history.to_csv(os.path.join(history_dir, 'history_{}.csv'.format(pretrained_model)), index=False)
   return  running_loss/(len(valid_loader)), total_recall

plist = [
        {'params': model.backbone.layer0.parameters(),  'lr': learning_rate/50},
        {'params': model.backbone.layer1.parameters(),  'lr': learning_rate/50},
        {'params': model.backbone.layer2.parameters(),  'lr': learning_rate/50},
        {'params': model.backbone.layer3.parameters(),  'lr': learning_rate/50},
        {'params': model.backbone.layer4.parameters(),  'lr': learning_rate/50}
    ]
# plist = [
#   {"params": model.head1.parameters(), "lr": learning_rate},
#   {"params": model.head2.parameters(), "lr": learning_rate},
#   {"params": model.head3.parameters(), "lr": learning_rate},
#   # {"params": model.backbone.extract_features.parameters(), "lr": learning_rate/100}
# ]
# optimizer = Over9000(plist, lr=learning_rate, weight_decay=1e-3)
optimizer = optim.Adam(plist, lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, learning_rate, total_steps=None, epochs=n_epochs, steps_per_epoch=3348, pct_start=0.0,
                                  #  anneal_strategy='cos', cycle_momentum=True,base_momentum=0.85, max_momentum=0.95,  div_factor=100.0)
lr_reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08)
criterion = nn.CrossEntropyLoss()

if load_model:
  tmp = torch.load(model_name+'_rec.pth')
  model.load_state_dict(tmp['model'])
  # optimizer.load_state_dict(tmp['optim'])
  best_valid_recall = tmp['best_recall']
  prev_epoch_num = tmp['epoch']
  best_valid_loss = tmp['loss']
  del tmp
  print('Model Loaded!')

if apex:
    amp.initialize(model, optimizer, opt_level='O1')

for epoch in range(prev_epoch_num, n_epochs):
    torch.cuda.empty_cache()
    print(gc.collect())
    # stdscr = curses.initscr()
    train_recall = train(epoch,history)
    valid_loss, valid_recall = evaluate(epoch,history)
    if valid_recall > best_valid_recall:
        print(f'Validation recall has increased from:  {best_valid_recall:.4f} to: {valid_recall:.4f}. Saving checkpoint')
        best_state = {'model': model.state_dict(), 'optim': optimizer.state_dict(), 'scheduler': lr_reduce_scheduler.state_dict(), 'loss':valid_loss, 'best_recall':valid_recall, 'epoch':epoch}
        # torch.save(best_state, model_name+'.pth')
        torch.save(best_state, os.path.join(model_dir, model_name+'_rec.pth'))
        torch.save(model.state_dict(), os.path.join(model_dir, '{}_model_weights_best_recall.pth'.format(model_name))) ## Saving model weights based on best validation accuracy.
        best_valid_recall = valid_recall ## Set the new validation Recall score to compare with next epoch
    if valid_loss<best_valid_loss:
        print(f'Validation loss has decreased from:  {best_valid_loss:.4f} to: {valid_loss:.4f}. Saving checkpoint')
        best_state = {'model': model.state_dict(), 'optim': optimizer.state_dict(), 'scheduler': lr_reduce_scheduler.state_dict(), 'recall':valid_recall, 'best_loss':valid_loss, 'epoch':epoch}
        torch.save(best_state, os.path.join(model_dir, model_name+'_loss.pth'))
        torch.save(model.state_dict(), os.path.join(model_dir, '{}_model_weights_best_loss.pth'.format(model_name))) ## Saving model weights based on best validation accuracy.
        best_valid_loss = valid_loss ## Set the new validation Recall score to compare with next epoch