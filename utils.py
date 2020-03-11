import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
import albumentations
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as F_alb

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def erode(img, kernel_size=5, iterations = 1, p=0.5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    erosion = cv2.erode(img,kernel,iterations = iterations)
    return erosion

def dilate(img, kernel_size=5, iterations = 1, p=0.5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilation = cv2.erode(img, kernel, iterations = iterations)
    return dilation

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
    
def cutmix(data, targets1, targets2, targets3, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]
    return data, targets

def mixup(data, targets1, targets2, targets3, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)
    targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]

    return data, targets

def cutmix_criterion(preds1,preds2,preds3, targets, criterion='ohem', rate=0.7):
    targets1, targets2,targets3, targets4,targets5, targets6, lam = targets[0], targets[1], targets[2], targets[3], targets[4], targets[5], targets[6]
    if criterion=='ohem':
        criterion = ohem_loss
        return lam * criterion(rate, preds1, targets1) + (1 - lam) * criterion(rate, preds1, targets2), lam * criterion(rate, preds2, targets3) + (1 - lam) * criterion(rate, preds2, targets4), lam * criterion(rate, preds3, targets5) + (1 - lam) * criterion(rate, preds3, targets6)
    else:
        criterion = nn.CrossEntropyLoss(reduction='mean')
        return lam * criterion(preds1, targets1) + (1 - lam) * criterion(preds1, targets2), lam * criterion(preds2, targets3) + (1 - lam) * criterion(preds2, targets4), lam * criterion(preds3, targets5) + (1 - lam) * criterion(preds3, targets6)

def mixup_criterion(preds1,preds2,preds3, targets, criterion='ohem', rate=0.7):
    targets1, targets2,targets3, targets4,targets5, targets6, lam = targets[0], targets[1], targets[2], targets[3], targets[4], targets[5], targets[6]
    if criterion=='ohem':
        criterion = ohem_loss
        return lam * criterion(rate, preds1, targets1) + (1 - lam) * criterion(rate, preds1, targets2), lam * criterion(rate, preds2, targets3) + (1 - lam) * criterion(rate, preds2, targets4), lam * criterion(rate, preds3, targets5) + (1 - lam) * criterion(rate, preds3, targets6)
    else:
        criterion = nn.CrossEntropyLoss(reduction='mean')
        return lam * criterion(preds1, targets1) + (1 - lam) * criterion(preds1, targets2), lam * criterion(preds2, targets3) + (1 - lam) * criterion(preds2, targets4), lam * criterion(preds3, targets5) + (1 - lam) * criterion(preds3, targets6)    


class RandomErasing:
    def __init__(self, p, area_ratio_range, min_aspect_ratio, max_attempt):
        self.p = p
        self.max_attempt = max_attempt
        self.sl, self.sh = area_ratio_range
        self.rl, self.rh = min_aspect_ratio, 1. / min_aspect_ratio

    def __call__(self, image):
        image = np.asarray(image).copy()

        if np.random.random() > self.p:
            return image

        h, w = image.shape[:2]
        image_area = h * w

        for _ in range(self.max_attempt):
            mask_area = np.random.uniform(self.sl, self.sh) * image_area
            aspect_ratio = np.random.uniform(self.rl, self.rh)
            mask_h = int(np.sqrt(mask_area * aspect_ratio))
            mask_w = int(np.sqrt(mask_area / aspect_ratio))

            if mask_w < w and mask_h < h:
                x0 = np.random.randint(0, w - mask_w)
                y0 = np.random.randint(0, h - mask_h)
                x1 = x0 + mask_w
                y1 = y0 + mask_h
                image[y0:y1, x0:x1] = np.random.uniform(0, 1)
                break

        return image  

def ohem_loss( rate, cls_pred, cls_target ):

    batch_size = cls_pred.size(0) 
    ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='none', ignore_index=-1)

    sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
    keep_num = min(sorted_ohem_loss.size()[0], int(batch_size*rate) )
    if keep_num < sorted_ohem_loss.size()[0]:
        keep_idx_cuda = idx[:keep_num]
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
    cls_loss = ohem_cls_loss.sum() / keep_num
    return cls_loss

