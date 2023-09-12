import torch
import torch.nn as nn
from thop import profile

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
from PIL import Image
import os
import random
from natsort import natsorted
import numpy as np
import cv2

import torch.optim
import glob
torch.cuda.empty_cache()

# random seed
torch.backends.cudnn.benchmark = True
# random seed
seed = 0
if seed is None:
    seed = random.randint(1, 10000)
random.seed(seed)
torch.manual_seed(seed)


class DIV2K(Dataset):
    def __init__(self, root, patch_size = 192, mode = "train", repeat = 1):
        super(DIV2K, self).__init__()
        self.root = root
        self.mode = mode
        self.repeat = repeat
        self.images_hr = natsorted(os.listdir(os.path.join(self.root, 'DIV2K_{}_HR'.format(self.mode))))
        self.images_lr = natsorted(os.listdir(os.path.join(self.root, 
                                                           'DIV2K_{}_LR_bicubic'.format(self.mode), 'X3')))
        self.hr_list = []
        self.lr_list = []
        for i in self.images_hr:
            self.hr_list.append(os.path.join(self.root, 'DIV2K_{}_HR'.format(self.mode), i))
        for i in self.images_lr:
            self.lr_list.append(os.path.join(self.root, 'DIV2K_{}_LR_bicubic'.format(self.mode), 'X3', i))
        self.patch_size = patch_size

    def __getitem__(self, idx):
        img_hr, img_lr = self.load_file(idx)
        img_hr, img_lr = self.get_patches(img_hr, img_lr)
        img_hr, img_lr = self.toTensor(img_hr, img_lr)
        return img_hr, img_lr
    
    def __len__(self):
        return len(self.images_hr) * self.repeat
    
    def load_file(self, idx):
        idx = idx % len(self.images_hr)
        img_hr = np.load(self.hr_list[idx])
        img_lr = np.load(self.lr_list[idx])
        return img_hr, img_lr
    
    def get_patches(self, img_hr, img_lr):
        patch_size = self.patch_size
        scale = 3
        
        # t for target, i for input
        tp = patch_size
        ip = tp // scale
        ih, iw, _ = img_lr.shape

        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)
        tx, ty = scale * ix, scale * iy

        img_lr = img_lr[iy:iy + ip, ix:ix + ip, :]
        img_hr = img_hr[ty:ty + tp, tx:tx + tp, :]
        return img_hr, img_lr
    
    def toTensor(self, img_hr, img_lr):
        img_hr = np.ascontiguousarray(img_hr.transpose(2, 0, 1))
        img_hr = torch.from_numpy(img_hr).float() / 255.

        img_lr = np.ascontiguousarray(img_lr.transpose(2, 0, 1))
        img_lr = torch.from_numpy(img_lr).float() / 255.
        return img_hr, img_lr

class Flickr2K(Dataset):
    def __init__(self, root, patch_size = 192, mode = "train", repeat = 1):
        super(Flickr2K, self).__init__()
        self.root = root
        self.mode = mode
        self.repeat = repeat
        self.images_hr = natsorted(os.listdir(os.path.join(self.root, 'Flickr2K_HR')))
        self.images_lr = natsorted(os.listdir(os.path.join(self.root, 
                                                           'Flickr2K_LR_bicubic', 'X3')))
        self.hr_list = []
        self.lr_list = []
        for i in self.images_hr:
            self.hr_list.append(os.path.join(self.root, 'Flickr2K_HR', i))
        for i in self.images_lr:
            self.lr_list.append(os.path.join(self.root, 'Flickr2K_LR_bicubic', 'X3', i))
        self.patch_size = patch_size

    def __getitem__(self, idx):
        img_hr, img_lr = self.load_file(idx)
        img_hr, img_lr = self.get_patches(img_hr, img_lr)
        img_hr, img_lr = self.toTensor(img_hr, img_lr)
        return img_hr, img_lr
    
    def __len__(self):
        return len(self.images_hr) * self.repeat
    
    def load_file(self, idx):
        idx = idx % len(self.images_hr)
        img_hr = np.load(self.hr_list[idx])
        img_lr = np.load(self.lr_list[idx])
        return img_hr, img_lr
    
    def get_patches(self, img_hr, img_lr):
        patch_size = self.patch_size
        scale = 3
        
        # t for target, i for input
        tp = patch_size
        ip = tp // scale
        ih, iw, _ = img_lr.shape

        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)
        tx, ty = scale * ix, scale * iy

        img_lr = img_lr[iy:iy + ip, ix:ix + ip, :]
        img_hr = img_hr[ty:ty + tp, tx:tx + tp, :]
        return img_hr, img_lr
    
    def toTensor(self, img_hr, img_lr):
        img_hr = np.ascontiguousarray(img_hr.transpose(2, 0, 1))
        img_hr = torch.from_numpy(img_hr).float() / 255.

        img_lr = np.ascontiguousarray(img_lr.transpose(2, 0, 1))
        img_lr = torch.from_numpy(img_lr).float() / 255.
        return img_hr, img_lr



class TestSet(Dataset):
    def __init__(self, lq_paths, gt_paths) -> None:
        super().__init__()
        self.lq_paths   = lq_paths
        self.gt_paths   = gt_paths

        self.imagelist1 = []
        self.imagelist2 = []
        for lq_path in lq_paths:
            self.imagelist1 = self.imagelist1 + glob.glob(os.path.join(lq_path, '*'))
        for gt_path in gt_paths:
            self.imagelist2 = self.imagelist2 + glob.glob(os.path.join(gt_path, '*'))

        # =============== Need to add these two lines when training at TWCC ==============
        self.imagelist1 = sorted(self.imagelist1)
        self.imagelist2 = sorted(self.imagelist2)
        # ================================================================================
        assert (len(self.imagelist1) == len(self.imagelist2))
        for idx in range(len(self.imagelist1)):
            path1 = os.path.basename(self.imagelist1[idx]).replace('x3', '')
            path2 = os.path.basename(self.imagelist2[idx])        
            assert path1 == path2, f'{path1} not match {path2}.'

        # print(f'Number of LR imgs (patches): {str(len(self.imagelist1)):6s},
        #               Number of HR imgs (patches): {str(len(self.imagelist2)):6s}')


    def __getitem__(self, index):
        img_lq = cv2.imread(self.imagelist1[index], cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img_gt = cv2.imread(self.imagelist2[index], cv2.IMREAD_COLOR).astype(np.float32) / 255.

        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))   # HWC-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float()

        img_gt = np.transpose(img_gt if img_gt.shape[2] == 1 else img_gt[:, :, [2, 1, 0]], (2, 0, 1))   # HWC-BGR to CHW-RGB
        img_gt = torch.from_numpy(img_gt).float()

        img_name = os.path.basename(self.imagelist1[index])
        return img_name, img_lq, img_gt


    def __len__(self):
        return len(self.imagelist1)

