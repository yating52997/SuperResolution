import torch
import torch.nn as nn
from thop import profile
from tqdm import tqdm
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
from model.drln_dw import DRLN as SuperResolution

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
    
test_ds = TestSet(lq_paths=['Set5/LRbicx3'], gt_paths=['Set5/original'])
test_dl = DataLoader(test_ds, batch_size=1)

def save_pic(pic, height, width, dir):
    pic = pic.squeeze(0).cpu().numpy()
    pic = np.transpose(pic if pic.shape[0] == 1 else pic[[2, 1, 0], :, :], (1, 2, 0))   # CHW-RGB to HWC-BGR
    pic = (pic * 255).astype(np.uint8)
    pic = cv2.resize(pic, (width, height), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(dir, pic)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SuperResolution().to(device)

    state_dict = torch.load('./checkpoint/model.pth', map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    for idx, data in enumerate(tqdm(test_dl)):
        img_name = data[0][0]
        img_lq = data[1].to(device)
        img_gt = data[2].to(device)
        height, width = img_gt.shape[2:]
        with torch.no_grad():
            output = model(img_lq)
            save_pic(output, height, width, f'./pictures/{idx}_pred.png')
            save_pic(img_gt, height, width, f'./pictures/{idx}_gt.png')
            save_pic(img_lq, height, width, f'./pictures/{idx}_lq.png')
        