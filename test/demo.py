# =============================================
# =============================================
# Author: Lyu-Ming Ho
# Contact: holyuming.ee12@nycu.edu.tw
# You can not modify anything in this file
# =============================================
# =============================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import os
import cv2
import numpy as np
from thop import profile
import math
import json

from model import SuperResolution


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

        print(f'Number of LR imgs (patches): {str(len(self.imagelist1)):6s}, Number of HR imgs (patches): {str(len(self.imagelist2)):6s}')


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


def psnr_tensor(img1, img2):
    """
    Args:
        img1: pytorch tensor with size = [batch, 3, H, W]
        img2: pytorch tensor with size = [batch, 3, H, W]
    Returns:
        float: psnr result
    """
    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')

    sqrt_mse = ((img1 - img2)**2).mean().sqrt()
    if sqrt_mse == 0:
        return float('inf')
    return 20 * torch.log10(255. / sqrt_mse)


def demo_UHD_fast(img, model):
    # test the image tile by tile
    # print(img.shape) # [1, 3, 2048, 1152] for ali forward data
    scale = 3
    b, c, h, w = img.size()
    tile = min(256, h, w)
    tile_overlap = 0
    stride = tile - tile_overlap
    
    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
    E = torch.zeros(b, c, h*scale, w*scale).type_as(img)
    W = torch.zeros_like(E)
    
    in_patch = []
    # append all 256x256 patches in a batch with size = 135
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch.append(img[..., h_idx:h_idx+tile, w_idx:w_idx+tile].squeeze(0))

    in_patch = torch.stack(in_patch, 0)
    out_patch = model(in_patch)
    
    for ii, h_idx in enumerate(h_idx_list):
        for jj, w_idx in enumerate(w_idx_list):
            idx = ii * len(w_idx_list) + jj
            out_patch_mask = torch.ones_like(out_patch[idx])

            E[..., h_idx*scale:(h_idx+tile)*scale, w_idx*scale:(w_idx+tile)*scale].add_(out_patch[idx])
            W[..., h_idx*scale:(h_idx+tile)*scale, w_idx*scale:(w_idx+tile)*scale].add_(out_patch_mask)
            
    output = E.div_(W)
    return output




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SuperResolution().to(device)

    state_dict = torch.load('model.pth', map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    test_ds = TestSet(lq_paths=['Set5/LRbicx3'], gt_paths=['Set5/original'])
    test_dl = DataLoader(test_ds, batch_size=1)

    print(f'================================EVALUATION=============================')
    psnrs = []
    losses = []
    for idx, data in enumerate(test_dl):
        img_name = data[0][0]
        img_lq = data[1].to(device)
        img_gt = data[2].to(device)
        with torch.no_grad():
            _, _, h_old, w_old = img_lq.size()
            h_pad = (2 ** math.ceil(math.log2(h_old))) - h_old
            w_pad = (2 ** math.ceil(math.log2(w_old))) - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = demo_UHD_fast(img_lq, model)
            preds = (output[:, :, :h_old*3, :w_old*3].clamp(0, 1) * 255).round()

        img_gt = (img_gt[:, :, :h_old*3, :w_old*3] * 255.).round()
        loss = ((preds - img_gt) ** 2).mean()
        losses.append(loss)
        psnr = psnr_tensor(preds, img_gt)
        psnrs.append(psnr)
        print(f'Tesing: {img_name:20s}, PSNR: {psnr}')
        
    psnrs = torch.tensor(psnrs)
    print(f'\nAverage PSNR: {psnrs.mean():.3f} dB\n')

    img = torch.randn(1, 3, 256, 256).to(device)
    macs, params = profile(model, inputs=(img, ), verbose=False)
    flops = macs * 2 / 1e9  # G
    params = params / 1e6   # M
    print('============================')
    print(f'FLOPs : { flops } G')
    print(f'PARAMS : { params } M ')
    print('============================')
    if flops < 11.1 and psnrs.mean() > 31.5:
        print('Your FLOPs is smaller than 11.1 G.')
        print('You will get this 10 points.')
        print('Congratulations !!!')
    else:
        print('Your FLOPs is larger than 11.1 G.')
        print('You will not get this 10 points.')
    print('============================')
    print(f'Overall Ranking Score: {1 / (flops * params) * math.e**(psnrs.mean()-20):.3f}')
    print('============================')


    with open('ID_result.csv', 'w') as f:
        f.write('Id,Loss\n')
        for i, data in enumerate(losses):
            f.write(str(i) + ',' + str(data.item()) + '\n')

    
    print(f'Your score on Kaggle should be {sum(losses):4f}')
    print('============================')