from tqdm import tqdm

import torch
import torch.nn as nn
from thop import profile

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
from PIL import Image
import os
import argparse
import random
from natsort import natsorted
import numpy as np
import cv2

import torch.optim
import math

torch.cuda.empty_cache()

from model.drln_dw import DRLN as SuperResolution
from dataset import DIV2K, Flickr2K
from dataset import TestSet
import draw as d
from demo import demo_UHD_fast, psnr_tensor

# random seed
torch.backends.cudnn.benchmark = True
# random seed
seed = 0
if seed is None:
    seed = random.randint(1, 10000)
random.seed(seed)
torch.manual_seed(seed)



parser = argparse.ArgumentParser(description="DRLN_dw")

parser.add_argument("--batch_size", type=int, default=16,
                    help="training batch size")
parser.add_argument("--save_dir", type=str, default="./checkpoint_drlndw",
                    help="saving checkpoint and pictures")
parser.add_argument("--resume", default="", type=str,
                    help="path to checkpoint") 
parser.add_argument("--n_epoch", default=1000, type=int,
                    help="number of epochs to train")
parser.add_argument("--lr", default=0.001, type=float,
                    help="initial learning rate") 

args = parser.parse_args()

save_dir = args.save_dir
#############################################################################################

# train dataset, val dataset, test dataset
batch_size = args.batch_size
DIV2K_dataset = DIV2K('./DIV2K_decoded', patch_size=192, mode='train', repeat = 20)
# Flickr2K_dataset = Flickr2K('./Flickr2K_decoded', patch_size=192, mode='train', repeat=5)
# train_dataset = ConcatDataset([DIV2K_dataset, Flickr2K_dataset])
train_dataloader = DataLoader(DIV2K_dataset, batch_size = batch_size, shuffle=True, pin_memory=True,
                              num_workers=8, drop_last=True)

val_dataset = DIV2K('./DIV2K_decoded', patch_size=240, mode='valid', repeat = 5)
val_dataloader = DataLoader(val_dataset, batch_size = 16, shuffle=False, pin_memory=True,
                              num_workers=8, drop_last=True)

test_ds = TestSet(lq_paths=['Set5/LRbicx3'], gt_paths=['Set5/original'])
test_dl = DataLoader(test_ds, batch_size=1)


#############################################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SuperResolution()
model = model.to(device)
# criterion
criterion = nn.L1Loss() # MSELoss
criterion = criterion.to(device)
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000 * len(train_dataloader), eta_min=0.0003)

#############################################################################################

def train_epoch(epoch):
    model.train()
    train_loss = 0
    lrs = []
    for idx, (img_hr, img_lr) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        
        img_hr = img_hr.to(device)
        img_lr = img_lr.to(device)
        output = model(img_lr)
        loss = criterion(output, img_hr)
        train_loss += loss.item()
        lrs.append(optimizer.param_groups[0]['lr'])
        
        loss.backward()
        optimizer.step()
        if epoch < 1000:
            scheduler.step()
    
    train_loss /= len(train_dataloader)
    return train_loss, lrs

def val_epoch():
    model.eval()
    psnrs = []
    for idx, (img_hr, img_lr) in enumerate(tqdm(val_dataloader)):
        img_hr = img_hr.to(device)
        img_lr = img_lr.to(device)
        with torch.no_grad():
            output = model(img_lr)
            psnr = psnr_tensor(output * 255, img_hr * 255)
            psnrs.append(psnr)
    return torch.tensor(psnrs)


def test_epoch():
    model.eval()
    psnrs = []
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
        
        psnr = psnr_tensor(preds, img_gt)
        psnrs.append(psnr)
    return torch.tensor(psnrs)

def save_checkpoint(epoch, model, optimizer, psnr_best, history, dir=save_dir):
    state = {'epoch': epoch,
             'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'psnr_best': psnr_best,
             'history': history}
    torch.save(state, os.path.join(dir, 'checkpoint.pth'))

result = {'train_loss': float, 'val_psnr': float, 'test_psnr': float, 'lrs': []}
history = []
resume = args.resume


def train(start_epoch, n_epochs):
    global psnr_best
    model.to(device)

    for epoch in range(start_epoch, n_epochs + 1):
        print('Epoch: {}'.format(epoch), 'lr: {}'.format(optimizer.param_groups[0]['lr']))
        train_loss, lrs= train_epoch(epoch)
        print('Train Loss: {}'.format(train_loss))
        
        with torch.no_grad():
            val_psnrs = val_epoch()
            print('Val PSNR: {}'.format(val_psnrs.mean()))
            test_psnrs = test_epoch()
            print('Test PSNR: {}'.format(test_psnrs.mean()))
        
            result['train_loss'] = train_loss
            result['val_psnr'] = val_psnrs.mean()
            result['test_psnr'] = test_psnrs.mean()
            result['lrs'] = lrs
            history.append(result.copy())

        if test_psnrs.mean() > psnr_best:
            psnr_best = test_psnrs.mean()
            torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
            save_checkpoint(epoch, model, optimizer, psnr_best, history)
            print("Model saved!")

        print('Best PSNR: {}'.format(psnr_best))
        
    
        d.plot_lrs(history)
        d.plot_losses(history)
        d.plot_val(history)
        d.plot_test(history)           

    return history




if __name__ == '__main__':
    
    start_epoch = 1
    n_epochs = args.n_epoch
    psnr_best = 0
    if resume:
        checkpoint = torch.load(resume)
        state = torch.load(os.path.join(save_dir, 'model.pth'))
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(state)
        optimizer.load_state_dict(checkpoint['optimizer'])
        psnr_best = checkpoint['psnr_best']
        history = checkpoint['history']

    history = train(start_epoch, n_epochs)
    
    d.plot_lrs(history)
    d.plot_losses(history)
    d.plot_val(history)
    d.plot_test(history)
    

