import matplotlib.pyplot as plt
import numpy as np
import os
from train import save_dir

def plot_lrs(history):
  plt.cla()
  lrs = np.concatenate([x.get('lrs', []) for x in history])
  plt.plot(lrs)
  plt.xlabel('Batch no.')
  plt.ylabel('Learning rate')
  plt.title('Learning Rate vs. Batch no.')
  plt.savefig(os.path.join(save_dir, 'lrs.png'))



def plot_losses(history):
  plt.cla()
  train_losses = ([x.get('train_loss') for x in history])
  plt.plot(train_losses, '-bx')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend(['Training'])
  plt.title('Loss vs. No. of epochs')
  plt.savefig(os.path.join(save_dir, 'losses.png'))


def plot_val(history):
  plt.cla()
  val_p = ([x.get('val_psnr') for x in history])
  plt.plot(val_p, '-bx')
  plt.xlabel('epoch')
  plt.ylabel('psnr')
  plt.legend(['validation'])
  plt.title('psnr vs. No. of epochs')
  plt.savefig(os.path.join(save_dir, 'psnr.png'))

def plot_test(history):
  plt.cla()
  test_p = ([x.get('test_psnr') for x in history])
  plt.plot(test_p, '-bx')
  plt.xlabel('epoch')
  plt.ylabel('psnr')
  plt.legend(['Set5'])
  plt.title('psnr vs. No. of epochs')
  plt.savefig(os.path.join(save_dir, 'set5.png'))
