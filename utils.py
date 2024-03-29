# modified from: https://github.com/yinboc/liif
import os, sys
import time
import shutil
import math

import numpy as np
from numpy import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from bicubic_pytorch import core

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.backends.cudnn as cudnn
#from tensorboardX import SummaryWriter
import imageio
from PIL import Image


def center_crop(img, size):
    h,w = img.shape[-2:]
    cut_h, cut_w = h-size[0], w-size[1]
    
    lh = cut_h // 2
    rh = h - (cut_h - lh)
    lw = cut_w // 2
    rw = w - (cut_w - lw)
    
    img = img[:,:, lh:rh, lw:rw]
    return img


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    os.environ["PYTHONHASHSEED"] = str(seed)
    cudnn.benchmark = False # for real fix, but slower training
    cudnn.deterministic = True # for real fix, but slower training

    
def make_coord(shape, ranges=None, flatten=True, device='cpu'):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n, device=device).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples(img, device='cpu'):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    if img.ndim == 3:
        coord = make_coord(img.shape[-2:], device=device)
        rgb = img.view(3, -1).permute(1, 0)
    else:
        b = img.shape[0]
        coord = make_coord(img.shape[-2:], device=device)
        rgb = img.view(b, 3, -1).permute(0, 2, 1)
    return coord, rgb


def random_crop(img, size):
    # img: tensor (3,h,w), range independent
    x0 = random.randint(0, img.shape[-2]-size+1)
    y0 = random.randint(0, img.shape[-1]-size+1)
    return img[:, x0: x0 + size, y0: y0 + size]


def resize_fn(img, size):
    # img: tensor (c,h,w) or (b,c,h,w), range independent
    if isinstance(size, int):
        size = (size, size)
    return core.imresize(img, sizes=size)


def tensor2numpy(tensor, rgb_range=1.):
    rgb_coefficient = 255 / rgb_range
    img = tensor.mul(rgb_coefficient).clamp(0, 255).round()
    img = img[0].data
    img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    return img


def numpy2tensor(img, rgb_range=1., device='cpu'):
    img = np.array(img).astype('float64')
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # HWC -> CHW
    tensor = torch.from_numpy(np_transpose).float().to(device)  # numpy -> tensor
    tensor.mul_(rgb_range / 255)  # (0,255) -> (0,1)
    tensor = tensor.unsqueeze(0)
    return tensor


def pad_img(x, size_must_mode=8):
    b,c,h,w = x.size()
    pw = (size_must_mode - (w % size_must_mode)) % size_must_mode
    ph = (size_must_mode - (h % size_must_mode)) % size_must_mode
    pl = pw // 2
    pr = pw - pl
    pu = ph // 2
    pd = ph - pu
    pad = [pl, pr, pu, pd]
    x = F.pad(x, pad=pad, mode='replicate')
    return x, pad

def save_img(img, path, denorm=False):
    B, C,H,W = img.size()
    if denorm:
        img = (img+1)/2.
    if C is not 3:
        img = img.repeat(1,3,1,1)
    img = img[0]
    img = img.permute(1,2,0)
    img = img.cpu().data.numpy()
    img = (np.clip(img, 0,1)*255).astype(np.uint8)
    Image.fromarray(img).save(path)

def save_img_np(img, path):
    img = img.astype(np.uint8)
    Image.fromarray(img).save(path)


def rgb2ycbcr(img, maxVal=255):
    O = np.array([[16],
                [128],
                [128]])
    T = np.array([[0.256788235294118, 0.504129411764706, 0.097905882352941],
                [-0.148223529411765, -0.290992156862745, 0.439215686274510],
                [0.439215686274510, -0.367788235294118, -0.071427450980392]])

    if maxVal == 1:
        O = O / 255.0

    img_s = img.shape
    if len(img_s) >= 3:
        t = np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))
    else:
        t = img
    t = np.dot(t, np.transpose(T))
    t[:, 0] += O[0]
    t[:, 1] += O[1]
    t[:, 2] += O[2]
    if len(img_s) >= 3:
        ycbcr = np.reshape(t, [img.shape[0], img.shape[1], img.shape[2]])
    else:
        ycbcr = t
    return ycbcr


def psnr_measure(src, tar, shave_border=0):
    # Input must be 0-255, (h,w,3)
    def psnr(y_true,y_pred, shave_border=4):
        # Input must be 0-255, 2D
        target_data = np.array(y_true, dtype=np.float32)
        ref_data = np.array(y_pred, dtype=np.float32)

        diff = ref_data - target_data
        if shave_border > 0:
            diff = diff[shave_border:-shave_border, shave_border:-shave_border]
        rmse = np.sqrt(np.mean(np.power(diff, 2)))

        if rmse < 0.6502:
             return 50
        else:
            return 20 * np.log10(255./rmse)
    return psnr(rgb2ycbcr((src).astype(np.uint8))[:,:,0],\
        rgb2ycbcr((tar).astype(np.uint8))[:,:,0], shave_border=shave_border)
    

def psnr_measure_rgb(src, tar, shave_border=0):
    # Input must be 0-255, (h,w,3)
    target_data = np.array(src, dtype=np.float32)
    ref_data = np.array(tar, dtype=np.float32)

    diff = ref_data - target_data
    if shave_border > 0:
        diff = diff[shave_border:-shave_border, shave_border:-shave_border]
    err = np.mean(np.power(diff, 2))
    if err < 0.6502:
        return 50
    else:
        return 10 * np.log10((255. ** 2) / err)


class Averager():
    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():
    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    #writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    #return log, writer
    return log


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def make_lr_scheduler(optimizer, lr_scheduler_spec, load_sd=False):
    Scheduler = {
        'CosineAnnealingLR': CosineAnnealingLR
    }[lr_scheduler_spec['name']]

    lr_scheduler = Scheduler(optimizer, **lr_scheduler_spec['args'])
    if load_sd:
        lr_scheduler.load_state_dict(lr_scheduler_spec['sd'])
    return lr_scheduler