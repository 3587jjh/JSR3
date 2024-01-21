import numpy as np
from numpy import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import sys, os

from datasets import register
import utils
from utils import to_pixel_samples, make_coord


@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, augment=None):
        self.dataset = dataset
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        hr, lr = self.dataset[idx] # (3,h,w) tensor, range [0,1]
        
        if self.augment: # when training
            hflip = (random.random() < 0.5) if 'hflip' in self.augment else False
            vflip = (random.random() < 0.5) if 'vflip' in self.augment else False
            dflip = (random.random() < 0.5) if 'dflip' in self.augment else False

            def base_augment(img):
                if hflip:
                    img = img.flip(-2)
                if vflip:
                    img = img.flip(-1)
                if dflip:
                    img = img.transpose(-2, -1)
                return img
            lr = base_augment(lr)
            hr = base_augment(hr)

        h,w = hr.shape[-2:]
        hr_coord = make_coord([h,w], flatten=False)
        cell = torch.tensor([2/h, 2/w], dtype=torch.float32)
        return {
            'lr': lr,
            'coord': hr_coord,
            'cell': cell,
            'hr': hr
        }


@register('sr-implicit-downsampled-train')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size, scale_min, scale_max, augment):
        self.dataset = dataset
        self.inp_size = inp_size
        assert scale_min <= scale_max
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = (inp_size*scale_min)**2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx] # (3,h,w) tensor, range [0,1]
        s = random.uniform(self.scale_min, self.scale_max)
        w_lr = self.inp_size
        w_hr = round(w_lr * s)
        hr = utils.random_crop(img, w_hr)

        # augmentation
        hflip = (random.random() < 0.5) if 'hflip' in self.augment else False
        vflip = (random.random() < 0.5) if 'vflip' in self.augment else False
        dflip = (random.random() < 0.5) if 'dflip' in self.augment else False

        def base_augment(hr):
            if hflip:
                hr = hr.flip(-2)
            if vflip:
                hr = hr.flip(-1)
            if dflip:
                hr = hr.transpose(-2, -1)
            return hr
        hr = base_augment(hr)
        lr = utils.resize_fn(hr, w_lr)
        hr_coord, hr_rgb = to_pixel_samples(hr.contiguous())

        # sample pixels
        sample_lst = random.choice(len(hr_coord), self.sample_q, replace=False)
        hr_coord = hr_coord[sample_lst]
        hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones(2)
        cell[0] *= 2 / hr.shape[-2]
        cell[1] *= 2 / hr.shape[-1]

        return {
            'lr': lr,
            'coord': hr_coord,
            'cell': cell,
            'hr_rgb': hr_rgb
        }


@register('sr-implicit-downsampled-test')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, scale, crop_sz=None, step=None):
        self.dataset = dataset
        self.scale = scale
        self.crop_sz = crop_sz
        self.step = step

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        hr = self.dataset[idx] # (3,h,w) tensor, range [0,1]
        s = self.scale
        h,w = hr.shape[-2:]

        # left-corner crop hr (same with classSR)
        if self.crop_sz and self.step:
            crop_sz, step = self.crop_sz*s, self.step*s
            crop_h = ((h-crop_sz+step)//step-1)*step+crop_sz
            crop_w = ((w-crop_sz+step)//step-1)*step+crop_sz
        else:
            crop_h, crop_w = h//s*s, w//s*s
        hr = hr[:, :crop_h, :crop_w]
        h,w = hr.shape[-2:]
        assert h%s==0 and w%s == 0

        lr = utils.resize_fn(hr, (h//s, w//s))
        hr_coord, hr_rgb = to_pixel_samples(hr.contiguous())

        cell = torch.ones(2)
        cell[0] *= 2 / hr.shape[-2]
        cell[1] *= 2 / hr.shape[-1]

        return {
            'lr': lr,
            'coord': hr_coord,
            'cell': cell,
            'hr_rgb': hr_rgb
        }