# modified from: https://github.com/yinboc/liif
import argparse
import os, sys
import numpy as np

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

import datasets
import models
import utils
from utils import make_coord

import warnings
warnings.filterwarnings("ignore")


def load_model():
    resume_path = config['resume_path']
    if os.path.exists(resume_path):
        print('Model resumed from ...', resume_path)
        sv_file = torch.load(resume_path)
        model = models.make(sv_file['model'], load_sd=True).cuda()
        print('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    else:
        print('Failed to load model')
        exit()
    return model


def make_test_loader(): 
    spec = config['test_dataset'].copy()
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    test_loader = DataLoader(dataset, batch_size=1,
        shuffle=False, num_workers=8, pin_memory=True)
    return test_loader


def test(model):
    model.eval()
    test_loader = make_test_loader()
    total_flops = 0
    if phase >= 2:
        total_ratio_easy = 0
    total_patches = 0
    psnrs = []

    scale = config['scale']
    crop_sz = config['patch_size']
    step = config['step']

    batch_size = config['batch_size']
    diff_threshold = config['diff_threshold']

    rgb_mean = torch.tensor(config['data_norm']['mean'], device='cuda').view(1,3,1,1)
    rgb_std = torch.tensor(config['data_norm']['std'], device='cuda').view(1,3,1,1)

    for i, batch in enumerate(tqdm(test_loader, leave=True, desc=f'test (x{scale})')):
        for k, v in batch.items():
            batch[k] = v.cuda()
        
        lr = (batch['lr'] - rgb_mean) / rgb_std
        hr = batch['hr']

        # left corner crop lr, hr to patch-divisible size (for fair comparison)
        h,w = lr.shape[-2:]
        crop_h = ((h-crop_sz+step)//step-1)*step+crop_sz
        crop_w = ((w-crop_sz+step)//step-1)*step+crop_sz
        lr = lr[:,:, :crop_h, :crop_w]
        hr = hr[:,:, :scale*crop_h, :scale*crop_w]

        with torch.no_grad():
            # extract patches (no padding)
            lrs = nn.Unfold(kernel_size=crop_sz, stride=step)(lr) 
            lrs = lrs.transpose(0,2).contiguous().view(-1,3,crop_sz,crop_sz)
            hrs = nn.Unfold(kernel_size=crop_sz*scale, stride=step*scale)(hr) 
            hrs = hrs.transpose(0,2).contiguous().view(-1,3,crop_sz*4,crop_sz*4)
            num_patches = len(lrs)
            total_patches += num_patches

            if config['per_image']:
                # regenerate coord
                coord = make_coord(hr.shape[-2:], flatten=False, device=hr.device).unsqueeze(0)
                cell = batch['cell']
                if phase <= 1:
                    pred = model(lr, coord, cell)
                    total_flops += model.flops(lr.shape[-2:], hr.shape[-2]*hr.shape[-1])
                else:
                    pred, diff, flops, ratio_easy = model(lr, coord, cell, diff_threshold)
                    total_flops += flops
                    total_ratio_easy += ratio_easy * num_patches
                pred = pred * rgb_std + rgb_mean
            else:
                # batched(patch) model prediction
                preds = []
                l = 0
                # regenerate coord, cell
                h,w = hrs.shape[-2:]
                coord = make_coord((h,w), flatten=False, device=hrs.device).unsqueeze(0)
                cell = torch.tensor([2/h, 2/w], device=hrs.device).unsqueeze(0)

                while l < num_patches:
                    r = min(num_patches, l+batch_size)
                    if phase <= 1:
                        pred = model(lrs[l:r], coord.repeat(r-l,1,1,1), cell.repeat(r-l,1))
                        total_flops += model.flops(lrs.shape[-2:], hrs.shape[-2]*hrs.shape[-1]) * (r-l)
                    else:
                        pred, diff, flops, ratio_easy = model(lrs[l:r], coord.repeat(r-l,1,1,1),
                            cell.repeat(r-l,1), diff_threshold)
                        total_flops += flops
                        total_ratio_easy += ratio_easy * (r-l)
                    pred = pred * rgb_std + rgb_mean
                    preds.append(pred)
                    l = r
                preds = torch.cat(preds, dim=0)

                # combine preds
                preds = preds.flatten(1).unsqueeze(-1).transpose(0,2)
                mask = torch.ones_like(preds)
                mask = nn.Fold(output_size=hr.shape[-2:],
                    kernel_size=scale*crop_sz, stride=scale*step)(mask)
                pred = nn.Fold(output_size=hr.shape[-2:],
                    kernel_size=scale*crop_sz, stride=scale*step)(preds)/mask

        pred = utils.tensor2numpy(pred) # (h,w,3), range [0,255] 
        hr = utils.tensor2numpy(hr) # (h,w,3), range [0,255]
        if config.get('psnr_type') == 'rgb':
            psnr = utils.psnr_measure_rgb(pred, hr, shave_border=scale)
        else:
            psnr = utils.psnr_measure(pred, hr, shave_border=scale)
        psnrs.append(psnr)

    psnr = np.mean(np.array(psnrs))
    avg_flops = total_flops / total_patches
    if phase <= 1:
        return psnr, avg_flops
    else:
        avg_ratio_easy = total_ratio_easy / total_patches
        return psnr, avg_flops, avg_ratio_easy


def main(config_):
    global config, phase
    config = config_
    phase = config['phase']

    model = load_model()
    test_data_name = config['test_dataset']['dataset']['args']['root_path_1'].split('/')[2]

    if phase <= 1:
        psnr, flops = test(model)
        print('{} (x{}) | psnr({}): {:.2f} dB | flops (per patch): {:.2f}G'\
            .format(test_data_name, config['scale'], config['psnr_type'], psnr, flops/1e9))
    else:
        psnr, flops, ratio_easy = test(model)
        print('{} (x{}) | psnr({}): {:.2f} dB | flops (per patch): {:.2f}G | easy ratio (per patch): {:.1f} %'\
            .format(test_data_name, config['scale'], config['psnr_type'], psnr, flops/1e9, ratio_easy*100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--per_image', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--diff_threshold', type=float, default=0.5)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('Config loaded ...', args.config)

    if not config.get('psnr_type'):
        config['psnr_type'] = 'y'
    if not config.get('data_norm'):
        config['data_norm']['mean'] = [0.5, 0.5, 0.5]
        config['data_norm']['std'] = [0.5, 0.5, 0.5]

    config['per_image'] = args.per_image
    config['batch_size'] = args.batch_size
    config['diff_threshold'] = args.diff_threshold
    main(config)
