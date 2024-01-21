import argparse
import os, sys
import numpy as np

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader

import datasets
import models
import utils

import warnings
warnings.filterwarnings("ignore")


def make_train_loader():
    spec = config['train_dataset'].copy()
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    train_loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=True, num_workers=8, pin_memory=True)
    return train_loader


def make_valid_loader(vd_name=None):
    spec = config['valid_dataset'].copy()
    if vd_name:
        spec['dataset']['args']['root_path_1'] =\
            os.path.join('../datasets', vd_name, 'HR/x4')
        spec['dataset']['args']['root_path_2'] =\
            os.path.join('../datasets', vd_name, 'LR/x4')
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    valid_loader = DataLoader(dataset, batch_size=1,
        shuffle=False, num_workers=8, pin_memory=True)
    return valid_loader


def prepare_training():
    if config['resume'] and os.path.exists(config.get('resume_path')):
        log('Model resumed from ... {}'.format(config['resume_path']))
        sv_file = torch.load(config['resume_path'])        
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        lr_scheduler = utils.make_lr_scheduler(
            optimizer, sv_file['lr_scheduler'], load_sd=True)
        iter_start = sv_file['iter'] + 1

    else:
        log('Loading new model ...')
        model = models.make(config['model']).cuda()
        if phase > 0:
            sv_file = torch.load(config['init_path'])
            init_model = models.make(sv_file['model'], load_sd=True).cuda()
            if phase == 1: 
                log('[encoder] [heavy liif] Init from ... {}'.format(config['init_path']))
                model.encoder = init_model.encoder
                model.heavy_sampler = init_model.heavy_sampler
            elif phase == 2:
                log('[encoder] [light liif] [heavy liif] Init from ... {}'.format(config['init_path']))
                model.encoder = init_model.encoder
                model.light_sampler = init_model.light_sampler
                model.heavy_sampler = init_model.heavy_sampler
            else:
                log('[all modules] Init from ... {}'.format(config['init_path']))
                model = init_model
        optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
        lr_scheduler = utils.make_lr_scheduler(optimizer, config['lr_scheduler'])
        iter_start = 1

    for param in model.parameters():
        param.requires_grad = True
    if phase == 2:
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.light_sampler.parameters():
            param.requires_grad = False
        for param in model.heavy_sampler.parameters():
            param.requires_grad = False
    elif phase > 0:
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.heavy_sampler.parameters():
            param.requires_grad = False
    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))

    # load reference model
    model_ref = None
    if phase >= 2:
        log('Model_ref resumed from ... {}'.format(config['refer_path']))
        sv_file = torch.load(config['refer_path'])
        model_ref = models.make(sv_file['model'], load_sd=True).cuda()
        log('model_ref: #params={}'.format(utils.compute_num_params(model_ref, text=True)))
        model_ref.eval()
    return model, model_ref, optimizer, lr_scheduler, iter_start


def valid(model, model_ref=None, vd_name=None, diff_threshold=0.5):
    model.eval()
    valid_loader = make_valid_loader(vd_name)
    psnrs = []
    scale = config['scale']
    total_flops = 0
    if phase >= 2:
        total_ratio_easy = 0
        total_label_acc = 0

    rgb_mean = torch.tensor(config['data_norm']['mean'], device='cuda').view(1,3,1,1)
    rgb_std = torch.tensor(config['data_norm']['std'], device='cuda').view(1,3,1,1)

    for i, batch in enumerate(tqdm(valid_loader, leave=True, desc=f'valid (x{scale})')):
        for k, v in batch.items():
            batch[k] = v.cuda()

        lr = (batch['lr'] - rgb_mean) / rgb_std
        hr = batch['hr']

        with torch.no_grad():
            if phase <= 1:
                pred = model(lr, batch['coord'], batch['cell'])
                total_flops += model.flops(lr.shape[-2:], hr.shape[-2]*hr.shape[-1])
            else:
                pred, diff, flops, ratio_easy = model(lr, batch['coord'], batch['cell'], diff_threshold)
                total_flops += flops
                total_ratio_easy += ratio_easy

                # evaluation between diff and label
                pred_light, pred_heavy = model_ref(lr, batch['coord'], batch['cell'], both=True)
                pred_light = pred_light * rgb_std + rgb_mean
                pred_heavy = pred_heavy * rgb_std + rgb_mean

                mae_light = torch.abs(pred_light - hr).mean(dim=1)
                mae_heavy = torch.abs(pred_heavy - hr).mean(dim=1)
                label = torch.where(mae_light < mae_heavy, 0, 1) # (b,h,w)

                diff_easy, diff_hard = diff[:,0,:,:], diff[:,1,:,:]
                acc = (((diff_easy > diff_hard) & (label == 0)).sum()\
                    + ((diff_easy <= diff_hard) & (label == 1)).sum())/diff.numel()
                total_label_acc += acc
            pred = pred * rgb_std + rgb_mean

        pred = utils.tensor2numpy(pred) # (h,w,3), range [0,255] 
        hr = utils.tensor2numpy(hr) # (h,w,3), range [0,255]

        if config.get('psnr_type') == 'rgb':
            psnr = utils.psnr_measure_rgb(pred, hr, shave_border=0)
        else:
            psnr = utils.psnr_measure(pred, hr, shave_border=0)
        psnrs.append(psnr)
    psnr = np.mean(np.array(psnrs))
    avg_flops = total_flops / len(valid_loader)
    if phase <= 1:
        return psnr, avg_flops
    else:
        avg_ratio_easy = total_ratio_easy / len(valid_loader)
        avg_label_acc = total_label_acc / len(valid_loader)
        return psnr, avg_flops, avg_ratio_easy, avg_label_acc


def main(config_, save_path):
    global config, log, phase
    config = config_
    phase = config['phase']
    
    log = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    if config.get('seed') is not None:
        utils.set_random_seed(config['seed'])

    train_loader = make_train_loader()
    model, model_ref, optimizer, lr_scheduler, iter_start = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)
        if model_ref:
            model_ref = nn.parallel.DataParallel(model_ref)

    valid_data_name = config['valid_dataset']['dataset']['args']['root_path_1'].split('/')[2]
    iter_max = config['iter_max']
    iter_print = config['iter_print']
    iter_val = config['iter_val']
    iter_save = config['iter_save']

    iter_cur = iter_start
    timer = utils.Timer()
    t_iter_start = timer.t()

    if phase <= 1:
        loss_fn = nn.L1Loss()   
    else:
        loss_fn_rgb = nn.L1Loss()
        loss_fn_avg = nn.L1Loss()
        loss_fn_ce = nn.NLLLoss()
        train_loss_rgb = utils.Averager()
        train_loss_avg = utils.Averager()
        train_loss_ce = utils.Averager()
        train_label_acc = utils.Averager()
        train_label_error = utils.Averager()
    train_loss = utils.Averager()

    rgb_mean = torch.tensor(config['data_norm']['mean'], device='cuda').view(1,3,1,1)
    rgb_std = torch.tensor(config['data_norm']['std'], device='cuda').view(1,3,1,1)
    while True:
        for batch in train_loader:
            # process one iteration
            optimizer.zero_grad()
            model.train()
            if phase == 2:
                model.encoder.eval()
                model.light_sampler.eval()
                model.heavy_sampler.eval()
            elif phase > 0:
                model.encoder.eval()
                model.heavy_sampler.eval()

            for k, v in batch.items():
                batch[k] = v.cuda()

            lr = (batch['lr'] - rgb_mean) / rgb_std
            hr = (batch['hr'] - rgb_mean) / rgb_std

            if phase <= 1:
                pred = model(lr, batch['coord'], batch['cell'])
                loss = loss_fn(pred, hr)
            else:
                pred, diff = model(lr, batch['coord'], batch['cell']) # (b,c,h,w), (b,2,h,w)
                # generate difficulty label
                with torch.no_grad():
                    pred_light, pred_heavy = model_ref(lr, batch['coord'], batch['cell'], both=True)
                    mae_light = torch.abs(pred_light - hr).mean(dim=1)
                    mae_heavy = torch.abs(pred_heavy - hr).mean(dim=1)
                    label = torch.where(mae_light < mae_heavy, 0, 1) # (b,h,w)

                    tot_cnt = torch.ones(1, device=label.device) * label.numel() # (bhw,)
                    if config.get('use_ref_cnt'):
                        hard_cnt = label.sum()
                    else:
                        hard_cnt_ratio = config.get('hard_cnt_ratio')
                        if hard_cnt_ratio is None:
                            hard_cnt_ratio = 0.5
                        hard_cnt = tot_cnt * hard_cnt_ratio
                    easy_cnt = tot_cnt - hard_cnt

                    # evaluation between diff and label
                    diff_easy, diff_hard = diff[:,0,:,:], diff[:,1,:,:]
                    acc = (((diff_easy > diff_hard) & (label == 0)).sum()\
                        + ((diff_easy <= diff_hard) & (label == 1)).sum())/tot_cnt
                    train_label_acc.add(acc.item())
                    
                    error = (diff_hard - label) ** 2
                    train_label_error.add(error.mean().item())

                loss_rgb = loss_fn_rgb(pred, hr)
                loss_avg = (loss_fn_avg(easy_cnt, diff[:,0,:,:].sum()) +\
                     loss_fn_avg(hard_cnt, diff[:,1,:,:].sum())) / tot_cnt
                loss_ce = loss_fn_ce(torch.log(diff), label)

                loss_rgb *= config['loss_rgb_w']
                loss_avg *= config['loss_avg_w']
                loss_ce *= config['loss_ce_w']

                train_loss_rgb.add(loss_rgb.item())
                train_loss_avg.add(loss_avg.item())
                train_loss_ce.add(loss_ce.item())

                loss = loss_rgb + loss_avg+ loss_ce

            train_loss.add(loss.item())
            loss.backward()
            optimizer.step() 
            lr_scheduler.step()
            
            model_ = model.module if n_gpus > 1 else model
            if model_ref:
                model_ref_ = model_ref.module if n_gpus > 1 else model_ref

            if iter_cur % iter_print == 0 or iter_cur % iter_save == 0:
                # save current model state
                model_spec = config['model']
                model_spec['sd'] = model_.state_dict()
                optimizer_spec = config['optimizer']
                optimizer_spec['sd'] = optimizer.state_dict()
                lr_scheduler_spec = config['lr_scheduler']
                lr_scheduler_spec['sd'] = lr_scheduler.state_dict()
                sv_file = {
                    'model': model_spec,
                    'optimizer': optimizer_spec,
                    'lr_scheduler': lr_scheduler_spec,
                    'iter': iter_cur
                }
                if iter_cur % iter_print == 0: # log
                    log_info = ['iter {}/{}'.format(iter_cur, iter_max)]
                    if phase <= 1:
                        log_info.append('train: loss={:.4f}'.format(train_loss.item()))
                    else:
                        log_info.append('train: loss={:.4f} | loss_rgb={:.4f} | loss_avg={:.4f} | loss_ce={:.4f} | label_acc={:.1f} % | label_error={:.4f}'\
                            .format(train_loss.item(), train_loss_rgb.item(), train_loss_avg.item(), train_loss_ce.item(), 
                                train_label_acc.item()*100, train_label_error.item()))
                    log_info.append('lr: {:.4e}'.format(lr_scheduler.get_last_lr()[0]))

                    t = timer.t()
                    prog = (iter_cur - iter_start + 1) / (iter_max - iter_start + 1)
                    t_iter = utils.time_text(t - t_iter_start)
                    t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
                    log_info.append('{} {}/{}'.format(t_iter, t_elapsed, t_all))
                    log(', '.join(log_info))
                    train_loss = utils.Averager()
                    if phase >= 2:
                        train_loss_rgb = utils.Averager()
                        train_loss_avg = utils.Averager()
                        train_loss_ce = utils.Averager()
                        train_label_acc = utils.Averager()
                        train_label_error = utils.Averager()
                    t_iter_start = timer.t()
                    torch.save(sv_file, os.path.join(save_path, 'iter_last.pth'))

                if iter_cur % iter_save == 0:
                    torch.save(sv_file, os.path.join(save_path, 'iter_{}.pth'.format(iter_cur)))

            if iter_cur % iter_val == 0: # validation
                if phase <= 1:
                    psnr, flops = valid(model_)
                    log('{} (x{}) | psnr({}): {:.2f} dB | flops (per patch): {:.2f}G'\
                        .format(valid_data_name, config['scale'], config['psnr_type'], psnr, flops/1e9))
                else:
                    psnr, flops, ratio_easy, label_acc = valid(model_, model_ref=model_ref_, diff_threshold=0)
                    log('{} (x{}) | psnr({}): {:.2f} dB | flops (per patch): {:.2f}G | easy ratio (per patch): {:.1f} % | label_acc (per patch): {:.1f} %'\
                        .format(valid_data_name, config['scale'], config['psnr_type'], psnr, flops/1e9, ratio_easy*100, label_acc*100))
                    psnr, flops, ratio_easy, label_acc = valid(model_, model_ref=model_ref_, diff_threshold=0.5)
                    log('{} (x{}) | psnr({}): {:.2f} dB | flops (per patch): {:.2f}G | easy ratio (per patch): {:.1f} % | label_acc (per patch): {:.1f} %'\
                        .format(valid_data_name, config['scale'], config['psnr_type'], psnr, flops/1e9, ratio_easy*100, label_acc*100))

            if iter_cur == iter_max:
                log('--- End training ---')
                return
            iter_cur += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('Config loaded ...', args.config)

    config['gpu'] = args.gpu
    config['resume'] = args.resume
    if not config.get('psnr_type'):
        config['psnr_type'] = 'y'
    if not config.get('data_norm'):
        config['data_norm']['mean'] = [0.5, 0.5, 0.5]
        config['data_norm']['std'] = [0.5, 0.5, 0.5]

    save_path = os.path.join('save', args.config.split('/')[-1][:-len('.yaml')])
    os.makedirs(save_path, exist_ok=True)
    main(config, save_path)
