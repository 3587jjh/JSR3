import torch
import torch.nn as nn
import torch.nn.functional as F
from bicubic_pytorch import core

import models
from models import register


@register('pcsr-phase0')
class PCSR(nn.Module):

    def __init__(self, encoder_spec, heavy_sampler_spec):
        super().__init__()
        self.encoder = models.make(encoder_spec, args={'lr_connection': False})
        in_dim = self.encoder.out_dim
        self.heavy_sampler = models.make(heavy_sampler_spec,
            args={'in_dim': in_dim, 'out_dim': 3})

    def forward(self, lr, coord, cell):
        feat = self.encoder(lr)
        pred = self.heavy_sampler(feat, coord, cell) + F.grid_sample(lr, 
            coord.flip(-1), mode='bilinear', padding_mode='border', align_corners=False)
        return pred

    def flops(self, input_resolution, sample_q):
        flops = self.encoder.flops(input_resolution)
        flops += self.heavy_sampler.flops(sample_q)
        return flops


@register('pcsr-phase1')
class PCSR(nn.Module):

    def __init__(self, encoder_spec, light_sampler_spec, heavy_sampler_spec):
        super().__init__()
        self.encoder = models.make(encoder_spec, args={'lr_connection': False})
        in_dim = self.encoder.out_dim
        self.light_sampler = models.make(light_sampler_spec,
            args={'in_dim': in_dim, 'out_dim': 3})
        self.heavy_sampler = models.make(heavy_sampler_spec,
            args={'in_dim': in_dim, 'out_dim': 3}) # not used

    def forward(self, lr, coord, cell, both=False):
        feat = self.encoder(lr)
        pred = self.light_sampler(feat, coord, cell) + F.grid_sample(lr, 
            coord.flip(-1), mode='bilinear', padding_mode='border', align_corners=False)
        if both:
            pred_heavy = self.heavy_sampler(feat, coord, cell) + F.grid_sample(lr, 
                coord.flip(-1), mode='bilinear', padding_mode='border', align_corners=False)
            return pred, pred_heavy
        else:
            return pred

    def flops(self, input_resolution, sample_q):
        flops = self.encoder.flops(input_resolution)
        flops += self.light_sampler.flops(sample_q)
        return flops


@register('pcsr')
class PCSR(nn.Module):

    def __init__(self, encoder_spec, light_sampler_spec, heavy_sampler_spec,
        classifier_type, classifier_spec):
        super().__init__()
        self.encoder = models.make(encoder_spec, args={'lr_connection': False})
        in_dim = self.encoder.out_dim
        self.light_sampler = models.make(light_sampler_spec,
            args={'in_dim': in_dim, 'out_dim': 3})
        self.heavy_sampler = models.make(heavy_sampler_spec,
            args={'in_dim': in_dim, 'out_dim': 3})
        self.classifier = models.make(classifier_spec,
            args={'in_dim': in_dim, 'out_dim': 2, 'add_type': classifier_type})


    def forward(self, lr, coord, cell, diff_threshold=0.5):
        if self.training:
            return self.forward_train(lr, coord, cell)
        else:
            return self.forward_test(lr, coord, cell, diff_threshold)

    def forward_train(self, lr, coord, cell):  
        feat = self.encoder(lr)
        diff = self.classifier(feat, coord, cell, lr=lr)
        diff = F.softmax(diff, dim=1)

        pred_light = self.light_sampler(feat, coord, cell)
        pred_heavy = self.heavy_sampler(feat, coord, cell)
        pred = diff[:,0:1,:,:] * pred_light + diff[:,1:2,:,:] * pred_heavy
        pred = pred + F.grid_sample(lr, coord.flip(-1), mode='bilinear',
            padding_mode='border', align_corners=False)
        return pred, diff 

    def forward_test(self, lr, coord, cell, threshold):
        b,h,w = coord.shape[:3]
        tot = b*h*w

        feat = self.encoder(lr)
        diff = self.classifier(feat, coord, cell, lr=lr)
        diff = F.softmax(diff, dim=1)
        #diff = 1 - torch.sqrt(1-diff) # make distribution uniform
        flops = (self.encoder.flops(lr.shape[-2:]) + self.classifier.flops(h*w)) * b

        inp_light = self.light_sampler.make_inp(feat, coord, cell)\
            .permute(0,2,3,1).contiguous().view(tot, -1) # (tot,c)
        inp_heavy = self.heavy_sampler.make_inp(feat, coord, cell)\
            .permute(0,2,3,1).contiguous().view(tot, -1) # (tot,c)
        diff = diff[:,1:2,:,:].flatten() # (tot,)
        pred = torch.zeros((tot,3), device=diff.device) # (tot,3)

        idx_easy = torch.where(diff <= threshold)[0]
        idx_hard = torch.where(diff > threshold)[0]
        num_easy, num_hard = len(idx_easy), len(idx_hard)
        ratio_easy = num_easy / (num_easy + num_hard)

        if num_easy > 0:
            inp = inp_light[idx_easy].transpose(0,1).unsqueeze(0).unsqueeze(-1) # (b=1,c,k,1)
            pred_light = self.light_sampler(inp)
            pred[idx_easy] = pred_light.view(3,-1).transpose(0,1)
            flops += self.light_sampler.flops(num_easy)

        if num_hard > 0:
            inp = inp_heavy[idx_hard].transpose(0,1).unsqueeze(0).unsqueeze(-1) # (b=1,c,k,1)
            pred_heavy = self.heavy_sampler(inp)
            pred[idx_hard] = pred_heavy.view(3,-1).transpose(0,1)
            flops += self.heavy_sampler.flops(num_hard)

        pred = pred.view(b,h,w,3).permute(0,3,1,2) # (b,3,h,w)
        pred = pred + F.grid_sample(lr, coord.flip(-1), mode='bilinear',
            padding_mode='border', align_corners=False)
        return pred, flops, ratio_easy