import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models
from models import register
from utils import make_coord


@register('lte')
class LTE(nn.Module):

    def __init__(self, encoder_spec, imnet_spec, hidden_dim, local_ensemble=False):
        super().__init__()        
        self.encoder = models.make(encoder_spec)
        self.scale = self.encoder.scale

        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.phase = nn.Linear(2, hidden_dim//2, bias=False)        
        self.imnet = models.make(imnet_spec, args={'in_dim': hidden_dim, 'out_dim': 3})

        self.hidden_dim = hidden_dim
        self.local_ensemble = local_ensemble

    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        self.feat_coord = make_coord(self.feat.shape[-2:], flatten=False, device=self.feat.device) \
            .permute(2,0,1).unsqueeze(0).expand(self.feat.shape[0], 2, *self.feat.shape[-2:])
        
        self.coeff = self.coef(self.feat)
        self.freqq = self.freq(self.feat)

    def query_rgb(self, coord, cell):
        h,w = coord.shape[1:3]
        feat = self.feat
        coef = self.coeff
        freq = self.freqq
        feat_coord = self.feat_coord

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:,:,:,0] += vx * rx + eps_shift
                coord_[:,:,:,1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                q_coef = F.grid_sample(coef, coord_.flip(-1), mode='nearest', align_corners=False)
                q_freq = F.grid_sample(freq, coord_.flip(-1), mode='nearest', align_corners=False)
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1), mode='nearest', align_corners=False)

                rel_coord = coord.permute(0,3,1,2) - q_coord
                rel_coord[:,0,:,:] *= feat.shape[-2]
                rel_coord[:,1,:,:] *= feat.shape[-1]
                area = torch.abs(rel_coord[:,0,:,:] * rel_coord[:,1,:,:])
                areas.append(area + 1e-9)

                rel_cell = cell.clone()
                rel_cell[:, 0] *= feat.shape[-2]
                rel_cell[:, 1] *= feat.shape[-1]
                rel_cell = rel_cell.unsqueeze(-1).unsqueeze(-1).repeat(1,1,*rel_coord.shape[-2:])

                # basis generation
                q_coef = q_coef.flatten(-2).transpose(1,2)
                q_freq = q_freq.flatten(-2).transpose(1,2)
                rel_coord = rel_coord.flatten(-2).transpose(1,2)
                rel_cell = rel_cell.flatten(-2).transpose(1,2)

                q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
                q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))
                q_freq = torch.sum(q_freq, dim=-2)
                q_freq += self.phase(rel_cell.contiguous())
                q_freq = torch.cat((torch.cos(np.pi*q_freq), torch.sin(np.pi*q_freq)), dim=-1)
                inp = torch.mul(q_coef, q_freq)            

                pred = self.imnet(inp.contiguous()).transpose(1,2)
                pred = pred.view(*pred.shape[:2], h, w)
                preds.append(pred)     

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(1)
        return ret + F.grid_sample(self.inp, coord.flip(-1), mode='bilinear',\
            padding_mode='border', align_corners=False)

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)

    def flops(self, input_resolution, sample_q=None):
        h,w = input_resolution
        if not sample_q:
            H,W = h*self.scale, w*self.scale
            sample_q = H*W

        # self.phase is negligible
        flops = self.imnet.flops() * sample_q 
        if self.local_ensemble:
            flops *= 4
        flops += self.encoder.flops(input_resolution) # encoder
        flops += 2*2 * self.encoder.out_dim * self.hidden_dim * h*w * 3*3 # freqq,coeff
        return flops