import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from utils import make_coord
import models.utils as mutils
import functools

import models
from models import register
import warnings
warnings.filterwarnings("ignore")


@register('light-conv')
class LightConv(nn.Module):
    def __init__(self, hidden_dim=128, out_dim=32, stride=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.stride = stride

        self.convs = nn.Sequential(
            nn.Conv2d(3, hidden_dim, stride, stride),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(hidden_dim, out_dim, 1)
        )
        mutils.initialize_weights([self.convs], 0.1)

    def forward(self, inp):
        h,w = inp.shape[-2:]
        feat = self.convs(inp) # (b,c,h//stride,w//stride)
        feat = feat.unsqueeze(3).unsqueeze(-1).repeat(1,1,1,self.stride,1,self.stride)\
            .view(*feat.shape[:2], h, w) # (b,c,h,w)
        return feat

    def flops(self, input_resolution):
        h,w = input_resolution
        h,w = h//self.stride, w//self.stride
        flops = 2 * 3*self.hidden_dim * h*w * self.stride*self.stride
        flops += 3 * 2 * self.hidden_dim*self.hidden_dim * h*w * 1*1
        flops += 2 * self.hidden_dim*self.out_dim * h*w * 1*1
        return flops


@register('classifier')
class Classifier(nn.Module):

    def __init__(self, conv_spec, hidden_depth, feat_unfold=False, local_ensemble=False):
        super().__init__()
        self.encoder = models.make(conv_spec)
        imnet_in_dim = self.encoder.out_dim
        imnet_spec = {'name': 'mlp', 'args': {'hidden_list': [imnet_in_dim]*hidden_depth}}
        if feat_unfold:
            imnet_in_dim *= 9
        imnet_in_dim += 4 # attach coord, cell

        self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim, 'out_dim': 2})
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold

    def forward(self, inp, coord, cell):
        feat = self.encoder(inp) # (b,c,h,w)
        feat_coord = make_coord((feat.shape[-2:]), flatten=False, device=feat.device)\
            .permute(2,0,1).unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:]) # (b,2,h,w)
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

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

                q_feat = F.grid_sample(feat, coord_.flip(-1), mode='nearest', align_corners=False)
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1), mode='nearest', align_corners=False)

                rel_coord = coord.permute(0,3,1,2) - q_coord
                rel_coord[:,0,:,:] *= feat.shape[-2]
                rel_coord[:,1,:,:] *= feat.shape[-1]

                rel_cell = cell.clone()
                rel_cell[:, 0] *= feat.shape[-2]
                rel_cell[:, 1] *= feat.shape[-1]
                rel_cell = rel_cell.unsqueeze(-1).unsqueeze(-1).repeat(1,1,*rel_coord.shape[-2:])

                inp = torch.cat([q_feat, rel_coord, rel_cell], dim=1).transpose(1,3) # (b,w,h,c)
                pred = self.imnet(inp.contiguous()).transpose(1,3) # (b,c,h,w)
                preds.append(pred)

                area = torch.abs(rel_coord[:,0,:,:] * rel_coord[:,1,:,:])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(1)
        ret = F.softmax(ret, dim=1)
        return ret

    def flops(self, input_resolution, sample_q):
        flops = self.encoder.flops(input_resolution)
        flops_imnet = self.imnet.flops() * sample_q
        if self.local_ensemble:
            flops_imnet *= 4
        flops += flops_imnet
        return flops

if __name__ == '__main__':
    input_resolution = (32,32)
    conv_spec = {'name': 'light-conv', 'args': {'hidden_dim': 128, 'out_dim': 32, 'stride': 4}}

    model = Classifier(conv_spec, hidden_depth=2, feat_unfold=False, local_ensemble=False)
    print('input_resolution: {} | FLOPs: {:.2f} G'.format(\
        input_resolution, model.flops(input_resolution, 128*128)/1e9))
    print('model: #params={}'.format(utils.compute_num_params(model, text=True)))

    h,w = 128,128
    coord = make_coord((h,w), flatten=False).unsqueeze(0)
    cell = torch.tensor([2/h, 2/w]).unsqueeze(0)

    x = torch.randn((1, 3, 32, 32))
    x = model(x, coord, cell)
    print(x.shape, x.min().item(), x.max().item())