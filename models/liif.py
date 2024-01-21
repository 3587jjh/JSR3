import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord


@register('liif')
class LIIF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, hidden_list=None, hidden_depth=None, 
        feat_unfold=False, feat_unfold2=False, local_ensemble=False):
        super().__init__()
        self.encoder = models.make(encoder_spec, args={'lr_connection': False})
        self.scale = self.encoder.scale
        assert not (feat_unfold and feat_unfold2)

        imnet_in_dim = self.encoder.out_dim
        if not imnet_spec:
            if not hidden_list:
                assert hidden_depth
                hidden_list = [imnet_in_dim]*hidden_depth
            imnet_spec = {'name': 'mlp', 'args': {'hidden_list': hidden_list}}

        if feat_unfold:
            imnet_in_dim *= 9
        elif feat_unfold2:
            imnet_in_dim *= 25
        imnet_in_dim += 4 # attach coord, cell

        self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim, 'out_dim': 3})
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.feat_unfold2 = feat_unfold2

    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        self.feat_coord = make_coord(self.feat.shape[-2:], flatten=False, device=self.feat.device) \
            .permute(2, 0, 1).unsqueeze(0).expand(self.feat.shape[0], 2, *self.feat.shape[-2:])

    def query_rgb(self, coord, cell):
        feat = self.feat # (b,c,h,w)
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
        elif self.feat_unfold2:
            feat = F.unfold(feat, 5, padding=2).view(
                feat.shape[0], feat.shape[1] * 25, feat.shape[2], feat.shape[3])
        feat_coord = self.feat_coord # (b,2,h,w)

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
        ret = ret + F.grid_sample(self.inp, coord.flip(-1), mode='bilinear',
            padding_mode='border', align_corners=False)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)

    def flops(self, input_resolution, sample_q=None):
        flops = self.encoder.flops(input_resolution)
        if not sample_q:
            h,w = input_resolution
            H,W = h*self.scale, w*self.scale
            sample_q = H*W

        flops_imnet = self.imnet.flops() * sample_q
        if self.local_ensemble:
            flops_imnet *= 4
        flops += flops_imnet
        return flops


@register('eliif')
class Expert_LIIF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec, num_experts, reduction, hidden_dim):
        super().__init__()
        self.encoder = models.make(encoder_spec, args={'lr_connection': False})
        self.scale = self.encoder.scale

        self.experts = nn.ModuleList([nn.Conv2d(self.encoder.out_dim, hidden_dim, 3,1,1)\
            for _ in range(num_experts)])
        if reduction == 'concat':
            self.contraction = nn.Conv2d(hidden_dim*num_experts, hidden_dim, 1,1,0)
        self.imnet = models.make(imnet_spec, args={'in_dim': hidden_dim+4, 'out_dim': 3})

        self.num_experts = num_experts
        self.reduction = reduction
        self.hidden_dim = hidden_dim

    def gen_feat(self, inp):
        self.inp = inp
        feat = self.encoder(inp)
        feats = [expert(feat) for expert in self.experts]
        if self.reduction == 'sum':
            self.feat = torch.sum(torch.stack(feats), dim=0)
        elif self.reduction == 'mean':
            self.feat = torch.mean(torch.stack(feats), dim=0)
        else:
            self.feat = self.contraction(torch.cat(feats, dim=1))
        self.feat_coord = make_coord(self.feat.shape[-2:], flatten=False, device=self.feat.device) \
            .permute(2, 0, 1).unsqueeze(0).expand(self.feat.shape[0], 2, *self.feat.shape[-2:])

    def query_rgb(self, coord, cell):
        feat = self.feat # (b,c,h,w)
        feat_coord = self.feat_coord # (b,2,h,w)
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
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(1)
        ret = ret + F.grid_sample(self.inp, coord.flip(-1), mode='bilinear',
            padding_mode='border', align_corners=False)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)

    def flops(self, input_resolution, sample_q=None):
        h,w = input_resolution
        flops = self.encoder.flops(input_resolution)
        flops += self.num_experts*2 * self.encoder.out_dim * self.hidden_dim * h*w * 3*3
        if self.reduction == 'concat':
            flops += 2 * self.hidden_dim*self.num_experts * self.hidden_dim * h*w * 1*1

        if not sample_q:
            H,W = h*self.scale, w*self.scale
            sample_q = H*W

        flops_imnet = self.imnet.flops() * sample_q
        flops += flops_imnet
        return flops


@register('liif-sampler')
class LIIF_Sampler(nn.Module):
    # local ensemble not supported
    def __init__(self, imnet_spec, in_dim, out_dim, feat_unfold=False, add_type=0):
        super().__init__()
        if feat_unfold:
            in_dim *= 9
        in_dim += 4 # attach coord, cell
        if add_type == 1:
            in_dim += 3 # attach 2x2 neighboring rgb deviation
        elif add_type >= 2:
            raise NotImplementedError

        self.imnet = models.make(imnet_spec, args={'in_dim': in_dim, 'out_dim': out_dim})
        self.feat_unfold = feat_unfold
        self.add_type = add_type

    def make_inp(self, feat, coord, cell, lr=None):
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        feat_coord = make_coord(feat.shape[-2:], flatten=False, device=feat.device)\
            .permute(2,0,1).unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
        q_feat = F.grid_sample(feat, coord.flip(-1), mode='nearest', align_corners=False)
        q_coord = F.grid_sample(feat_coord, coord.flip(-1), mode='nearest', align_corners=False)

        rel_coord = coord.permute(0,3,1,2) - q_coord
        rel_coord[:,0,:,:] *= feat.shape[-2]
        rel_coord[:,1,:,:] *= feat.shape[-1]

        rel_cell = cell.clone()
        rel_cell[:, 0] *= feat.shape[-2]
        rel_cell[:, 1] *= feat.shape[-1]
        rel_cell = rel_cell.unsqueeze(-1).unsqueeze(-1).repeat(1,1,*rel_coord.shape[-2:])
        inp = torch.cat([q_feat, rel_coord, rel_cell], dim=1)

        if self.add_type == 1:
            assert lr is not None
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6

            # field radius (global: [-1, 1])
            rx = 2 / feat.shape[-2] / 2
            ry = 2 / feat.shape[-1] / 2

            q_rgbs = []
            for vx in vx_lst:
                for vy in vy_lst:
                    coord_ = coord.clone()
                    coord_[:,:,:,0] += vx * rx + eps_shift
                    coord_[:,:,:,1] += vy * ry + eps_shift
                    coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                    q_rgb = F.grid_sample(lr, coord_.flip(-1), mode='nearest', align_corners=False)
                    q_rgbs.append(q_rgb)
            rgb_std = torch.std(torch.stack(q_rgbs), dim=0)
            inp = torch.cat([inp, rgb_std], dim=1)

        elif self.add_type >= 2:
            assert NotImplementedError
        return inp    

    def forward(self, x, coord=None, cell=None, lr=None):
        inp = self.make_inp(x, coord, cell, lr=lr) if coord is not None else x
        ret = self.imnet(inp.transpose(1,3).contiguous()).transpose(1,3) # (b,c,h,w)
        return ret

    def flops(self, sample_q):
        flops = self.imnet.flops() * sample_q
        return flops