import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import models.utils as mutils
import functools
from models import register

class Block(nn.Module):
    def __init__(self, nf, group=1, use_conv1=False):
        super(Block, self).__init__()
        self.b1 = mutils.EResidualBlock(nf, nf, group=group, use_conv1=use_conv1)
        self.c1 = mutils.BasicBlock(nf*2, nf, 1, 1, 0)
        self.c2 = mutils.BasicBlock(nf*3, nf, 1, 1, 0)
        self.c3 = mutils.BasicBlock(nf*4, nf, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b1(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b1(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3

    def flops(self, input_resolution):
        h,w = input_resolution
        flops = 0
        flops += 3*self.b1.flops(input_resolution)
        flops += self.c1.flops(input_resolution)
        flops += self.c2.flops(input_resolution)
        flops += self.c3.flops(input_resolution)
        return flops

@register('carn')
class CARN_M(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, scale=4, group=4,
        lr_connection=False, no_upsampling=False, branch_point=None):

        super(CARN_M, self).__init__()
        self.scale = scale
        self.out_dim = nf
        self.mid_dim = nf

        #rgb_range = 1
        #rgb_mean = (0.4488, 0.4371, 0.4040)
        #rgb_std = (1.0, 1.0, 1.0)
        #self.sub_mean = mutils.MeanShift(rgb_range, rgb_mean, rgb_std)
        #self.add_mean = mutils.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.in_nc = in_nc
        self.out_nc = out_nc
        self.nf = nf
        self.branch_point = branch_point # 1,2,3,None

        self.entry = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.b1 = Block(nf, group=group)
        self.c1 = mutils.BasicBlock(nf*2, nf, 1, 1, 0)

        if not branch_point or branch_point > 1:
            self.b2 = Block(nf, group=group)
            self.c2 = mutils.BasicBlock(nf*3, nf, 1, 1, 0)
        
        if not branch_point or branch_point > 2:
            self.b3 = Block(nf, group=group)
            self.c3 = mutils.BasicBlock(nf*4, nf, 1, 1, 0)

        self.no_upsampling = no_upsampling
        if not no_upsampling:
            self.lr_connection = lr_connection
            self.upscale_func = functools.partial(
                F.interpolate, mode='bicubic', align_corners=False) # for upsampling inp_lr
            self.upsample = mutils.UpsampleBlock(nf, scale=scale, multi_scale=False, group=group)
            self.exit = nn.Conv2d(nf, out_nc, 3, 1, 1)  
                
    def forward(self, x):
        #x = self.sub_mean(x)
        inp_lr = x.clone()
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        out = o1

        if not self.branch_point or self.branch_point > 1:
            b2 = self.b2(o1)
            c2 = torch.cat([c1, b2], dim=1)
            o2 = self.c2(c2)
            out = o2

        if not self.branch_point or self.branch_point > 2:
            b3 = self.b3(o2)
            c3 = torch.cat([c2, b3], dim=1)
            o3 = self.c3(c3)
            out = o3

        if not self.no_upsampling:
            out = self.upsample(out, scale=self.scale)
            out = self.exit(out)
            if self.lr_connection:
               out = out + self.upscale_func(inp_lr, size=out.size()[2:])   
        #out = self.add_mean(out)
        return out

    def flops(self, input_resolution):
        h,w = input_resolution
        flops = 0
        flops += 2 * self.in_nc * self.nf * h*w * 3*3 # entry
        flops += self.b1.flops(input_resolution) + self.c1.flops(input_resolution)

        if not self.branch_point or self.branch_point > 1:
            flops += self.b2.flops(input_resolution) + self.c2.flops(input_resolution)

        if not self.branch_point or self.branch_point > 2:
            flops += self.b3.flops(input_resolution) + self.c3.flops(input_resolution)

        if not self.no_upsampling:
            flops += self.upsample.flops(input_resolution)
            flops += 2 * self.nf * self.out_nc * h*self.scale*w*self.scale * 3*3 # exit
        return flops

class CARN_M_Modified(nn.Module):
    def __init__(self, in_nc, nf, scale=4, group=4, branch_point=None):
        # assume lr_connection=false, no_upsampling=true
        super().__init__()
        self.out_dim = nf
        self.in_nc = in_nc
        self.nf = nf
        self.scale = scale
        self.branch_point = branch_point # 1,2,3,None

        self.entry = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.b1 = Block(nf, group=group, use_conv1=branch_point<1)
        self.c1 = mutils.BasicBlock(nf*2, nf, 1, 1, 0)
        self.b2 = Block(nf, group=group, use_conv1=branch_point<2)
        self.c2 = mutils.BasicBlock(nf*3, nf, 1, 1, 0)
        self.b3 = Block(nf, group=group, use_conv1=branch_point<3)
        self.c3 = mutils.BasicBlock(nf*4, nf, 1, 1, 0)

    def forward(self, x):
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)
        return o3

    def flops(self, input_resolution):
        h,w = input_resolution
        flops = 2 * self.in_nc * self.nf * h*w * 3*3 # entry
        flops += self.b1.flops(input_resolution) + self.c1.flops(input_resolution)
        flops += self.b2.flops(input_resolution) + self.c2.flops(input_resolution)
        flops += self.b3.flops(input_resolution) + self.c3.flops(input_resolution)
        return flops

class CARN_Separate(nn.Module):
    def __init__(self, in_nc, nf, scale, group): 
        # assume lr_connection=false, no_upsampling=true, branch_point=1
        super().__init__()
        self.out_dim = nf
        self.in_nc = in_nc
        self.nf = nf
        self.scale = scale
        self.group = group

        self.entry = nn.Conv2d(in_nc, nf, 3, 1, 1)        
        self.b1 = Block(nf, group=group)
        self.c1 = mutils.BasicBlock(nf*2, nf, 1, 1, 0)
        self.b2 = Block(nf, group=group)
        self.c2 = mutils.BasicBlock(nf*3, nf, 1, 1, 0)
        self.b3 = Block(nf, group=group)
        self.c3 = mutils.BasicBlock(nf*4, nf, 1, 1, 0)
                
    def forward(self, x, before_bp=True):
        if before_bp:
            x = self.entry(x)
            c0 = o0 = x
            b1 = self.b1(o0)
            c1 = torch.cat([c0, b1], dim=1)
            o1 = self.c1(c1)
            self.prev_c = c1
            return o1
        
        c1, o1 = self.prev_c, x
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)
        return o3

    def flops(self, input_resolution, before_bp=True):
        h,w = input_resolution
        if before_bp:
            flops = 2 * self.in_nc * self.nf * h*w * 3*3 # entry
            flops += self.b1.flops(input_resolution) + self.c1.flops(input_resolution)
            return flops
        flops = self.b2.flops(input_resolution) + self.c2.flops(input_resolution)
        flops += self.b3.flops(input_resolution) + self.c3.flops(input_resolution)
        return flops

@register('carn-upsampler')
class CARN_Upsampler(nn.Module):
    def __init__(self, feat_dim, out_dim, scale, group):
        super().__init__()
        self.feat_dim = feat_dim
        self.out_dim = out_dim
        self.scale = scale
        self.upsample = mutils.UpsampleBlock(feat_dim, scale=scale, multi_scale=False, group=group)
        self.exit = nn.Conv2d(feat_dim, out_dim, 3, 1, 1)

    def forward(self, x):
        x = self.upsample(x, scale=self.scale)
        x = self.exit(x)
        return x

    def flops(self, input_resolution):
        h,w = input_resolution
        flops = self.upsample.flops(input_resolution)
        flops += 2 * self.feat_dim * self.out_dim * h*self.scale*w*self.scale * 3*3
        return flops


@register('carn-small')
def make_carn(in_nc=3, out_nc=3, nf=36, scale=4, group=4,
    lr_connection=False, no_upsampling=False, branch_point=None):
    return CARN_M(in_nc=in_nc, out_nc=out_nc, nf=nf, scale=scale, group=group, 
        lr_connection=lr_connection, no_upsampling=no_upsampling, branch_point=branch_point)

@register('carn-medium')
def make_carn(in_nc=3, out_nc=3, nf=52, scale=4, group=4,
    lr_connection=False, no_upsampling=False, branch_point=None):
    return CARN_M(in_nc=in_nc, out_nc=out_nc, nf=nf, scale=scale, group=group, 
        lr_connection=lr_connection, no_upsampling=no_upsampling, branch_point=branch_point)

@register('carn-large')
def make_carn(in_nc=3, out_nc=3, nf=64, scale=4, group=4,
    lr_connection=False, no_upsampling=False, branch_point=None):
    return CARN_M(in_nc=in_nc, out_nc=out_nc, nf=nf, scale=scale, group=group, 
        lr_connection=lr_connection, no_upsampling=no_upsampling, branch_point=branch_point)

@register('carn-large-sep')
def make_carn_sep(in_nc=3, nf=64, scale=4, group=4):
    return CARN_Separate(in_nc=in_nc, nf=nf, scale=scale, group=group)

@register('carn-large-modified')
def make_carn_modified(in_nc=3, nf=64, scale=4, group=4, branch_point=None, **kwargs):
    return CARN_M_Modified(in_nc=in_nc, nf=nf, scale=scale, group=group, branch_point=branch_point)


if __name__ == '__main__':
    h,w = 32,32
    input_resolution = (h,w)
    model = CARN_M(in_nc=3, out_nc=3, nf=64, scale=4, group=4, 
        lr_connection=False, no_upsampling=False, branch_point=None)
    print('input_resolution: {} | FLOPs: {:.2f} G'.format(\
        input_resolution, model.flops(input_resolution)/1e9))
    print('model: #params={}'.format(utils.compute_num_params(model, text=True)))

    x = torch.randn((1, 3, h, w))
    x = model(x)
    print(x.shape)
