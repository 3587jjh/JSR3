import torch
import torch.nn as nn
import torch.nn.functional as F
from bicubic_pytorch import core

import models
from models import register


@register('ewb')
class EWB(nn.Module):

    def __init__(self, srnet_spec, classifier_spec):
        super().__init__()
        self.srnet = models.make(srnet_spec)
        self.classifier = models.make(classifier_spec)

    def forward(self, inp, coord, cell, diff_threshold=0.5):  
        pred = self.srnet(inp, coord, cell)
        pred_bi = core.imresize(inp, sizes=coord.shape[1:3]) # assume dense coord
        pred_prob = self.classifier(inp, coord, cell)

        if self.training:
            pred = pred_prob[:,0:1,:,:] * pred_bi + pred_prob[:,1:2,:,:] * pred
            return pred, pred_prob
        else:
            pred_prob_bi = pred_prob[:,0:1,:,:].repeat(1,3,1,1)
            pred_prob = pred_prob[:,1:2,:,:].repeat(1,3,1,1)
            pred = torch.where(pred_prob > diff_threshold, pred, pred_bi)
            cnt_bi = (pred_prob <= diff_threshold).sum().item()
            cnt_model = (pred_prob > diff_threshold).sum().item()
            ratio_bi = cnt_bi / (cnt_bi + cnt_model)
            return pred, ratio_bi