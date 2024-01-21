import torch
import torch.nn as nn
import torch.nn.functional as F
from bicubic_pytorch import core

import models
from models import register


@register('ewm')
class EWM(nn.Module):

    def __init__(self, srnet1_spec, srnet2_spec, classifier_spec):
        super().__init__()
        self.srnet1 = models.make(srnet1_spec)
        self.srnet2 = models.make(srnet2_spec)
        self.classifier = models.make(classifier_spec)

    def forward(self, inp, coord, cell, diff_threshold=0.5):  
        pred1 = self.srnet1(inp, coord, cell)
        pred2 = self.srnet2(inp, coord, cell)
        pred_prob = self.classifier(inp, coord, cell)

        if self.training:
            pred = pred_prob[:,0:1,:,:] * pred1 + pred_prob[:,1:2,:,:] * pred2
            return pred, pred_prob
        else:
            pred_prob = pred_prob[:,1:2,:,:].repeat(1,3,1,1)
            pred = torch.where(pred_prob > diff_threshold, pred2, pred1)
            cnt_small = (pred_prob <= diff_threshold).sum().item()
            cnt_large = (pred_prob > diff_threshold).sum().item()
            ratio_small = cnt_small / (cnt_small + cnt_large)
            return pred, ratio_small