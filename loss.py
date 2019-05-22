# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class BalancedBCEWithLogitsLoss(nn.Module):
    
    def __init__(self, grad_clip=False):
        super(BalancedBCEWithLogitsLoss, self).__init__()
        self.grad_clip = grad_clip
    
    def forward(self, logits, labels):
        # logits: shape(batch_size, num_classes), dtype=float
        # labels: shape(batch_size, num_classes), dtype=float
        # labels must be a binary valued tensor
        assert logits.shape == labels.shape, "logits shape %r != labels shape %r" % (logits.shape, labels.shape)
        # number of classes
        nc = labels.shape[1]
        # number of positive classes per example in batch
        npos_per_example = labels.sum(1)                # shape: [batch_size]
        
        # alpha: ratio of negative classes per example in batch
        alpha = (nc - npos_per_example) / npos_per_example
        alpha = alpha.unsqueeze(1).expand_as(labels)    # shape: [batch_size, num_classes]
        
        # positive weights
        pos_weight = labels * alpha
        
        # to avoid gradients vanishing
        if self.grad_clip:
            logits = logits.clamp(min=-14.0, max=14.0)
        
        proba = torch.sigmoid(logits)
        # see https://pytorch.org/docs/stable/nn.html#torch.nn.BCEWithLogitsLoss for loss eq.
        loss = -(torch.log(proba) * pos_weight + torch.log(1. - proba) * (1. - labels))
        loss = loss.mean()
        
        return loss
