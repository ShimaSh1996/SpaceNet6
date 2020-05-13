"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

from core.nn.DualTaskLoss import DualTaskLoss
from core.config import cfg


class JointEdgeSegLoss(nn.Module):
    def __init__(
        self,
        classes,
        ignore_index=255,
        upper_bound=1.0,
    ):
        super().__init__()
        self.num_classes = classes

        self.weighted_seg_loss = ImageBasedCrossEntropyLoss(
            classes=classes, ignore_index=ignore_index, upper_bound=upper_bound
        )
        self.seg_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dual_task = DualTaskLoss(self.num_classes)

    def bce2d(self, output, target):        
        output = output.view(-1)
        target = target.view(-1)

        pos_ratio = (target == 1).float().mean()
        class_weight = torch.stack([1 - pos_ratio, pos_ratio])
        position_weight = class_weight[target.long()]

        loss = position_weight * F.binary_cross_entropy(output, target.float(), reduction="none")
        loss = loss.mean()

        return loss

    def edge_attention(self, input, target, edge):
        n, c, h, w = input.size()
        filler = torch.ones_like(target) * 255
        return self.seg_loss(
            input, torch.where(edge.max(1)[0] > 0.8, target, filler)
        )

    def forward(self, inputs, targets, mode):
        segin, edgein = inputs
        segmask, edgemask = targets

        losses = {}

        seg_loss_fn = (
            self.weighted_seg_loss if mode == "train" else self.seg_loss
        )
        losses["seg_loss"] = seg_loss_fn(segin, segmask)
        losses["edge_loss"] = 20 * self.bce2d(edgein, edgemask)
        losses["att_loss"] = self.edge_attention(
            segin, segmask, edgein
        )
        losses["dual_loss"] = self.dual_task(segin, segmask)

        return losses


# Img Weighted Loss
class ImageBasedCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        classes,
        weight=None,
        ignore_index=255,
        norm=False,
        upper_bound=1.0,
    ):
        super().__init__()
        logging.info("Using Per Image based weighted loss")
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss(weight, ignore_index)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = cfg.LOSS.BATCH_WEIGHTING

    def calculateWeights(self, target):
        hist = np.histogram(
            target.flatten(), range(self.num_classes + 1), normed=True
        )[0]
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def forward(self, inputs, targets):
        device = inputs.device
        target_cpu = targets.cpu().numpy()
        if self.batch_weights:
            weights = self.calculateWeights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights).to(device)

        loss = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculateWeights(target_cpu[i])
                self.nll_loss.weight = torch.Tensor(weights).to(device)

            loss += self.nll_loss(
                F.log_softmax(inputs[i].unsqueeze(0)), targets[i].unsqueeze(0)
            )
        return loss
