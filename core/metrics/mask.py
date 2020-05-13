import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm


def mask_IoU(prediction, target, num_classes):
    ious = torch.Tensor(
        [
            IoU(pred_frame == class_idx, gt_frame == class_idx)
            for class_idx in range(num_classes)
            for pred_frame, gt_frame in zip(prediction, target)
        ]
    )
    result = ious.mean() if len(ious) else 0
    return {"segmap_iou": result}


def IoU(binary_mask1, binary_mask_2):
    intersection = torch.sum(binary_mask1 & binary_mask_2)
    union = torch.sum(binary_mask1 | binary_mask_2)
    iou = intersection / union
    return iou


def class_evaluation(prediction, target, num_classes):
    cm = confusion_matrix(prediction, target, num_classes).float()

    recall = cm.diag() / (cm.sum(dim=1) + 1e-15)
    precision = cm.diag() / (cm.sum(dim=0) + 1e-15)
    iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-15)
    accuracy = cm.diag().sum() / (cm.sum() + 1e-15)

    m_recall = recall.mean()
    m_precision = precision.mean()
    m_iou = iou.mean()

    meters = {
        "m_iou": m_iou,
        "m_precision": m_precision,
        "m_recall": m_recall,
        "accuracy": accuracy,
    }
    return meters


def confusion_matrix(prediction, target, num_classes=2):
    prediction = prediction.flatten()
    target = target.flatten()

    target_mask = (target >= 0) & (target < num_classes)
    target = target[target_mask]
    prediction = prediction[target_mask]

    indices = num_classes * target + prediction
    cm = torch.bincount(indices, minlength=num_classes ** 2).reshape(
        num_classes, num_classes
    )
    return cm
