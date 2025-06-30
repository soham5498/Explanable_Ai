import torch
from sklearn.metrics import average_precision_score

def compute_pixel_accuracy(pred_mask, gt_mask):
    """
    Pixel-wise accuracy between binary predicted mask and ground truth mask.
    Args:
        pred_mask: torch tensor [H,W] with 0/1
        gt_mask: torch tensor [H,W] with 0/1
    Returns:
        Pixel Accuracy (float)
    """
    correct = (pred_mask == gt_mask).sum().item()
    total = gt_mask.numel()
    return correct / total

def compute_iou(pred_mask, gt_mask):
    """
    Intersection over Union.
    Args:
        pred_mask: torch tensor [H,W] with 0/1
        gt_mask: torch tensor [H,W] with 0/1
    Returns:
        IoU (float)
    """
    intersection = ((pred_mask == 1) & (gt_mask == 1)).sum().item()
    union = ((pred_mask == 1) | (gt_mask == 1)).sum().item()
    return intersection / union if union != 0 else 0.0

def compute_ap(heatmap, gt_mask):
    """
    Average Precision score using raw heatmap vs. GT mask.
    Args:
        heatmap: torch tensor [H,W] raw normalized [0,1]
        gt_mask: torch tensor [H,W] with 0/1
    Returns:
        Average Precision (float)
    """
    y_true = gt_mask.flatten().cpu().numpy()
    y_score = heatmap.flatten().cpu().numpy()
    return average_precision_score(y_true, y_score)
