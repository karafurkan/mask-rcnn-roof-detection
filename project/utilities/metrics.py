import torch
import numpy as np


def compute_f1_score(pred_masks, target_masks, n_classes=2):
    f1_scores = []
    for cls in range(n_classes):
        pred_inds = (pred_masks == cls).long()
        target_inds = (target_masks == cls).long()
        tp = (pred_inds & target_inds).sum().float()
        fp = (pred_inds & (~target_inds)).sum().float()
        fn = ((~pred_inds) & target_inds).sum().float()
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_scores.append(
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )
    return torch.tensor(f1_scores).float()


def compute_iou(pred_masks, target_masks, n_classes=2):
    ious = []
    for idx in range(target_masks.shape[0]):  # loop over all instances in the image
        iou = []
        for cls in range(n_classes):
            pred_inds = pred_masks == cls
            target_inds = target_masks[idx] == cls
            intersection = (pred_inds & target_inds).float().sum()
            union = (pred_inds | target_inds).float().sum()
            iou.append((intersection + 1e-7) / (union + 1e-7))
        ious.append(iou)
    return ious


def compute_miou(pred_masks, target_masks, n_classes=2):
    ious = compute_iou(pred_masks, target_masks, n_classes=2)
    mean_iou = torch.tensor(ious).mean().item()
    return mean_iou


def compute_dice_score(pred, target, eps=1e-7, n_classes=2):
    scores = []
    for cls in range(n_classes):
        pred_inds = pred["masks"].int() == cls
        target_inds = (target["masks"] == cls).float()
        intersection = (pred_inds * target_inds).sum()
        union = pred_inds.sum() + target_inds.sum()
        score = (2.0 * intersection + eps) / (union + eps)
        scores.append(score)
    return scores


def mean_dice_score(pred, target, eps=1e-7, n_classes=2):
    scores = compute_dice_score(pred, target, eps, n_classes)
    mean_score = torch.mean(torch.tensor(scores))
    return mean_score.item()


# BUG: Accuracy is not working properly
def compute_accuracy(pred_masks, target_masks, n_classes=2):
    total = [0] * n_classes
    correct = [0] * n_classes
    for cls in range(n_classes):
        pred_inds = pred_masks.int() == cls
        target_inds = target_masks == cls
        correct[cls] += (pred_inds == target_inds).float().sum().float()
        total[cls] += target_inds.long().sum().float()
    accuracies = []
    for cls in range(n_classes):
        if total[cls] == 0:
            accuracies.append(float("nan"))
        else:
            accuracies.append(correct[cls] / total[cls])
    return accuracies


def mean_accuracy(pred_masks, target_masks, n_classes=2):
    accs = compute_accuracy(pred_masks, target_masks, n_classes)
    mean_acc = sum(accs[1:]) / (n_classes - 1)  # Skip background
    return mean_acc.item()


def save_metric_scores(
    loss_epoch_mean,
    iter_loss,
    accuracy_score,
    dice_score,
    miou_score,
    pred_score,
    f1_score,
    file_name="hl_256_",
):
    path = "metric_scores/" + file_name + "loss_epoch.npy"
    np.save(path, np.asarray(loss_epoch_mean))

    path = "metric_scores/" + file_name + "iter_loss.npy"
    np.save(path, np.asarray(iter_loss))

    path = "metric_scores/" + file_name + "accuracy_score.npy"
    np.save(path, np.asarray(accuracy_score))

    path = "metric_scores/" + file_name + "dice_score.npy"
    np.save(path, np.asarray(dice_score))

    path = "metric_scores/" + file_name + "miou_score.npy"
    np.save(path, np.asarray(miou_score))

    path = "metric_scores/" + file_name + "pred_score.npy"
    np.save(path, np.asarray(pred_score))

    path = "metric_scores/" + file_name + "f1_score.npy"
    np.save(path, np.asarray(f1_score))
