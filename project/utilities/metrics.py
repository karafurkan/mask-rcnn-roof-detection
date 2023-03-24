import torch
import numpy as np
from sklearn.metrics import f1_score


def f1_score_per_class(outputs, targets, num_classes, avg_option="macro"):
    # Todo: Research average options
    """
    # micro: Calculate metrics globally by counting the total true positives,
       false negatives and false positives.
    # macro: Calculate metrics for each label, and find their unweighted mean.
       This does not take label imbalance into account.
    # weighted: Calculate metrics for each label, and find their average weighted
      by support (the number of true instances for each label). This alters ‘macro’
      to account for label imbalance; it can result in an F-score that is not between
      precision and recall.
    """
    f1_scores = []
    for i in range(num_classes):
        f1_scores.append(
            f1_score(
                outputs[:, i].flatten(),
                targets[:, i].flatten(),
                average=avg_option,
                zero_division=0,
            )
        )
    return f1_scores


def iou(pred, target, n_classes=3):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (
            (pred_inds[target_inds]).long().sum().data.cpu().item()
        )  # Cast to long to prevent overflows
        union = (
            pred_inds.long().sum().data.cpu().item()
            + target_inds.long().sum().data.cpu().item()
            - intersection
        )
        if union > 0:
            ious.append(float(intersection) / float(max(union, 1)))

    return np.array(ious)


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
