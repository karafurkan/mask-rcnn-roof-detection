import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
from dataset import ImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import random as rng

rng.seed(123)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    cp_filename = "checkpoints/" + filename
    torch.save(state, cp_filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    train_images_root=None,
    val_images_root=None,
    batch_size=2,
    num_workers=2,
    pin_memory=True,
):
    """Get the train and validation data loaders.

    Args:
        train_dir (_type_): _description_
        train_maskdir (_type_): _description_
        val_dir (_type_): _description_
        val_maskdir (_type_): _description_
        batch_size (_type_): _description_
        train_transform (_type_): _description_
        val_transform (_type_): _description_
        num_workers (int, optional): _description_. Defaults to 4.
        pin_memory (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    if train_images_root is not None:
        dataset_train = ImageDataset(
            train_images_root, transforms=torchvision.transforms.ToTensor()
        )
        train_loader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=lambda x: list(zip(*x)),
        )
    else:
        train_loader = None
    dataset_val = ImageDataset(
        val_images_root, transforms=torchvision.transforms.ToTensor()
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda x: list(zip(*x)),
    )

    return train_loader, val_loader


def get_test_loader(
    test_dir,
    batch_size,
    test_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = ImageDataset(
        image_dir=test_dir,
        mask_dir=None,
        transform=test_transform,
    )

    test_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return test_loader


def compute_iou(pred_masks, target_masks, n_classes=3):
    ious = []
    for cls in range(n_classes):
        pred_inds = pred_masks == cls
        target_inds = target_masks == cls
        # BUG: dimension error here: 108
        intersection = (pred_inds[target_inds]).long().sum().float()
        union = (
            pred_inds.long().sum().float()
            + target_inds.long().sum().float()
            - intersection
        )
        if union == 0:
            ious.append(
                float("nan")
            )  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(intersection / union)
    return ious


def compute_miou(pred_masks, target_masks, n_classes=3):
    ious = compute_iou(pred_masks, target_masks, n_classes)
    miou = torch.mean(torch.tensor(ious))
    return miou.item()


def dice_score(pred_masks, target_masks, eps=1e-7, n_classes=3):
    scores = []
    for cls in range(n_classes):
        pred_inds = (pred_masks == cls).float()
        target_inds = (target_masks == cls).float()
        intersection = (pred_inds * target_inds).sum()
        union = pred_inds.sum() + target_inds.sum()
        score = (2.0 * intersection + eps) / (union + eps)
        scores.append(score)
    return scores


def mean_dice_score(pred_masks, target_masks, eps=1e-7, n_classes=3):
    scores = dice_score(pred_masks, target_masks, eps, n_classes)
    mean_score = torch.mean(torch.tensor(scores))
    return mean_score.item()


def accuracy(pred_masks, target_masks, n_classes=3):
    total = 0
    correct = 0
    for cls in range(n_classes):
        pred_inds = pred_masks == cls
        target_inds = target_masks == cls
        correct += (pred_inds == target_inds).long().sum().float()
        total += target_inds.long().sum().float()
    if total == 0:
        return float("nan")
    return correct / total


def mean_accuracy(pred_masks, target_masks, n_classes=3):
    acc = accuracy(pred_masks, target_masks, n_classes)
    mean_acc = acc / n_classes
    return mean_acc


def validate_model(loader, model, device="cuda"):
    """Calculate the accuracy of the model.

    Args:
        loader (_type_): Data loader
        model (_type_): NN model
        device (str, optional): GPU or CPU. Defaults to "cuda".
    """

    accuracy_scores = 0
    dice_scores = 0
    miou_scores = 0
    pred_scores = 0
    model.eval()

    with torch.no_grad():
        loop = tqdm(loader)
        for idx, (images, targets) in enumerate(loop):
            # Move images and target to device (cpu or cuda)
            name = loader.sampler.data_source.imgs[idx]
            images = list(image.to(device).double() for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            pred = model(images)[0]

            # Get the masks that have confidence level bigger than the threshold
            threshold = 0.5
            scores = pred["scores"]
            pred_scores = scores.mean().item()
            indexes = (scores > threshold).nonzero().squeeze()
            pred_masks = pred["masks"][indexes]
            target_masks = targets[0]["masks"]

            accuracy_scores += mean_accuracy(
                pred_masks=pred_masks, target_masks=target_masks
            )
            miou_scores += compute_miou(
                pred_masks=pred_masks, target_masks=target_masks
            )
            dice_scores += mean_dice_score(
                pred_masks=pred_masks, target_masks=target_masks
            )
            print()

    accuracy_score = accuracy_scores / len(loader)
    dice_score = dice_scores / len(loader)
    miou_score = miou_scores / len(loader)
    pred_score = pred_scores / len(loader)

    print(f"Accuracy score: {accuracy_score}")
    print(f"Dice score: {dice_score}")
    print(f"mIoU score: {miou_score}")
    print(f"Pred score: {pred_score}")

    model.train()
    return accuracy_score, dice_score, miou_score, pred_score


def save_metric_scores(
    loss_epoch_mean,
    iter_loss,
    accuracy_score,
    dice_score,
    miou_score,
    pred_score,
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


def plot_loss_graph(losses):
    """Plot the loss graph.

    Args:
        losses (_type_): the array comprising of the loss values
    """
    plt.plot(range(len(losses)), losses)
    plt.savefig("results/losses.png")
