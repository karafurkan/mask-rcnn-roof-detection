import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
from dataset import ImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import random as rng

rng.seed(12345)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    cp_filename = "checkpoints/" + filename
    torch.save(state, cp_filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    train_images_root,
    val_images_root,
    batch_size,
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
    dataset_train = ImageDataset(
        train_images_root, transforms=torchvision.transforms.ToTensor()
    )
    dataset_val = ImageDataset(
        val_images_root, transforms=torchvision.transforms.ToTensor()
    )
    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=lambda x: list(zip(*x))
    )
    val_loader = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=lambda x: list(zip(*x))
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


def compute_iou(pred, target, n_classes=3):
    ious = []
    for cls in range(n_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds[target_inds]).long().sum().float()
        union = pred_inds.long().sum().float() + target_inds.long().sum().float() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(intersection / union)
    return ious


def compute_miou(pred, target, n_classes=3):
    ious = compute_iou(pred, target, n_classes)
    miou = torch.mean(torch.tensor(ious))
    return miou.item()


def dice_score(pred, target, eps=1e-7, n_classes=3):
    scores = []
    for cls in range(n_classes):
        pred_inds = (pred == cls).float()
        target_inds = (target == cls).float()
        intersection = (pred_inds * target_inds).sum()
        union = pred_inds.sum() + target_inds.sum()
        score = (2.0 * intersection + eps) / (union + eps)
        scores.append(score)
    return scores


def mean_dice_score(pred, target, eps=1e-7, n_classes=3):
    scores = dice_score(pred, target, eps, n_classes)
    mean_score = torch.mean(torch.tensor(scores))
    return mean_score.item()


def accuracy(pred, target, n_classes=3):
    total = 0
    correct = 0
    for cls in range(n_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        correct += (pred_inds == target_inds).long().sum().float()
        total += target_inds.long().sum().float()
    if total == 0:
        return float('nan')
    return correct / total


def mean_accuracy(pred, target, n_classes=3):
    acc = accuracy(pred, target, n_classes)
    mean_acc = torch.mean(torch.tensor(acc))
    return mean_acc.item()


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
            images = list(image.to(device).double() for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            pred = model(images)[0]
            pred["scores"]
            accuracy_scores += mean_accuracy(pred=pred, target=targets[0])
            miou_scores += compute_miou(pred=pred, target=targets[0])
            miou_scores += compute_miou(pred=pred, target=targets[0])
            print()
            # preds = (preds > 0.5).float()
            # num_correct += (preds == y).sum()
            # num_pixels += torch.numel(preds)
            
    # print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_scores/len(loader)}")
    model.train()

    return 0, dice_scores/len(loader), miou_scores/len(loader), 0



def plot_loss_graph(losses):
    """Plot the loss graph.

    Args:
        losses (_type_): the array comprising of the loss values
    """
    plt.plot(range(len(losses)), losses)
    plt.savefig("results/losses.png")
