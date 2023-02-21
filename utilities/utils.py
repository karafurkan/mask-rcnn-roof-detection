import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
from dataset import ImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import random as rng
from utilities.metrics import *

rng.seed(12345)


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
    resize=True,
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
            train_images_root,
            transforms=torchvision.transforms.ToTensor(),
            resize=resize,
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
        val_images_root, transforms=torchvision.transforms.ToTensor(), resize=resize
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


def validate_model(loader, model, device="cuda"):
    """Calculate the accuracy of the model.

    Args:
        loader (_type_): Data loader
        model (_type_): NN model
        device (str, optional): GPU or CPU. Defaults to "cuda".
    """
    print("Validating model...")
    threshold = 0.5
    accuracy_scores = 0
    dice_scores = 0
    miou_scores = 0
    f1_scores = 0
    pred_scores = 0
    torch.cuda.empty_cache()
    model.to(device)
    model.eval()
    with torch.no_grad():
        loop = tqdm(loader)
        for idx, (images, targets) in enumerate(loop):
            # Move images and target to device (cpu or cuda)
            images = list(image.to(device).double() for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            pred = model(images)[0]
            scores = pred["scores"]

            pred_scores += scores.mean()
            # Get predictions above threshold
            indexes = (scores > threshold).nonzero().squeeze()
            pred_masks = pred["masks"][indexes]
            target_masks = targets[0]["masks"]

            # accuracy_scores += mean_accuracy(
            #     pred_masks=pred_masks, target_masks=target_masks
            # )
            miou_scores += compute_miou(
                pred_masks=pred_masks, target_masks=target_masks, n_classes=2
            )
            f1_scores += compute_f1_score(
                pred_masks=pred_masks, target_masks=target_masks, n_classes=2
            )
            # dice_scores += mean_dice_score(
            #     pred_masks=pred_masks, target_masks=target_masks
            # )

    # accuracy_score = accuracy_scores / len(loader)
    # dice_score = dice_scores / len(loader)
    miou_score = miou_scores / len(loader)
    pred_score = (pred_scores / len(loader)).detach().cpu().numpy()
    f1_score = (f1_scores / len(loader)).detach().cpu().numpy()

    # print(f"Accuracy score: {accuracy_score}")
    # print(f"Dice score: {dice_score}")
    print(f"mIoU score: {miou_score}")
    print(f"F1 score: {f1_score}")
    print(f"Pred score: {pred_score}")

    model.train()
    torch.cuda.empty_cache()
    return miou_score, pred_score, f1_score


def plot_loss_graph(losses):
    """Plot the loss graph.

    Args:
        losses (_type_): the array comprising of the loss values
    """
    plt.plot(range(len(losses)), losses)
    plt.savefig("results/losses.png")
