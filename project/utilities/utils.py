import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
from project.dataset import ImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import random as rng
from project.utilities.metrics import *

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
        batch_size=batch_size,
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


def combine_instance_masks(masks_list, labels_list):
    combined_mask = np.zeros(masks_list[0].shape, dtype=np.uint8)
    for mask, label in zip(masks_list, labels_list):
        combined_mask[mask > 0.5] = label
    return combined_mask


def validate_model(loader, model, device="cuda", num_classes=3):
    """Calculate the accuracy of the model.

    Args:
        loader (_type_): Data loader
        model (_type_): NN model
        device (str, optional): GPU or CPU. Defaults to "cuda".
    """
    print("Validating model...")
    threshold = 0.5  # TODO: Change threshold maybe: 0.75
    sum_f1_scores = 0

    torch.cuda.empty_cache()
    model.to(device)
    model.eval()
    with torch.no_grad():
        loop = tqdm(loader)
        for idx, (images, targets) in enumerate(loop):
            # Move images and target to device (cpu or cuda)
            images = list(image.to(device).double() for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Perform inference
            pred = model(images)

            # Collect predicted and ground truth masks and labels for each image in the batch
            combined_preds = []
            combined_targes = []

            for i in range(len(images)):
                # Get predicted masks, labels, and scores for the image
                pred_masks = pred[i]["masks"].detach().cpu().numpy().squeeze(1)
                pred_labels = pred[i]["labels"].detach().cpu().numpy()
                pred_scores = pred[i]["scores"].detach().cpu().numpy()

                # Keep only masks and labels with score above threshold
                indexes = np.nonzero(pred_scores > threshold)[0]
                pred_masks = pred_masks[indexes]
                pred_labels = pred_labels[indexes]

                # Combine masks in prediction into a single mask
                pred_mask = combine_instance_masks(pred_masks, pred_labels)
                combined_preds.append(pred_mask)

                # Get ground truth masks and labels for the image
                target_masks = targets[i]["masks"].detach().cpu().numpy()
                target_labels = targets[i]["labels"].detach().cpu().numpy()

                # Combine masks in target into a single mask
                target_mask = combine_instance_masks(target_masks, target_labels)
                combined_targes.append(target_mask)

            pred_mask = np.stack(combined_preds, axis=0)
            target_mask = np.stack(combined_targes, axis=0)

            # Calculate F1 score for each class
            f1_scores = f1_score_per_class(
                outputs=pred_mask, targets=target_mask, num_classes=num_classes
            )
            sum_f1_scores = np.add(sum_f1_scores, f1_scores)

    model.train()
    miou_score = 0
    pred_score = 0
    f1_score = sum_f1_scores / len(loader)
    return miou_score, pred_score, f1_score


def plot_loss_graph(losses):
    """Plot the loss graph.

    Args:
        losses (_type_): the array comprising of the loss values
    """
    plt.plot(range(len(losses)), losses)
    plt.savefig("results/losses.png")
