import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import matplotlib.colors as mcolors
import cv2
import torch
from torchvision import transforms
import project.utilities.utils as utils
import os
from os import listdir
from os.path import isfile, join


def visualize_target_image_with_masks(image: np.array, mask: np.array, savepath: str):
    # image = (image.float() * 255).to(torch.uint8)
    image = image.to(torch.uint8).numpy()
    image = np.moveaxis(image, 0, -1)
    gray_image = image.copy()
    gray_image = Image.fromarray(gray_image).convert("L")
    gray_image = np.array(gray_image)

    mask = mask.astype(np.float32)
    mask[mask == 0] = np.nan

    labels = {
        0: "Background",
        1: "Flat",
        2: "North",
        3: "Northeast",
        4: "East",
        5: "Southeast",
        6: "South",
        7: "Southwest",
        8: "West",
        9: "Northwest",
    }
    fig, ax = plt.subplots(1, 3, figsize=(19, 6))
    # add a pixel from each class to the mask (for consistent color mapping)
    for i in range(len(labels)):
        mask[i, 0] = i
    # create colormap for mask
    colors_all = plt.cm.viridis(np.linspace(0, 1, 256))
    # get index of colors to use
    colors_idx = np.linspace(0, 256, len(labels) - 1)
    # insert 0 at the beginning
    colors_idx = np.insert(colors_idx, 0, 0)
    # set last color to 255
    colors_idx[-1] = 255
    # get colors with index
    colors = colors_all[colors_idx.astype(int)]
    # set background to transparent
    colors[0] = (1.0, 1.0, 1.0, 0.0)
    # set flat to blue
    colors[1] = (1.0000, 0.5000, 0.7000, 1.0)

    ax[0].imshow(image)
    # set all 0 values of mask to nan
    ax[1].imshow(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cmap="gray")
    ax[1].imshow(mask, alpha=0.5, cmap=mcolors.ListedColormap(colors))
    ax[2].imshow(mask, cmap=mcolors.ListedColormap(colors))
    for axes in ax:
        axes.set_axis_off()
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
        axes.set_xticklabels([])
        axes.set_yticklabels([])
    plt.subplots_adjust(wspace=0, hspace=0)
    # create a patch (proxy artist) for every color
    patches = [
        mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))
    ]
    # put those patched as legend-handles into the legend
    plt.legend(
        handles=patches,
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.0,
        prop={"size": 6},
    )

    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, bbox_inches="tight")

    plt.close()


def visualize_data_with_data_loader(train_loader, save_folder):
    for idx, (images, targets) in enumerate(train_loader):
        if idx == 20:
            break

        test_image = images[0].double()
        test_image = test_image.to(device="cpu")

        combined_mask = np.zeros(test_image.shape[1:], dtype=np.uint8)
        for i, (mask, label) in enumerate(
            zip(targets[0]["masks"].numpy(), targets[0]["labels"].numpy())
        ):
            combined_mask[mask > 0.5] = label

        visualize_target_image_with_masks(
            test_image,
            combined_mask,
            f"{save_folder}/{idx}.png",
        )


def visualize_from_image_folder(train_path, mask_path, start_count):
    image_files = [f for f in os.listdir(train_path) if isfile(join(train_path, f))]

    for im_name in image_files:
        index = im_name.split("_")[1].split(".")[0]
        if int(index) < start_count:
            continue

        image = cv2.imread(train_path + "/" + im_name)
        mask = cv2.imread(mask_path + "/" + im_name, 0)
        image = torch.from_numpy(image)
        visualize_target_image_with_masks(image, mask, None)


if __name__ == "__main__":
    data_folder = "dataset"
    train_images_root = f"/home/furkan/Projects/master_project/mask-rcnn-roof-detection/project/{data_folder}/train"
    val_images_root = f"/home/furkan/Projects/master_project/mask-rcnn-roof-detection/project/{data_folder}/val"
    save_path = "/home/furkan/Projects/master_project/mask-rcnn-roof-detection/project/imgs_visualized_before_training"

    batch_size = 1
    num_classes = 10
    image_transforms = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ]
    )
    mask_transforms = transforms.Compose([transforms.Resize((512, 512))])

    train_loader, val_loader = utils.get_loaders(
        train_images_root,
        val_images_root,
        batch_size,
        resize=(512, 512),
        # img_transforms=image_transforms,
        # mask_transforms=None,
    )

    # visualize_data_with_data_loader(train_loader, save_folder=save_path)
    visualize_from_image_folder(
        "project/dataset/train/images",
        "project/dataset/train/masks",
        3000,
    )
