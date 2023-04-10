import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import matplotlib.colors as mcolors
import cv2


def visualize_target_image_with_masks(image: np.array, mask: np.array, savepath: str):
    image = (image.float() * 255).to(torch.uint8)
    image = image.numpy()
    image = np.moveaxis(image, 0, -1)
    gray_image = image.copy()
    gray_image = Image.fromarray(gray_image).convert("L")
    gray_image = np.array(gray_image)

    mask = mask.astype(np.float32)
    # mask[mask == 0] = np.nan
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
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()


## Torchvision visualization


def show(folder, imgs, index, pred_box_count=0, target_box_count=0):
    """Show the images with the predicted and target masks and boxes.

    Args:
        imgs (_type_): Image
        index (_type_): Index of the image in the batch.
        pred_box_count (int, optional): Count of boxes in the prediction.
        target_box_count (int, optional): Count of boxes in the target.
    """
    plt.rcParams.update({"font.size": 22})
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 15), squeeze=False)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    labels = [
        "Test input",
        "Pred Masks and Boxes",
        "Test input",
        "Target Masks and Boxes",
    ]
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        if i > 1:
            axs[1, i - 2].imshow(np.asarray(img))
            axs[1, i - 2].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            axs[1, i - 2].set_xlabel(labels[i])
        else:
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            axs[0, i].set_xlabel(labels[i])

    axs[0, 1].text(
        0.5,
        0.5,
        f"Number of roofs: {pred_box_count}",
        color="green",
        fontsize=18,
        position=(50, 0),
    )
    axs[1, 1].text(
        0.5,
        0.5,
        f"Number of roofs: {target_box_count}",
        color="green",
        fontsize=18,
        position=(50, 0),
    )
    plt.savefig(f"{folder}/{index}.png", dpi=300)


def create_labels_and_colors(class_names, labels):
    """Create labels and colors for the masks and boxes.

    Args:
        class_names (_type_):
        labels (_type_):

    Returns:
        _type_: tuple of labels and colors
    """
    if type(labels) is not list:
        labels = [labels]

    for i, class_idx in enumerate(labels):
        labels[i] = class_names[class_idx]

    # Set up the colors according to the object type in the image
    colors = []
    for label in labels:
        if label == "Flat":
            colors.append("lightgreen")
        if label == "Gable":
            colors.append("lightblue")

    return labels, colors


def prepare_masks_and_boxes(num_classes, image, masks, boxes, labels=None, colors=None):
    """Prepare the masks and boxes for rendering.

    Args:
        num_classes (_type_): # of classes in the masks
        image (_type_): image
        masks (_type_): masks
        boxes (_type_): number of boxes
        labels (_type_, optional): _description_. Defaults to None.
        colors (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # Render masks on the test image

    # Check if masks and boxes are empty
    if len(masks) == 0 or len(boxes) == 0:
        return [image, image, image]

    class_dim = 0
    all_classes_masks = (
        masks.argmax(class_dim) == torch.arange(num_classes)[:, None, None, None]
    )
    all_classes_masks = all_classes_masks.swapaxes(0, 1)

    final_result = [
        draw_segmentation_masks(img, masks=mask, alpha=0.6)
        for img, mask in zip([image], all_classes_masks)
    ]
    final_result.insert(0, image)

    # Render boxes and labels on the test image
    if boxes.shape == torch.Size([4]):
        boxes = boxes[None]

    for box in boxes:
        box[2] = box[2] - box[0]
        box[3] = box[3] - box[1]
    # Convert the box data to the format that torchvision accepts
    boxes = torchvision.ops.box_convert(boxes=boxes, in_fmt="xywh", out_fmt="xyxy")
    # Render boxes and labels on the test image
    masks_and_boxes = draw_bounding_boxes(
        final_result[1],
        boxes,
        labels=labels,
        colors=colors,
        font="LiberationMono-Regular",
        width=5,
        font_size=25,
    )
    final_result[1] = masks_and_boxes

    return final_result
