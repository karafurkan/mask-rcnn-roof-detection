import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
import matplotlib.pyplot as plt
import numpy as np


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
