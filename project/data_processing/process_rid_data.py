import numpy as np
import cv2
import os
from os.path import isfile, join
from os import listdir
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import matplotlib.colors as colors
from tqdm import tqdm

# _labels = [
#  0   "N",
#  1   "NNE",
#  2  "NE",
#  3   "ENE",
#  4   "E",
#  5   "ESE",
#  6  "SE",
#  7   "SSE",
#  8   "S",
#  9   "SSW",
#  10   "SW",
#  11   "WSW",
#  12   "W",
#  13   "WNW",
#  14   "NW",
#  15   "NNW",
#  16   "flat",
#  17  "Background",
# ]

# labels = {
#         0: "Background",
#         1: "Flat",
#         2: "North",
#         3: "Northeast",
#         4: "East",
#         5: "Southeast",
#         6: "South",
#         7: "Southwest",
#         8: "West",
#         9: "Northwest",
#     }


def visualize_target_image_with_masks(
    image: np.array, target_mask: np.array, savepath: str
):
    # image = (image.float() * 255).to(torch.uint8)
    # image = np.moveaxis(image.numpy(), 0, -1)
    gray_image = image.copy()
    gray_image = Image.fromarray(gray_image).convert("L")
    gray_image = np.array(gray_image)

    target_mask = target_mask.astype(np.float32)
    # target_mask[target_mask == 0] = np.nan

    _labels = [
        "Background",
        "Flat",
        "N",
        "NE",
        "E",
        "SE",
        "S",
        "SW",
        "W",
        "NW",
    ]

    labels = {i: _labels[i] for i in range(len(_labels))}
    fig, ax = plt.subplots(1, 3, figsize=(19, 6))
    # First row

    for i in range(1, 3):
        target_mask[i, 0] = i

    ax[0].imshow(image)
    ax[1].imshow(gray_image, cmap="gray")
    ax[1].imshow(target_mask, alpha=0.5, cmap="viridis")
    im = ax[2].imshow(target_mask)

    for axes in ax:
        axes.set_axis_off()
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
        axes.set_xticklabels([])
        axes.set_yticklabels([])
    plt.subplots_adjust(wspace=0, hspace=0)

    patches = []
    for label in labels:
        patches.append(
            mpatches.Patch(
                color=im.cmap(im.norm(label)), label=str(label) + " - " + labels[label]
            )
        )

    # put those patched as legend-handles into the legend
    plt.legend(
        handles=patches,
        prop={"size": 8},
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.0,
    )

    plt.show()
    # plt.savefig(savepath, bbox_inches="tight")
    plt.close()


def reduce_classes_to_9(mask):
    height, width = mask.shape
    new_mask = np.zeros((height, width, 1), dtype=np.uint8)

    new_mask[np.where(mask == 0)] = 2
    new_mask[np.where(mask == 2)] = 3
    new_mask[np.where(mask == 4)] = 4
    new_mask[np.where(mask == 6)] = 5
    new_mask[np.where(mask == 8)] = 6
    new_mask[np.where(mask == 10)] = 7
    new_mask[np.where(mask == 12)] = 8
    new_mask[np.where(mask == 14)] = 9
    new_mask[np.where(mask == 16)] = 1

    new_mask[np.where(mask == 1)] = 2 if np.random.rand() < 0.5 else 3
    new_mask[np.where(mask == 3)] = 3 if np.random.rand() < 0.5 else 4
    new_mask[np.where(mask == 5)] = 4 if np.random.rand() < 0.5 else 5
    new_mask[np.where(mask == 7)] = 5 if np.random.rand() < 0.5 else 6
    new_mask[np.where(mask == 9)] = 6 if np.random.rand() < 0.5 else 7
    new_mask[np.where(mask == 11)] = 7 if np.random.rand() < 0.5 else 8
    new_mask[np.where(mask == 13)] = 8 if np.random.rand() < 0.5 else 9
    new_mask[np.where(mask == 15)] = 2 if np.random.rand() < 0.5 else 9

    return new_mask


if __name__ == "__main__":
    im_path = "/home/furkan/gd/pranet/datasets/rid/images_roof_centered_geotiff"
    mask_path = "/home/furkan/gd/pranet/datasets/rid/masks_segments_reviewed"

    local_im_path = "/home/furkan/Projects/master_project/mask-rcnn-roof-detection/project/dataset/train/images"
    local_mask_path = "/home/furkan/Projects/master_project/mask-rcnn-roof-detection/project/dataset/train/masks"

    files = [f for f in listdir(mask_path) if isfile(join(mask_path, f))]

    for f in tqdm(files):
        im = cv2.imread(im_path + "/" + f.replace(".png", ".tif"))
        mask = cv2.imread(mask_path + "/" + f, 0)
        mask = reduce_classes_to_9(mask)
        path_to_write = (
            f"/home/furkan/gd/pranet/datasets/rid/masks_reduced_9_classes/{f}"
        )
        print(f)
        print(cv2.imwrite(path_to_write, mask))
        ###
        mask_file_name = f.replace(".tif", ".png")
        print(cv2.imwrite(f"{local_im_path}/{mask_file_name}", im))
        print(cv2.imwrite(f"{local_mask_path}/{f}", mask))
        print("---------")
        # visualize_target_image_with_masks(im, mask, "")
