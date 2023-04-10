import numpy as np
import random
import cv2
from tqdm import tqdm
import os
from os.path import isfile, join
import shutil

PATH_IMAGE_PKL = "/home/furkan/Desktop/pkl_files/images/"
PATH_MASK_PKL = "/home/furkan/Desktop/pkl_files/masks/"

TRAIN_IMAGE_PATH = "/home/furkan/Projects/master_project/mask-rcnn-roof-detection/project/dataset/train/images"
TRAIN_MASK_PATH = "/home/furkan/Projects/master_project/mask-rcnn-roof-detection/project/dataset/train/masks"

VAL_IMAGE_PATH = "/home/furkan/Projects/master_project/mask-rcnn-roof-detection/project/dataset/val/images"
VAL_MASK_PATH = "/home/furkan/Projects/master_project/mask-rcnn-roof-detection/project/dataset/val/masks"

TEST_IMAGE_PATH = "/home/furkan/Projects/master_project/mask-rcnn-roof-detection/project/dataset/test/images"
TEST_MASK_PATH = "/home/furkan/Projects/master_project/mask-rcnn-roof-detection/project/dataset/test/masks"


# 0: Background
# 1: East_Tilt<35
# 2: East_Tilt>35
# 3: Flat
# 4: North_Tilt<35
# 5: North_Tilt>35
# 6: Northeast_Tilt<35
# 7: Northeast_Tilt>35
# 8: Northwest_Tilt<35
# 9: Northwest_Tilt>35
# 10: South_Tilt<35
# 11: South_Tilt>35
# 12: Southeast_Tilt<35
# 13: Southeast_Tilt>35
# 14: Southwest_Tilt<35
# 15: Southwest_Tilt>35
# 16: West_Tilt<35
# 17: West_Tilt>35


reduced_classes = [
    "Background",  # 0
    "Flat",  # 1
    "East_Tilt",  # 2
    "North_Tilt",  # 3
    "Northeast_Tilt",  # 4
    "Northwest_Tilt",  # 5
    "South_Tilt",  # 6
    "Southeast_Tilt",  # 7
    "Southwest_Tilt",  # 8
    "West_Tilt",  # 9
]

reduced_class_names = {i: class_name for i, class_name in enumerate(reduced_classes)}
class_numbers = [i for i in range(1, len(reduced_classes))]


def reduce_number_of_classes(mask):
    height, width = mask.shape[:2]
    new_mask = np.zeros((height, width, 1), dtype=np.uint8)
    new_mask[np.where(mask == 0)] = 0

    new_mask[np.where(mask == 1)] = 2
    new_mask[np.where(mask == 2)] = 2

    new_mask[np.where(mask == 4)] = 2
    new_mask[np.where(mask == 5)] = 2

    new_mask[np.where(mask == 6)] = 2
    new_mask[np.where(mask == 7)] = 2

    new_mask[np.where(mask == 8)] = 2
    new_mask[np.where(mask == 9)] = 2

    new_mask[np.where(mask == 10)] = 2
    new_mask[np.where(mask == 11)] = 2

    new_mask[np.where(mask == 12)] = 2
    new_mask[np.where(mask == 13)] = 2

    new_mask[np.where(mask == 14)] = 2
    new_mask[np.where(mask == 15)] = 2

    new_mask[np.where(mask == 16)] = 2
    new_mask[np.where(mask == 17)] = 2

    # Convert Flat to 1
    new_mask[np.where(mask == 3)] = 1

    return new_mask


def process_images(filename="images.pkl", counter=0):
    images = np.load(PATH_IMAGE_PKL + filename, allow_pickle=True)

    for idx, image in enumerate(tqdm(images)):
        cv2.imwrite(f"{TRAIN_IMAGE_PATH}/image_{counter}.png", image)
        counter += 1

    return counter


def process_masks(filename="masks.pkl", counter=0):
    images = np.load(PATH_MASK_PKL + filename, allow_pickle=True)

    for idx, image in enumerate(tqdm(images)):
        image = reduce_number_of_classes(image)
        cv2.imwrite(f"{TRAIN_MASK_PATH}/image_{counter}.png", image)
        counter += 1

    return counter


if __name__ == "__main__":

    image_files = sorted(
        [f for f in os.listdir(PATH_IMAGE_PKL) if isfile(join(PATH_IMAGE_PKL, f))]
    )

    img_counter = 0
    mask_counter = 0

    for image_file in image_files:
        mask_file = "masks_" + image_file.split("_")[1]
        print(f"Processing image file: {image_file}")
        img_counter = process_images(filename=image_file, counter=img_counter)
        print(f"Processing mask file: {mask_file}")
        mask_counter = process_masks(filename=mask_file, counter=mask_counter)
        if img_counter > 9000:
            break
