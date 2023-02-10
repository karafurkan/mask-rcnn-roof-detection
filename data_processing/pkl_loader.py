import numpy as np
import random
import cv2
from tqdm import tqdm
import os
from os.path import isfile, join
import shutil

PATH_IMAGE_PKL = (
    "pkl_files/images/"
)
PATH_MASK_PKL = (
    "pkl_files/masks/"
)

TRAIN_IMAGE_PATH = "dataset/train/images"
TRAIN_MASK_PATH = "dataset/train/masks"

VAL_IMAGE_PATH = "dataset/val/images"
VAL_MASK_PATH = "dataset/val/masks"

TEST_IMAGE_PATH = "dataset/test/images"
TEST_MASK_PATH = "dataset/test/masks"

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
    new_mask[np.where(mask == 3)] = 1

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


def create_train_val_test_split(validation_percent=5, test_percent=2):
    image_files = [f for f in os.listdir(TRAIN_IMAGE_PATH) if isfile(join(TRAIN_IMAGE_PATH, f))]
    random.shuffle(image_files)

    number_of_val_imgs = int((len(image_files) * validation_percent) / 100)
    number_of_test_imgs = int((len(image_files) * test_percent) / 100)

    # Move to validation image folder
    for idx, img in enumerate(image_files):
        # move images
        shutil.move(f"{TRAIN_IMAGE_PATH}/{img}", f"{VAL_IMAGE_PATH}/{img}")
        # move masks
        shutil.move(f"{TRAIN_MASK_PATH}/{img}", f"{VAL_MASK_PATH}/{img}")
        if idx == number_of_val_imgs:
            break

    image_files = [f for f in os.listdir(TRAIN_IMAGE_PATH) if isfile(join(TRAIN_IMAGE_PATH, f))]
    random.shuffle(image_files)
    # Move to test image folder
    for idx, img in enumerate(image_files):
        # move images
        shutil.move(f"{TRAIN_IMAGE_PATH}/{img}", f"{TEST_IMAGE_PATH}/{img}")
        # move masks
        shutil.move(f"{TRAIN_MASK_PATH}/{img}", f"{TEST_MASK_PATH}/{img}")
        if idx == number_of_test_imgs:
            break


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
        if img_counter > 7000:
            break
    create_train_val_test_split()