import random
import os
from os.path import isfile, join
import shutil

dataset_folder = "dataset"

TRAIN_IMAGE_PATH = f"/home/furkan/Projects/master_project/mask-rcnn-roof-detection/project/{dataset_folder}/train/images"
TRAIN_MASK_PATH = f"/home/furkan/Projects/master_project/mask-rcnn-roof-detection/project/{dataset_folder}/train/masks"

VAL_IMAGE_PATH = f"/home/furkan/Projects/master_project/mask-rcnn-roof-detection/project/{dataset_folder}/val/images"
VAL_MASK_PATH = f"/home/furkan/Projects/master_project/mask-rcnn-roof-detection/project/{dataset_folder}/val/masks"

TEST_IMAGE_PATH = f"/home/furkan/Projects/master_project/mask-rcnn-roof-detection/project/{dataset_folder}/test/images"
TEST_MASK_PATH = f"/home/furkan/Projects/master_project/mask-rcnn-roof-detection/project/{dataset_folder}/test/masks"


def create_train_val_test_split(validation_percent=20, test_percent=10):
    image_files = [
        f for f in os.listdir(TRAIN_IMAGE_PATH) if isfile(join(TRAIN_IMAGE_PATH, f))
    ]
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

    image_files = [
        f for f in os.listdir(TRAIN_IMAGE_PATH) if isfile(join(TRAIN_IMAGE_PATH, f))
    ]
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
    create_train_val_test_split(validation_percent=1, test_percent=1)
