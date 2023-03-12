import numpy as np
import pickle
import cv2
from tqdm import tqdm
import os
from os.path import isfile, join

PATH_IMAGE_PKL = (
    "/home/furkan/Projects/master_project/mask_rcnn/pkl_files/images/"
)
PATH_MASK_PKL = (
    "/home/furkan/Projects/master_project/mask_rcnn/pkl_files/masks/"
)

IMG_DESTINATION = (
    "/home/furkan/Projects/master_project/mask_rcnn/dataset/images"
)
MASK_DESTINATION = (
    "/home/furkan/Projects/master_project/mask_rcnn/dataset/masks"
)


def process_images(filename="images.pkl", counter=0):
    images = np.load(PATH_IMAGE_PKL + filename, allow_pickle=True)

    for idx, image in enumerate(tqdm(images)):
        cv2.imwrite(f"{IMG_DESTINATION}/image_{counter}.png", image)
        counter += 1

    return counter


def process_masks(filename="masks.pkl", counter=0):
    images = np.load(PATH_MASK_PKL + filename, allow_pickle=True)

    for idx, image in enumerate(tqdm(images)):

        mask = image[:] == [0]



        # cv2.imwrite(f"{MASK_DESTINATION}/image_{counter}.png", image)
        counter += 1

    return counter


if __name__ == "__main__":

    im = cv2.imread("/home/furkan/Projects/master_project/mask_rcnn/dataset/images/image_53.png")
    mask = cv2.imread("/home/furkan/Projects/master_project/mask_rcnn/dataset/masks/image_53.png", 0)

    cv2.imshow('image window', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('image window', mask*20)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # image_files = sorted(
    #     [f for f in os.listdir(PATH_IMAGE_PKL) if isfile(join(PATH_IMAGE_PKL, f))]
    # )

    # img_counter = 0
    # mask_counter = 0

    # for image_file in image_files:
    #     mask_file = "masks_" + image_file.split("_")[1]
    #     # print(f"Processing image file: {image_file}")
    #     # img_counter = process_images(filename=image_file, counter=img_counter)
    #     # print(f"Processing mask file: {mask_file}")
    #     mask_counter = process_masks(filename=mask_file, counter=mask_counter)
