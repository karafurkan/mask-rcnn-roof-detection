import numpy as np
import cv2
import os
from os.path import isfile, join
from tqdm import tqdm

IMAGE_DIR = "data/train_images/"
TARGET_DIR = "images/"


def preprocess(folder):
    """Load the annotated images and return images with the file path.

    Args:
        folder (_type_): Image folder comprised of annotated images.

    Returns:
        tuple(np_array, [string]): images and file paths
    """

    filenames = sorted(
        [f for f in os.listdir(IMAGE_DIR) if isfile(join(folder, f))]
    )  # noqa

    counter = 6074

    for filename in tqdm(filenames):
        img = cv2.imread(join(folder, filename))
        if img is not None:
            counter = split_images_into_pieces(img, filename, fraction=4, counter=counter)


def split_images_into_pieces(img, filename, fraction, counter):
    """_summary_

    Args:
        images (_type_): _description_
        filenames (_type_): _description_
    """
    M = int(img.shape[0] // fraction)
    N = int(img.shape[1] // fraction)

    tiles = [
        img[x : x + M, y : y + N]
        for x in range(0, img.shape[0], M)
        for y in range(0, img.shape[1], N)
    ]

    for idx, tile in enumerate(tiles):
        cv2.imwrite(f"{TARGET_DIR}image_{counter}.png", tile)
        counter += 1

    return counter

if __name__ == "__main__":
    preprocess(IMAGE_DIR)
