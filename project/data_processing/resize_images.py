from os.path import isfile, join
from os import listdir
import cv2


def resize_images(path_to_read, path_to_write, new_size):
    image_files = [
        f
        for f in listdir(path_to_read + "/images")
        if isfile(join(path_to_read + "/images", f))
    ]

    for image_name in image_files:
        image = cv2.imread(path_to_read + "/images/" + image_name)
        mask = cv2.imread(path_to_read + "/masks/" + image_name)

        image = cv2.resize(image, new_size)
        mask = cv2.resize(mask, new_size)

        cv2.imwrite(
            path_to_write + "/images/" + image_name,
            image,
        )
        cv2.imwrite(
            path_to_write + "/masks/" + image_name,
            mask,
        )


if __name__ == "__main__":
    path_to_read = "/home/furkan/Projects/master_project/mask-rcnn-roof-detection/project/dataset/train"
    path_to_write = "/home/furkan/Projects/master_project/mask-rcnn-roof-detection/project/dataset/train"
    resize_images(path_to_read, path_to_read, (512, 512))
