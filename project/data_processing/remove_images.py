import os
from os.path import isfile, join
from os import listdir


def remove_images():
    PATH = "project/dataset/train/images"
    image_files = sorted([f for f in os.listdir(PATH) if isfile(join(PATH, f))])

    train_img_limit = 1000
    val_img_limit = 10
    test_img_limit = 10

    for idx, file in enumerate(image_files):
        if idx > train_img_limit:
            path = f"project/dataset/train/images/{file}"
            print(path)
            os.remove(path)

    #####

    PATH = "project/dataset/train/masks"
    image_files = sorted([f for f in os.listdir(PATH) if isfile(join(PATH, f))])

    for idx, file in enumerate(image_files):
        if idx > train_img_limit:
            path = f"project/dataset/train/masks/{file}"
            print(path)
            os.remove(path)

    #####

    PATH = "project/dataset/val/images"
    image_files = sorted([f for f in os.listdir(PATH) if isfile(join(PATH, f))])

    for idx, file in enumerate(image_files):
        if idx > val_img_limit:
            path = f"project/dataset/val/images/{file}"
            print(path)
            os.remove(path)
    #####

    PATH = "project/dataset/val/masks"
    image_files = sorted([f for f in os.listdir(PATH) if isfile(join(PATH, f))])

    for idx, file in enumerate(image_files):
        if idx > val_img_limit:
            path = f"project/dataset/val/masks/{file}"
            print(path)
            os.remove(path)
    #####

    PATH = "project/dataset/test/images"
    image_files = sorted([f for f in os.listdir(PATH) if isfile(join(PATH, f))])

    for idx, file in enumerate(image_files):
        if idx > test_img_limit:
            path = f"project/dataset/test/images/{file}"
            print(path)
            os.remove(path)
    #####

    PATH = "project/dataset/test/masks"
    image_files = sorted([f for f in os.listdir(PATH) if isfile(join(PATH, f))])

    for idx, file in enumerate(image_files):
        if idx > test_img_limit:
            path = f"project/dataset/test/masks/{file}"
            print(path)
            os.remove(path)


def change_img_extensions(dir_path, old_ext, new_ext):
    # loop through all the files in the directory
    for file_name in os.listdir(dir_path):
        # check if the file has the old extension
        if file_name.endswith(old_ext):
            # construct the new file name with the new extension
            new_name = file_name.replace(old_ext, new_ext)
            # rename the file with the new name and extension
            os.rename(
                os.path.join(dir_path, file_name), os.path.join(dir_path, new_name)
            )


def remove_images_without_masks(path_to_images, path_to_masks):
    files = [f for f in listdir(path_to_images) if isfile(join(path_to_images, f))]
    images_to_remove = []

    for f in files:
        if not os.path.exists(path_to_masks + "/" + f):
            images_to_remove.append(f)

    for image in images_to_remove:
        os.remove(path_to_images + "/" + image)

    print(f"Removed {len(images_to_remove)} images without masks.")


if __name__ == "__main__":
    # path_to_images = "/home/furkan/Projects/master_project/mask-rcnn-roof-detection/project/dataset/train/images"
    # path_to_masks = "/home/furkan/Projects/master_project/mask-rcnn-roof-detection/project/dataset/train/masks"

    # remove_images_without_masks(path_to_images, path_to_masks)
    remove_images()
