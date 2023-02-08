import os
from os.path import isfile, join

PATH = "dataset/train/images"
image_files = sorted(
    [f for f in os.listdir(PATH) if isfile(join(PATH, f))]
)

for idx, file in enumerate(image_files):
    if idx > 2000:
        path = f"dataset/train/images/{file}"
        print(path)
        os.remove(path)

#####

PATH = "dataset/train/masks"
image_files = sorted(
    [f for f in os.listdir(PATH) if isfile(join(PATH, f))]
)

for idx, file in enumerate(image_files):
    if idx > 2000:
        path = f"dataset/train/masks/{file}"
        print(path)
        os.remove(path)

#####

PATH = "dataset/val/images"
image_files = sorted(
    [f for f in os.listdir(PATH) if isfile(join(PATH, f))]
)

for idx, file in enumerate(image_files):
    if idx > 70:
        path = f"dataset/val/images/{file}"
        print(path)
        os.remove(path)
#####

PATH = "dataset/val/masks"
image_files = sorted(
    [f for f in os.listdir(PATH) if isfile(join(PATH, f))]
)

for idx, file in enumerate(image_files):
    if idx > 70:
        path = f"dataset/val/masks/{file}"
        print(path)
        os.remove(path)
#####

PATH = "dataset/test/images"
image_files = sorted(
    [f for f in os.listdir(PATH) if isfile(join(PATH, f))]
)

for idx, file in enumerate(image_files):
    if idx > 200:
        path = f"dataset/test/images/{file}"
        print(path)
        os.remove(path)
#####

PATH = "dataset/test/masks"
image_files = sorted(
    [f for f in os.listdir(PATH) if isfile(join(PATH, f))]
)

for idx, file in enumerate(image_files):
    if idx > 200:
        path = f"dataset/test/masks/{file}"
        print(path)
        os.remove(path)