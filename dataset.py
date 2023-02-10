import os
import numpy as np
import torch
from PIL import Image
import cv2


reduced_classes = [
    "Background",  # 0
    "Flat",  # 1
    "Gable",  # 2
]

reduced_class_names = {i: class_name for i, class_name in enumerate(reduced_classes)}
class_numbers = [i for i in range(1, len(reduced_classes))]


def reduce_number_of_classes(mask):
    mask_bool = np.logical_and(mask != 0, mask != 3)
    mask[mask_bool] = 2
    mask[mask == 3] = 1

    return mask


def find_threshold(mask, id):
    height, width = mask.shape[:2]
    class_mask = np.zeros((height, width, 1), dtype=np.uint8)
    class_mask[np.where(mask == id)] = 255
    class_mask[np.where(mask != id)] = 0
    return class_mask


def extract_data_from_mask(mask_image):
    """
    Extract bounding boxes from a binary mask image

    mask_image: PyTorch tensor (height, width)

    Returns:
    boxes: list of bounding boxes, represented as (xmin, ymin, xmax, ymax)
    labels: list of class labels for each box
    """
    masks = []
    boxes = []
    labels = []

    is_no_object = True

    for class_number in class_numbers:
        binary_mask = (mask_image == class_number).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            instance_mask = np.zeros_like(binary_mask)
            cv2.drawContours(instance_mask, [contour], -1, 1, -1)

            # Reshape the masks
            # instance_mask = np.squeeze(instance_mask, 2)

            masks.append(instance_mask)
            boxes.append([x, y, x + w, y + h])
            labels.append(class_number)
            is_no_object = False

    if is_no_object:
        # Create a dummy object in the image
        instance_mask = np.zeros_like(binary_mask)
        # instance_mask = np.squeeze(instance_mask, 2)
        masks.append(instance_mask)
        boxes.append([0, 0, 0.00001, 0.00001])
        labels.append(0)

    return boxes, labels, masks


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        # img = cv2.imread(img_path)
        # mask = cv2.imread(mask_path, 0)

        mask = np.array(mask)

        mask = reduce_number_of_classes(mask)

        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)

        # convert everything into a torch.Tensor
        boxes, labels, masks = extract_data_from_mask(mask)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        # masks = torch.tensor([]) # torch.tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        # target["area"] = area
        # target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)
            # img = self.transforms(img)
            # img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class TestDataset(object):
    def __init__(self, root, transforms=None):
        self.root = root
        # self.transforms = transforms
        self.transforms = []
        if transforms is not None:
            self.transforms.append(transforms)

        self.imgs = list(sorted(os.listdir(root)))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])

        img = Image.open(img_path).convert("RGB")

        for transform in self.transforms:
            img = transform(img)

        return img.double()

    def __len__(self):
        return len(self.imgs)
