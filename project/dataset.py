import os
import numpy as np
import torch
from PIL import Image
import cv2

reduced_classes_3 = [
    "Background",  # 0
    "Flat",  # 1
    "Gable",  # 2
]

reduced_classes_10 = [
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

reduced_class_names = {i: class_name for i, class_name in enumerate(reduced_classes_10)}
class_numbers = [i for i in range(1, len(reduced_classes_10))]


def reduce_number_of_classes_to_3(mask):
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


def reduce_number_of_classes_to_10(mask):
    height, width = mask.shape[:2]
    new_mask = np.zeros((height, width, 1), dtype=np.uint8)
    new_mask[np.where(mask == 0)] = 0

    new_mask[np.where(mask == 1)] = 2
    new_mask[np.where(mask == 2)] = 2

    new_mask[np.where(mask == 4)] = 3
    new_mask[np.where(mask == 5)] = 3

    new_mask[np.where(mask == 6)] = 4
    new_mask[np.where(mask == 7)] = 4

    new_mask[np.where(mask == 8)] = 5
    new_mask[np.where(mask == 9)] = 5

    new_mask[np.where(mask == 10)] = 6
    new_mask[np.where(mask == 11)] = 6

    new_mask[np.where(mask == 12)] = 7
    new_mask[np.where(mask == 13)] = 7

    new_mask[np.where(mask == 14)] = 8
    new_mask[np.where(mask == 15)] = 8

    new_mask[np.where(mask == 16)] = 9
    new_mask[np.where(mask == 17)] = 9

    # Convert Flat to 1
    new_mask[np.where(mask == 3)] = 1

    return new_mask


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
    def __init__(self, root, resize, img_transforms, mask_transforms):
        self.root = root
        self.resize = resize
        self.img_transforms = img_transforms
        self.mask_transforms = mask_transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # cv2.imwrite(f"image window{img_path}", np.array(img))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if self.img_transforms is not None:
            img = self.img_transforms(img)

        if self.mask_transforms is not None:
            mask = cv2.resize(np.array(mask), (512, 512))
        else:
            mask = np.array(mask)

        # img = np.array(img)
        # mask = np.array(mask)
        # if self.resize is not None:
        #     img = cv2.resize(img, self.resize, interpolation=cv2.INTER_AREA)
        #     mask = cv2.resize(mask, self.resize)

        # img = torch.from_numpy(img).permute(2, 0, 1)

        # We want masks shape to be (height, width)
        if mask.shape[0] == 1:
            mask = np.squeeze(mask, 0)

        # mask = reduce_number_of_classes_to_10(mask).squeeze(2)
        # Uncomment to visualize the mask
        # _mask = np.expand_dims(mask, 2)
        # cv2.imwrite(f"mask_{mask_path}", _mask * 40)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        boxes, labels, masks = extract_data_from_mask(mask)

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        masks = np.array(masks, dtype=np.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id

        return img, target

    def __len__(self):
        return len(self.imgs)
