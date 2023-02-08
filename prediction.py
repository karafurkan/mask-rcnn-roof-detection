import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
import numpy as np
from dataset import TestDataset
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reduced_classes = [
    "Background",  # 0
    "Flat",  # 1
    "Gable" # 3
]

reduced_class_names = {i: class_name for i, class_name in enumerate(reduced_classes)}


def load_model():
    num_classes = 3
    model = torchvision.models.detection.maskrcnn_resnet50_fpn()

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    PATH = "checkpoints/my_checkpoint_epoch_7.pth.tar"
    model = model.to(DEVICE)
    model = model.double()
    model.load_state_dict(torch.load(PATH, map_location=DEVICE))
    model.eval()

    return model


def show(imgs, index):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), figsize=(15, 15), squeeze=False)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    labels = ["Test input", "Masks blended", "ROI Boxes"]
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[0, i].set_xlabel(labels[i])
    plt.savefig(f"results/{index}.png", dpi=300)


def predict(model, loader, score_threshold=0.8):
    for image_idx, data in enumerate(loader):
        test_image = data.squeeze(0).double()
        test_image = test_image.to(device=DEVICE)

        with torch.no_grad():
            pred = model([test_image])

            scores = pred[0]["scores"]
            indexes = (scores > score_threshold).nonzero().squeeze()

            # Get the data which has bigger score value than the threshold
            accepted_pred = dict()
            accepted_pred["boxes"] = pred[0]["boxes"][indexes]
            accepted_pred["masks"] = pred[0]["masks"][indexes]

            # If no boxes found, skip the rendering boxes and mask process
            if (len(accepted_pred["boxes"]) == 0):
                show([test_image, test_image, test_image], image_idx)

            # Create labels from indexes
            _labels = pred[0]["labels"][indexes].tolist()
            if type(_labels) == int:
                _labels = [_labels]
            labels = []
            for idx in _labels:
                if type(idx) == int:
                    labels.append(reduced_class_names[idx])

            # Change the colors according to the object type in the image
            colors = []
            for label in labels:
                if label == "Gable":
                    colors.append("lightgreen")
                if label == "Flat":
                    colors.append("blue")
            
            boxes = torchvision.ops.box_convert(boxes=accepted_pred["boxes"], in_fmt="xywh", out_fmt="xyxy")
            test_image = test_image.float()
            test_image = test_image * 255
            test_image = test_image.to(torch.uint8)

            # Fix this issue with the confiedence score
            if len(boxes) != len(labels):
                continue

            # Render boxes and labels on the test image
            boxes_image = draw_bounding_boxes(test_image, boxes, labels=labels, colors=colors, font="LiberationMono-Regular", width=5, font_size=25)

            # Render masks on the test image
            num_classes = 3
            class_dim = 0
            all_classes_masks = accepted_pred["masks"].argmax(class_dim) == torch.arange(num_classes)[:, None, None, None]
            all_classes_masks = all_classes_masks.swapaxes(0, 1)

            final_result = [
                draw_segmentation_masks(img, masks=mask, alpha=0.7)
                for img, mask in zip([test_image], all_classes_masks)
            ]
            final_result.insert(0, test_image)
            final_result.insert(2, boxes_image)

            show(final_result, image_idx)


if __name__ == "__main__":
    test_root = "dataset/test/images"
    test_dataset = TestDataset(
        test_root, transforms=torchvision.transforms.ToTensor()
    )

    loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False)

    model = load_model()

    predict(model, loader)
