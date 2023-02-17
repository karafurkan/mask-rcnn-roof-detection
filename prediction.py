import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
import utilities.utils as utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reduced_classes = ["Background", "Flat", "Gable"]  # 0  # 1  # 3
reduced_class_names = {i: class_name for i, class_name in enumerate(reduced_classes)}


def load_model(num_classes, hidden_layer, cp_path):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn()

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    model = model.to(DEVICE)
    model = model.double()
    model.load_state_dict(torch.load(cp_path, map_location=DEVICE))
    model.eval()

    return model


def show(imgs, index):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 15), squeeze=False)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    labels = [
        "Test input",
        "Pred Masks and Boxes",
        "Test input",
        "Target Masks and Boxes",
    ]
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        if i > 1:
            axs[1, i - 2].imshow(np.asarray(img))
            axs[1, i - 2].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            axs[1, i - 2].set_xlabel(labels[i])
        else:
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            axs[0, i].set_xlabel(labels[i])

    plt.savefig(f"results/{index}.png", dpi=300)


def create_labels_and_colors(labels):
    if type(labels) is not list:
        labels = [labels]

    for i, class_idx in enumerate(labels):
        labels[i] = reduced_class_names[class_idx]

    # Set up the colors according to the object type in the image
    colors = []
    for label in labels:
        if label == "Gable":
            colors.append("lightgreen")
        if label == "Flat":
            colors.append("blue")

    return labels, colors


def prepare_masks_and_boxes(num_classes, image, masks, boxes, is_pred_data, labels=None, colors=None):
    # Render masks on the test image

    # Check if masks and boxes are empty
    if len(masks) == 0 or len(boxes) == 0:
        return [image, image, image]

    class_dim = 0
    all_classes_masks = (
        masks.argmax(class_dim)
        == torch.arange(num_classes)[:, None, None, None]
    )
    all_classes_masks = all_classes_masks.swapaxes(0, 1)

    final_result = [
        draw_segmentation_masks(img, masks=mask, alpha=0.7)
        for img, mask in zip([image], all_classes_masks)
    ]
    final_result.insert(0, image)

    # Render boxes and labels on the test image
    if boxes.shape == torch.Size([4]):
        boxes = boxes[None]

    if is_pred_data:
        # Convert the box data from (x,y,x+w,y+h) to (xywh)
        for box in boxes:
            box[2] = box[2] - box[0]
            box[3] = box[3] - box[1]
    # Convert the box data to the format that torchvision accepts
    boxes = torchvision.ops.box_convert(boxes=boxes, in_fmt="xywh", out_fmt="xyxy")
    # Render boxes and labels on the test image
    masks_and_boxes = draw_bounding_boxes(
        final_result[1],
        boxes,
        labels=labels,
        colors=colors,
        font="LiberationMono-Regular",
        width=5,
        font_size=25,
    )
    final_result[1] = masks_and_boxes

    return final_result


def predict(model, loader, score_threshold=0.75):
    for image_idx, data in enumerate(loader):
        target = data[1][0]
        test_image = data[0][0].double()
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
            if len(accepted_pred["boxes"]) == 0:
                show([test_image, test_image, test_image], image_idx)

            test_image = test_image.float()
            test_image = test_image * 255
            test_image = test_image.to(torch.uint8)

            # Create labels from indexes
            _labels = pred[0]["labels"][indexes].tolist()

            pred_labels, pred_colors = create_labels_and_colors(_labels)
            final_pred = prepare_masks_and_boxes(num_classes=3,
                                                 image=test_image,
                                                 masks=accepted_pred["masks"],
                                                 boxes=accepted_pred["boxes"],
                                                 labels=pred_labels, colors=pred_colors,
                                                 is_pred_data=True)

            target_labels, target_colors = create_labels_and_colors(target["labels"].tolist())
            final_target = prepare_masks_and_boxes(num_classes=3,
                                                   image=test_image,
                                                   masks=target["masks"],
                                                   boxes=target["boxes"],
                                                   labels=target_labels, colors=target_colors,
                                                   is_pred_data=True)

            result = final_pred + final_target
            show(result, image_idx)


if __name__ == "__main__":
    num_classes = 3
    hidden_layer = 256
    cp_path = "checkpoints/my_checkpoint_epoch_7.pth.tar"

    test_images_root = "dataset/test/"
    _, test_loader = utils.get_loaders(
        None, test_images_root, num_workers=0, batch_size=1
    )

    model = load_model(num_classes=num_classes, hidden_layer=hidden_layer, cp_path=cp_path)
    predict(model, test_loader, score_threshold=0.55)
