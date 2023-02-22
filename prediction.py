import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
import utilities.utils as utils
import utilities.visualization as vis_utils
import cv2
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reduced_classes = ["Background", "Flat", "Gable"]  # 0  # 1 # 3
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


def predict(model, loader, num_classes, score_threshold=0.75):
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
                continue
                vis_utils.show([test_image, test_image, test_image], image_idx)

            test_image = test_image.float()
            test_image = test_image * 255
            test_image = test_image.to(torch.uint8)

            ##### TEST

            # for i in range(len(accepted_pred["masks"])):
            #     cv2.imshow("image window", accepted_pred["masks"][i][0].cpu().numpy())
            #     # add wait key. window waits until user presses a key
            #     cv2.waitKey(0)
            #     # and finally destroy/close all open windows
            #     cv2.destroyAllWindows()
            ######

            # Create labels from indexes
            _labels = pred[0]["labels"][indexes].tolist()

            pred_labels, pred_colors = vis_utils.create_labels_and_colors(
                reduced_class_names, _labels
            )
            final_pred = vis_utils.prepare_masks_and_boxes(
                num_classes=num_classes,
                image=test_image,
                masks=accepted_pred["masks"],
                boxes=accepted_pred["boxes"],
                labels=pred_labels,
                colors=pred_colors,
            )

            target_labels, target_colors = vis_utils.create_labels_and_colors(
                reduced_class_names, target["labels"].tolist()
            )
            final_target = vis_utils.prepare_masks_and_boxes(
                num_classes=num_classes,
                image=test_image,
                masks=target["masks"],
                boxes=target["boxes"],
                labels=target_labels,
                colors=target_colors,
            )

            result = final_pred + final_target
            vis_utils.show(
                "results",
                result,
                image_idx,
                pred_box_count=len(accepted_pred["boxes"]),
                target_box_count=len(target["boxes"]),
            )


if __name__ == "__main__":
    num_classes = 3
    hidden_layer = 256
    cp_path = "checkpoints/hl_256/cp_1.pth.tar"

    test_images_root = "dataset/val/"
    _, test_loader = utils.get_loaders(
        None, test_images_root, num_workers=0, batch_size=1, resize=False
    )

    model = load_model(
        num_classes=num_classes, hidden_layer=hidden_layer, cp_path=cp_path
    )
    predict(model, test_loader, num_classes=num_classes, score_threshold=0.75)
