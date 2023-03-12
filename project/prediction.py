import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
import project.utilities.utils as utils
import project.utilities.visualization as vis_utils
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reduced_classes = ["Background", "Flat", "Gable"]  # 0  # 1 # 3
reduced_class_names = {i: class_name for i, class_name in enumerate(reduced_classes)}


def load_model(num_classes, hidden_layer, cp_path):
    """Load the model from the checkpoint

    Args:
        num_classes (_type_): number of classes in the dataset
        hidden_layer (_type_): number of hidden layers in the model
        cp_path (_type_): path to the checkpoint

    Returns:
        _type_: PyTorch model
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one with the number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # Replace the pre-trained head with a new one with the number of classes
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
            _pred = model([test_image])

            scores = _pred[0]["scores"]
            indexes = (scores > score_threshold).nonzero().squeeze()

            # Get the data which has bigger score value than the threshold
            pred = dict()
            pred["boxes"] = _pred[0]["boxes"][indexes]
            pred["masks"] = _pred[0]["masks"][indexes]
            labels = _pred[0]["labels"][indexes]

            # If no boxes found, skip the rendering boxes and mask process
            if len(pred["boxes"]) == 0:
                print(f"Skipping image: {image_idx}")
                continue

            # If the indexes is a scalar, unsqueeze the result tensors
            if indexes.shape == torch.Size([]):
                pred["boxes"] = pred["boxes"].unsqueeze(0)
                pred["masks"] = pred["masks"].unsqueeze(0)
                labels = labels.unsqueeze(0)

            masks = pred["masks"].numpy().squeeze(1)

            # Combine the pred masks into one mask for visualization
            combined_pred_mask = np.zeros(masks[0].shape, dtype=np.uint8)
            for i, (mask, label) in enumerate(zip(masks, labels)):
                combined_pred_mask[mask > 0.5] = label

            # Combine the target masks into one mask for visualization
            combined_target_mask = np.zeros(target["masks"][0].shape, dtype=np.uint8)
            for i, (mask, label) in enumerate(zip(target["masks"], labels)):
                combined_target_mask[mask > 0.5] = label

            vis_utils.blend_image_masks(
                test_image,
                combined_pred_mask,
                combined_target_mask,
                f"project/results/{image_idx}.png",
            )


if __name__ == "__main__":
    num_classes = 3
    hidden_layer = 512
    cp_path = f"project/checkpoints/hl_{hidden_layer}/cp_49.pth.tar"

    test_images_root = "project/dataset/test/"
    _, test_loader = utils.get_loaders(
        None, test_images_root, num_workers=0, batch_size=1, resize=False
    )

    model = load_model(
        num_classes=num_classes, hidden_layer=hidden_layer, cp_path=cp_path
    )
    predict(model, test_loader, num_classes=num_classes, score_threshold=0.75)
