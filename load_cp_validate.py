import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import utilities.utils as utils
import numpy as np


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_cp(num_classes=3, hidden_layer=256, cp_name="hl_256/my_checkpoint_epoch_7.pth.tar")
    num_classes = num_classes
    model = torchvision.models.detection.maskrcnn_resnet50_fpn()

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = hidden_layer
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    PATH = "checkpoints/" + cp_name
    model = model.to(DEVICE)
    model = model.double()
    model.load_state_dict(torch.load(PATH, map_location=DEVICE))
    model.eval()

    return model


def validate_cp(model, loader, hidden_layer):
    accuracy, dice_score, miou_score, pred_score = utils.validate_model(model, loader, DEVICE)
    
    utils.save_metric_scores(
        loss_epoch_mean=None,
        iter_loss=None,
        accuracy_score=accuracy,
        dice_score=dice_score,
        miou_score=miou_score,
        pred_score=pred_score,
        file_name=f"hl_{hidden_layer}_",
    )


if __name__ == "__main__":
    hidden_layer = 256
    model = load_cp(num_classes=3, hidden_layer=hidden_layer, cp_name="hl_256/my_checkpoint_epoch_7.pth.tar")
    val_images_root = "dataset/val"
    batch_size = 1
    _, val_loader = utils.get_loaders(
        train_images_root=None, val_images_root=val_images_root, num_workers=0, batch_size=batch_size
    )

    validate_cp(model=model, loader=val_loader, hidden_layer=hidden_layer)

