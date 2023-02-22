import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import torch
import numpy as np
from tqdm import tqdm
import utilities.utils as utils
import utilities.metrics as metric_utils
import torchvision.transforms.functional as F
import utilities.visualization as vis_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_LAYER = 256


def train(model, train_loader, val_loader, optimizer, n_epochs=10):
    # Perform training loop for n epochs

    model.train()
    model = model.double()
    scaler = torch.cuda.amp.GradScaler()

    # running_loss_list = []
    # loss_epoch_mean_list = []
    # accuracy_list = []
    # dice_score_list = []
    # miou_list = []
    # pred_scores_list = []
    # f1_scores_list = []

    # # Validate model before training
    # miou_scores, pred_scores, f1_scores = utils.validate_model(
    #     val_loader, model, device=device
    # )
    # miou_list.append(miou_scores)
    # pred_scores_list.append(pred_scores)
    # f1_scores_list.append(f1_scores)

    # Star training loop
    for epoch in tqdm(range(n_epochs)):
        loss_epoch = []
        loop = tqdm(train_loader)
        for idx, (images, targets) in enumerate(loop):
            # Move images and target to device (cpu or cuda)
            images = list(image.to(device).double() for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            # losses.backward()
            # optimizer.step()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_epoch.append(losses.item())

            loop.set_postfix(loss=losses.item())

        # running_loss_list.append(loss_epoch)
        loss_epoch_mean = np.mean(loss_epoch)
        # loss_epoch_mean_list.append(loss_epoch_mean)
        print("Average loss for epoch = {:.4f} ".format(loss_epoch_mean))

        # miou_scores, pred_scores, f1_scores = utils.validate_model(
        #     val_loader, model, device=device
        # )

        # miou_list.append(miou_scores)
        # pred_scores_list.append(pred_scores)
        # f1_scores_list.append(f1_scores)
        # Save model
        utils.save_checkpoint(
            model.state_dict(), f"hl_{HIDDEN_LAYER}/cp_{epoch}.pth.tar"
        )

    # # Save metrics
    # metric_utils.save_metric_scores(
    #     loss_epoch_mean=loss_epoch_mean_list,
    #     iter_loss=running_loss_list,
    #     accuracy_score=accuracy_list,
    #     dice_score=dice_score_list,
    #     miou_score=miou_list,
    #     pred_score=pred_scores_list,
    #     f1_score=f1_scores_list,
    #     file_name=f"hl_{HIDDEN_LAYER}_",
    # )


def visualize_data(train_loader, num_classes):
    # Visualize data
    reduced_classes = ["Background", "Flat", "Gable"]  # 0  # 1 # 3
    reduced_class_names = {
        i: class_name for i, class_name in enumerate(reduced_classes)
    }

    for idx, (images, targets) in enumerate(train_loader):
        if idx == 10:
            break

        test_image = images[0].double()
        test_image = test_image.to(device="cpu")

        test_image = test_image.float()
        test_image = test_image * 255
        test_image = test_image.to(torch.uint8)

        target_labels, target_colors = vis_utils.create_labels_and_colors(
            reduced_class_names, targets[0]["labels"].tolist()
        )
        final_target = vis_utils.prepare_masks_and_boxes(
            num_classes=num_classes,
            image=test_image,
            masks=targets[0]["masks"],
            boxes=targets[0]["boxes"],
            labels=target_labels,
            colors=target_colors,
        )
        vis_utils.show(
            "test",
            final_target + final_target,
            idx,
            pred_box_count=len(targets[0]["boxes"]),
            target_box_count=len(targets[0]["boxes"]),
        )


if __name__ == "__main__":

    train_images_root = "dataset/train"
    val_images_root = "dataset/val"
    batch_size = 1
    num_classes = 3

    train_loader, val_loader = utils.get_loaders(
        train_images_root, val_images_root, batch_size, resize=True
    )

    visualize_data(train_loader, num_classes)

    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    )

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = HIDDEN_LAYER
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.Adam(params, lr=0.001)
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    torch.cuda.empty_cache()
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        n_epochs=15,
    )
