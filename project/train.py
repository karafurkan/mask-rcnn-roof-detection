from torchvision import transforms
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import torch
import numpy as np
from tqdm import tqdm
import project.utilities.utils as utils
import project.utilities.metrics as metric_utils
import project.utilities.visualization as vis_utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_LAYER = 256


def train(
    model, train_loader, val_loader, optimizer, num_classes, cp_epoch=0, n_epochs=10
):
    # Perform training loop for n epochs

    model.train()
    model = model.double()
    scaler = torch.cuda.amp.GradScaler()

    miou_list = []
    pred_scores_list = []
    f1_scores_list = []

    # Validate model before training
    miou_scores, pred_scores, f1_scores, val_score = utils.validate_model(
        val_loader, model, device=device, num_classes=num_classes
    )
    miou_list.append(miou_scores)
    pred_scores_list.append(pred_scores)
    f1_scores_list.append(f1_scores)
    print("F1 scores: ", f1_scores)
    print("Validation loss: ", val_score)
    # Start training loop
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

        miou_scores, pred_scores, f1_scores, val_score = utils.validate_model(
            val_loader, model, device=device, num_classes=num_classes
        )
        print("Validation loss: ", val_score)
        miou_list.append(miou_scores)
        pred_scores_list.append(pred_scores)
        f1_scores_list.append(f1_scores)
        print("F1 scores: ", f1_scores)
        # Save model
        utils.save_checkpoint(
            model.state_dict(), f"hl_{HIDDEN_LAYER}/cp_{epoch+cp_epoch}.pth.tar"
        )

    # path = "validation_results/f1_score.npy"
    # np.save(path, np.asarray(f1_scores_list))

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


if __name__ == "__main__":

    train_images_root = "project/dataset/train"
    val_images_root = "project/dataset/val"
    batch_size = 2
    num_classes = 10
    cp_epoch = None

    image_transforms = transforms.Compose(
        [
            # transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ]
    )
    mask_transforms = transforms.Compose([transforms.Resize((512, 512))])

    train_loader, val_loader = utils.get_loaders(
        train_images_root,
        val_images_root,
        batch_size,
        resize=None,  # (512, 512),
        img_transforms=image_transforms,
        mask_transforms=None,
    )

    # Create the model from scratch if there is no checkpoint
    if cp_epoch is None:
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

        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, HIDDEN_LAYER, num_classes
        )
        cp_epoch = 0
    else:
        print(f"Loading checkpoint from epoch {cp_epoch}...")
        cp_path = f"project/checkpoints/hl_{HIDDEN_LAYER}/cp_{cp_epoch}.pth.tar"
        model = utils.load_model(
            num_classes=num_classes,
            hidden_layer=HIDDEN_LAYER,
            device=device,
            cp_path=cp_path,
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
        num_classes=num_classes,
        cp_epoch=cp_epoch,
        n_epochs=20,
    )
