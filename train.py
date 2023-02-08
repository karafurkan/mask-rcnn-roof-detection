import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
import numpy as np
import os
from tqdm import tqdm
from dataset import ImageDataset
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import patches
from time import time
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def view(images, labels, n=2, std=1, mean=0, idx=0):
    figure = plt.figure(figsize=(15, 10))
    images = list(images)
    labels = list(labels)
    for i in range(n):
        out = torchvision.utils.make_grid(images[i])
        inp = out.cpu().numpy().transpose((1, 2, 0))
        inp = np.array(std) * inp + np.array(mean)
        inp = np.clip(inp, 0, 1)
        ax = figure.add_subplot(2, 2, i + 1)
        ax.imshow(images[i].cpu().numpy().transpose((1, 2, 0)))
        l = labels[i]["boxes"].cpu().numpy()
        l[:, 2] = l[:, 2] - l[:, 0]
        l[:, 3] = l[:, 3] - l[:, 1]
        for j in range(len(l)):
            ax.add_patch(
                patches.Rectangle(
                    (l[j][0], l[j][1]),
                    l[j][2],
                    l[j][3],
                    linewidth=1.5,
                    edgecolor="r",
                    facecolor="none",
                )
            )

    plt.savefig(f"test/{labels[0]['image_id']}.png")


def train(model, optimizer, n_epochs=10):
    # Perform training loop for n epochs
    iter_loss = []
    loss_list = []
    model.train()
    model = model.double()
    for epoch in tqdm(range(n_epochs)):
        loss_epoch = []
        loop = tqdm(data_loader_train)
        for idx, (images, targets) in enumerate(loop):
            # Move images and target to device (cpu or cuda)
            images = list(image.to(device).double() for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            loss_epoch.append(losses.item())

            loop.set_postfix(loss=losses.item())
        iter_loss.append(loss_epoch)
        loss_epoch_mean = np.mean(loss_epoch)
        loss_list.append(loss_epoch_mean)
        print("Average loss for epoch = {:.4f} ".format(loss_epoch_mean))
        
        # Save model
        torch.save(model.state_dict(), f"checkpoints/my_checkpoint_epoch_{epoch}.pth.tar")
    np.save(f"losses/iter_loss.npy", np.asarray(iter_loss))
    np.save(f"losses/mean_loss.npy", np.asarray(loss_epoch_mean))

if __name__ == "__main__":

    images_root = "dataset/train"
    batch_size = 2
    dataset_train = ImageDataset(
        images_root, transforms=torchvision.transforms.ToTensor()
    )

    data_loader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=False, collate_fn=lambda x: list(zip(*x))
    )
    print(len(data_loader_train))

    # images, labels = next(iter(data_loader_train))
    # for batch_idx, (images, labels) in enumerate(data_loader_train):
    #     view(images=images, labels=labels, n=batch_size, std=1, mean=0, idx=batch_idx)
    # exit()
    num_classes = 3
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.001)
    # optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    train(model=model, optimizer=optimizer, n_epochs=10)
