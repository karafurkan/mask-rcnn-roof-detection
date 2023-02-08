import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
from dataset import ImageDataset
from torch.utils.data import DataLoader

import random as rng

rng.seed(12345)


def save_checkpoint(state, filename="checkpoints/my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    """Get the train and validation data loaders.

    Args:
        train_dir (_type_): _description_
        train_maskdir (_type_): _description_
        val_dir (_type_): _description_
        val_maskdir (_type_): _description_
        batch_size (_type_): _description_
        train_transform (_type_): _description_
        val_transform (_type_): _description_
        num_workers (int, optional): _description_. Defaults to 4.
        pin_memory (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    train_ds = ImageDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = ImageDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def get_test_loader(
    test_dir,
    batch_size,
    test_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = ImageDataset(
        image_dir=test_dir,
        mask_dir=None,
        transform=test_transform,
    )

    test_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return test_loader


def check_accuracy(loader, model, device="cuda"):
    """Calculate the accuracy of the model.

    Args:
        loader (_type_): Data loader
        model (_type_): NN model
        device (str, optional): GPU or CPU. Defaults to "cuda".
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()


def make_prediction(loader, model, folder="test_images", device="cuda"):
    """Make prediction with the test data.

    Args:
        loader (_type_): Test data loader
        model (_type_): NN Model
        folder (str, optional): Test image folder path. Defaults to "test_images".
        device (str, optional): GPU or CPU. Defaults to "cuda".
    """
    model.eval()
    for idx, x in enumerate(loader):
        x = x.to(device=device)
        name = loader.sampler.data_source.images[idx]
        resolution = loader.sampler.data_source.resolutions[idx]

        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            # preds = (preds > 0.5).float() # binary classification

            preds = preds.squeeze(0)
            preds = torch.permute(preds, (1, 2, 0))
            preds = torch.argmax(preds, dim=2)

        if ".jpg" in name:
            filename = name[:-4] + ".png"
        else:
            filename = name

        # torchvision.utils.save_image(
        #     preds.squeeze(0), f"{folder}/pred_{filename}"
        # )

        im = preds.squeeze(0).squeeze(0).numpy()
        im = im * 16
        im = np.trunc(im)
        cv2.imwrite(f"{folder}/pred_{filename}", im)

        find_contours(
            img_path=f"data/test_images/{name}",
            masked_img_path=f"{folder}/pred_{filename}",
            resolution=resolution,
            show_image=False,
        )

        # apply_mask_to_rgb_image(
        #     img_path=f"data/test_images/{name}",
        #     masked_img_path=f"{folder}/pred_{filename}",
        # )


def apply_mask_to_rgb_image(img_path, masked_img_path):
    image = cv2.imread(img_path)
    masked_image = cv2.imread(masked_img_path)
    result = cv2.bitwise_and(image, image, mask=masked_image)

    filename = img_path.split("/")[-1]
    cv2.imwrite(f"results/masked/{filename}", result)


def plot_loss_graph(losses):
    """Plot the loss graph.

    Args:
        losses (_type_): the array comprising of the loss values
    """
    plt.plot(range(len(losses)), losses)
    plt.savefig("results/losses.png")


def create_text_on_image(image, text, font_scale=1, pos=(30, 30)):
    """Creates a text with the given properties for cv2 images.

    Args:
        image (_type_): Image to add the text
        text (_type_): Text value to add
        font_scale (int, optional): Fon scale. Defaults to 1.
        pos (tuple, optional): Position of the text on image. Defaults to (30,30).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (0, 255, 0)
    thickness = 1
    line_type = 2

    cv2.putText(image, text, pos, font, font_scale, font_color, thickness, line_type)


def find_contours(img_path, masked_img_path, resolution, show_image=False):
    """Find the contours in the semantically segmented image.

    Args:
        img_path (_type_): Image path
        masked_img_path (_type_): Masked image path (semantically segmented)
        resolution (_type_): Resolution of the image
        show_image (bool, optional): Flag to open the image. Defaults to False.
    """
    image = cv2.imread(img_path)
    masked_image = cv2.imread(masked_img_path)

    masked_image = cv2.resize(masked_image, resolution)
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Find Canny edges
    edged = cv2.Canny(blurred, 30, 200)

    contours, hierarchy = cv2.findContours(
        edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    text = "Number of buildings found = " + str(len(contours))

    create_text_on_image(image, text=text, font_scale=0.75)

    for contour in contours:
        area = cv2.contourArea(contour)

        # Create a rectangular box around the contour

        # rect = cv2.minAreaRect(contour)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

        # text_position = (contour[0][0][0], contour[0][0][1])
        # create_text_on_image(image, text=f"{area}", font_scale=0.5, pos=text_position)

    # Draw all contours
    # -1 signifies drawing all contours
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

    filename = img_path.split("/")[-1]
    cv2.imwrite(f"results/contours/{filename}", image)

    if show_image:
        cv2.imshow("Contours", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
