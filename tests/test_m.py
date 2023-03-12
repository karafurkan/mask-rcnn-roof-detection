import torch
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import unittest

from project.prediction import load_model
from project.utilities import utils
from project.utilities import metrics
from project.utilities import visualization


class TestModel(unittest.TestCase):
    def test_get_loaders(self):
        train_images_root = "project/dataset/train"
        val_images_root = "project/dataset/val"
        batch_size = 2
        num_classes = 3

        train_loader, val_loader = utils.get_loaders(
            train_images_root, val_images_root, batch_size, resize=True
        )
        return True

    def test_model_output(self):
        num_classes = 3
        hidden_layer = 512
        cp_path = "project/checkpoints/hl_512/cp_49.pth.tar"
        model = load_model(
            num_classes=num_classes, hidden_layer=hidden_layer, cp_path=cp_path
        )
        # self.assertEqual(predictions, 5)
        return True


if __name__ == "__main__":
    unittest.main()
