import torchvision.models as models
import torch.nn as nn

from nn.models.model import Model


class ResNutritionalNet50(Model):
    def __init__(self):
        super().__init__("ResNutritionalNet50")

    def get_model(self):
        self.model = models.resnet50(pretrained=True)

        # freeze first layers
        for param in self.model.parameters():
            # param.requires_grad = False
            pass

        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Linear(num_ftrs, 200)

        return self.model

