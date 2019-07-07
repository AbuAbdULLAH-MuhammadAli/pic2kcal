import torchvision.models as models
import torch.nn as nn

from nn.models.model import Model
from nn.dataset import class_count

class NutritionalDenseNet(Model):
    def __init__(self):
        super().__init__("NutritionalDenseNet")

    def get_model(self):
        self.model = models.densenet121(pretrained=True)


        # freeze first layers
        for param in self.model.parameters():
            # param.requires_grad = False
            pass

        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = self.model.classifier.in_features

        self.model.classifier = nn.Linear(num_ftrs, class_count)

        return self.model

