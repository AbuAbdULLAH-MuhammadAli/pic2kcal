import torchvision.models as models
import torch.nn as nn

from nn.models.model import Model
from nn.dataset import class_count

class ResNet50(Model):
    def __init__(self):
        super().__init__("ResNet50-kcal")

    def get_model(self):
        self.model = models.resnet50(pretrained=True)
        # self.model = models.resnet18(pretrained=True)

        # freeze first layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = self.model.fc.in_features

        # 1 output neuron to predict kcal
        self.model.fc = nn.Linear(num_ftrs, class_count)  # nn.Sequential(, nn.Softmax(dim=25))
        # self.model.softmax = nn.Softmax(dim=25)

        return self.model

    # def get_learnable_parameters(self):
    #    return self.model.fc.parameters()
