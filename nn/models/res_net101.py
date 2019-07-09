import torchvision.models as models
import torch.nn as nn

from nn.models.model import Model


class ResNet101(Model):
    def __init__(self, num_output_neurons):
        super().__init__("ResNet101-kcal", num_output_neurons)

    def get_model(self, all_layers_trainable):
        self.model = models.resnet101(pretrained=True)
        # self.model = models.resnet18(pretrained=True)

        # freeze first layers
        if not all_layers_trainable:
            for param in self.model.parameters():
                param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = self.model.fc.in_features

        # 1 output neuron to predict kcal
        self.model.fc = nn.Linear(num_ftrs, self.num_output_neurons)  # nn.Sequential(, nn.Softmax(dim=25))
        # self.model.softmax = nn.Softmax(dim=25)

        return self.model

    # def get_learnable_parameters(self):
    #    return self.model.fc.parameters()
