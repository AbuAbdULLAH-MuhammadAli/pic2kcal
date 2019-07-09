import torchvision.models as models
import torch.nn as nn

from nn.models.model import Model

class DenseNet(Model):
    def __init__(self, num_output_neurons):
        super().__init__("DenseNet", num_output_neurons)

    def get_model(self):
        self.model = models.densenet121(pretrained=True)


        # freeze first layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = self.model.classifier.in_features

        self.model.classifier = nn.Linear(num_ftrs, self.num_output_neurons)

        return self.model

