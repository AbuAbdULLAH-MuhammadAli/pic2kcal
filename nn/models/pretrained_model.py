import torchvision.models
import torch.nn as nn

from nn.models.model import Model


class PretrainedModel(Model):
    def __init__(self, num_output_neurons: int, pytorch_model: str):
        super().__init__(f"{pytorch_model}-kcal", num_output_neurons)
        self.pytorch_model = pytorch_model

    def get_last_layer(self):
        if self.pytorch_model.startswith("resnet") or self.pytorch_model.startswith("resnext"):
            return "fc"
        if self.pytorch_model.startswith("densenet"):
            return "classifier"
        raise Exception("what is this model")

    def get_model(self, all_layers_trainable):
        self.model = getattr(torchvision.models, self.pytorch_model)(pretrained=True)
        # freeze first layers
        if not all_layers_trainable:
            for param in self.model.parameters():
                param.requires_grad = False
        
        llayer = self.get_last_layer()

        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = getattr(self.model, llayer).in_features

        # 1 output neuron to predict kcal
        setattr(self.model, llayer, nn.Linear(num_ftrs, self.num_output_neurons))

        return self.model
