import torchvision.models
import torch.nn as nn
import torch


class PretrainedModel(torch.nn.Module):
    def __init__(
        self, num_output_neurons: int, pytorch_model: str, all_layers_trainable: bool
    ):
        super(PretrainedModel, self).__init__()
        self.name = f"{pytorch_model}-kcal"
        self.num_output_neurons = num_output_neurons
        self.pytorch_model = pytorch_model

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

    def save(self, model, run_name, path):
        full_path = path + "/" + run_name + ".pt"

        torch.save(model.state_dict(), full_path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def get_last_layer(self):
        if self.pytorch_model.startswith("resnet") or self.pytorch_model.startswith(
            "resnext"
        ):
            return "fc"
        if self.pytorch_model.startswith("densenet"):
            return "classifier"
        raise Exception("what is this model")

    def forward(self, X):
        return self.model(X)

