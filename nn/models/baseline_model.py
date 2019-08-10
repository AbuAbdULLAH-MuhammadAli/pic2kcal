import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np

from nn.models.model import Model
#from nn.dataset import class_count, granularity

class BaselineModel(Model):
    def __init__(self):
        super().__init__("Baseline-kcal")

    def get_model(self):
        return BaselineModule()
    

class BaselineModule(nn.Module):
    def __init__(self):
        super(BaselineModule, self).__init__()
        self.noop = nn.Parameter(torch.Tensor(1))
        pass
    def forward(self, inps):
        self.noop * inps
        print(inps.shape)
        oup = torch.zeros((inps.shape[0], class_count))
        oup[:, int(np.round(378 / granularity))] = self.noop
        return oup