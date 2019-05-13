from abc import abstractmethod
import torch
import time


class Model:

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def get_learnable_parameters(self):
        pass

    def save(self, path='./weights/'):
        model = self.get_model()

        time_str = str(time.time_ns())

        full_path = path+self.name+'-'+time_str+'.pt'

        torch.save(model.state_dict(), full_path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def get_model_on_device(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return self.get_model().to(device)
