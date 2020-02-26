from abc import abstractmethod
import torch
import time


class Model:

    def __init__(self, name, num_output_neurons):
        self.name = name
        self.num_output_neurons = num_output_neurons
        devname = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("device: ", devname)
        self.device = torch.device(devname)

    @abstractmethod
    def get_model(self, all_layers_trainable):
        pass

    @abstractmethod
    def get_learnable_parameters(self):
        pass

    def save(self, model, run_name, path='./weights/'):
        time_str = str(time.time())

        full_path = path + '/'+run_name + '.pt'

        torch.save(model.state_dict(), full_path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def get_model_on_device(self, all_layers_trainable):
        return self.get_model(all_layers_trainable).to(self.device)

    def get_device(self):
        return self.device
