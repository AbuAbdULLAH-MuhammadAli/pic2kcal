from nn.models.res_net import ResNet
from nn.dataset import ImageCaloriesDataset
if __name__ == '__main__':

    model = ResNet()
    net = model.get_model_on_device()

    dataset = ImageCaloriesDataset('data.json', 'img')

    print(net)


