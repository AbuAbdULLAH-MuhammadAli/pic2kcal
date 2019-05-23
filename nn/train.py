from nn.models.res_net import ResNet
from nn.dataset import ImageCaloriesDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

plt.ion()


if __name__ == '__main__':
    batch_size = 100
    shuffle = True

    writer = SummaryWriter()

    model = ResNet()
    net = model.get_model_on_device()

    train_dataset = ImageCaloriesDataset('train.json', 'train')
    val_dataset = ImageCaloriesDataset('val.json', 'val')

    optimizer = optim.Adadelta(model.get_learnable_parameters())
    criterion = nn.MSELoss()
    gpu = torch.device('cuda:0')

    for epoch in range(1):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            image_ongpu = data['image'].to(device=gpu)
            optimizer.zero_grad()

            outputs = net(image_ongpu)
            loss = criterion(outputs, data['kcal'].to(device=gpu)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    model.save()
