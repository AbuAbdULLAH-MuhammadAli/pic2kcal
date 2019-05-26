from nn.models.res_net import ResNet
from nn.dataset import ImageCaloriesDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import datetime
import torch.optim as optim
import argparse
from torch.utils.tensorboard import SummaryWriter

plt.ion()

def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runname', help='name this experiment', required=True)
    args = parser.parse_args()
    batch_size = 100
    shuffle = True
    validate_every = 1

    logdir = "runs/" + datetime.datetime.now().replace(microsecond=0).isoformat().replace(':', '.') + "-" + args.runname
    writer = SummaryWriter(logdir)
    print(f"tensorboard logdir: {writer.log_dir}")

    model = ResNet()
    net = model.get_model_on_device()
    device = model.ged_device()

    train_dataset = ImageCaloriesDataset('train.json', 'train')
    val_dataset = ImageCaloriesDataset('val.json', 'val')

    optimizer = optim.Adadelta(model.get_learnable_parameters())
    criterion = nn.MSELoss()
    gpu = torch.device('cuda:0')
    trainable_params, total_params = count_parameters(net)
    print(f"Parameters: {trainable_params} trainable, {total_params} total")

    for epoch in range(10):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            image_ongpu = data['image'].to(device)
            optimizer.zero_grad()

            outputs = net(image_ongpu)
            loss = criterion(outputs, data['kcal'].to(device))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            writer.add_scalar('loss', float(loss.item()), i)
            if (i + 1) % 10 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
            
            if (i + 1) % validate_every == 0:
                val_error = 0
                val_samples = 0
                # validation loop
                with torch.no_grad():
                    for vali, data in enumerate(val_loader):
                        val_samples += 1

                        image = data['image'].to(device)
                        kcal = data['kcal'].to(device)

                        output = net(image)

                        val_error += criterion(output, kcal).item()
                        # only single batch for now
                        break
                writer.add_scalar('val_loss', val_error / val_samples, i)
                print('[%d, %5d] val loss: %.3f' % (epoch + 1, i + 1, val_error / val_samples))

    writer.close()
    model.save()
