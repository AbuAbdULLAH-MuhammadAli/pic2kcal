from nn.models.res_net import ResNet
from nn.dataset import ImageCaloriesDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import datetime
import torch.optim as optim
import argparse
import numpy as np
from itertools import islice
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

plt.ion()


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runname", help="name this experiment", required=True)
    args = parser.parse_args()
    batch_size = 50
    shuffle = True
    validate_every = 100
    validate_batches = 50

    logdir = (
        "runs/"
        + datetime.datetime.now().replace(microsecond=0).isoformat().replace(":", ".")
        + "-"
        + args.runname
    )
    writer = SummaryWriter(logdir)
    print(f"tensorboard logdir: {writer.log_dir}")

    model = ResNet()
    net = model.get_model_on_device()
    device = model.ged_device()

    train_dataset = ImageCaloriesDataset("train.json", "train")
    val_dataset = ImageCaloriesDataset("val.json", "val")

    optimizer = optim.Adam(net.parameters())
    criterion = nn.MSELoss()
    criterion_l1_loss = nn.L1Loss()

    gpu = torch.device("cuda:0")
    trainable_params, total_params = count_parameters(net)
    print(f"Parameters: {trainable_params} trainable, {total_params} total")
    running_losses = defaultdict(list)
    batch_idx = 0
    for epoch in range(1, 11):
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4
        )

        for i, data in enumerate(train_loader, 0):
            batch_idx += 1
            image_ongpu = data["image"].to(device)
            optimizer.zero_grad()

            outputs = net(image_ongpu)
            kcal = data["kcal"].to(device)
            loss = criterion(outputs, kcal)
            l1_loss = criterion_l1_loss(outputs, kcal)

            loss.backward()
            optimizer.step()

            running_losses["loss"].append(float(loss.item()))
            running_losses["l1"].append(float(l1_loss.item()))

            if batch_idx % validate_every == 0:
                for loss_name, running_loss in running_losses.items():
                    avg_loss = np.mean(running_loss)
                    running_losses = defaultdict(list)
                    writer.add_scalar(loss_name, avg_loss, batch_idx)
                    print("[%d, %5d] %s: %.3f" % (epoch, i, loss_name, avg_loss))

            if batch_idx % validate_every == 0:
                val_error = defaultdict(list)
                # validation loop
                with torch.no_grad():
                    for data in islice(val_loader, validate_batches):
                        image = data["image"].to(device)
                        kcal = data["kcal"].to(device)

                        output = net(image)
                        val_error["loss"].append(criterion(output, kcal).item())
                        val_error["l1"].append(float(l1_loss.item()))
                for loss_name, running_loss in val_error.items():
                    avg_val_error = np.mean(running_loss)
                    writer.add_scalar(f"val_{loss_name}", avg_val_error, batch_idx)
                    print(
                        "[%d, %5d] val %s: %.3f" % (epoch, i, loss_name, avg_val_error)
                    )

    writer.close()
    model.save()
