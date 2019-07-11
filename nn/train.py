from nn.dataset import ImageDataset
import math
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import datetime
import torch.optim as optim
import argparse
import numpy as np
from itertools import islice
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from nn.models.dense_net import DenseNet
from nn.models.res_net101 import ResNet101
from nn.models.res_net50 import ResNet50
from nn.models.baseline_model import BaselineModel
from pathlib import Path

# https://github.com/microsoft/ptvsd/issues/943
import multiprocessing

multiprocessing.set_start_method("spawn", True)


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def put_text(imgs, texts):
    result = np.empty_like(imgs)
    for i, (img, text) in enumerate(zip(imgs, texts)):
        from PIL import Image
        from PIL import ImageFont
        from PIL import ImageDraw

        img = Image.fromarray((img.transpose((1, 2, 0)) * 255).astype("uint8"), "RGB")
        draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype(<font-file>, <font-size>)
        font = ImageFont.truetype("/usr/share/fonts/TTF/LiberationSans-Regular.ttf", 16)
        # draw.text((x, y),"Sample Text",(r,g,b))
        draw.text((0, 0), text, (255, 255, 255), font=font)
        result[i] = (np.asarray(img).astype("float32") / 255).transpose((2, 0, 1))
        result[i] = result[i] - result[i].min() / result[i].max() - result[i].min()
    return result




def write_losses(
    *, writer, running_losses, epoch, batch_idx, epoch_batch_idx, prefix=""
):
    for loss_name, running_loss in running_losses.items():
        avg_loss = np.mean(running_loss)
        loss_name_prefixed = f"{prefix}{loss_name}"
        writer.add_scalar(loss_name_prefixed, avg_loss, batch_idx)
        print(
            "[%d, %5d] %s: %.3f"
            % (epoch, epoch_batch_idx, loss_name_prefixed, avg_loss)
        )


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runname", help="name this experiment", required=True)
    parser.add_argument("--datadir",
                        help="input data dir generated by data/split.py (contains e.g. train.json and train/",
                        required=True)
    args = parser.parse_args()
    datadir = Path(args.datadir)
    batch_size = 50
    epochs = 20
    shuffle = True
    validate_every = 100
    validate_batches = 50
    show_img_count = 16

    is_regression = True

    # regression settings
    regression_output_neurons = 1

    # classification settings
    granularity = 50
    max_val = 2500

    if is_regression:
        num_output_neurons = regression_output_neurons

    else:
        num_output_neurons = math.ceil(max_val / granularity) + 1

    logdir = (
        "runs/"
        + datetime.datetime.now().replace(microsecond=0).isoformat().replace(":", ".")
        + "-"
        + args.runname
    )
    writer = SummaryWriter(logdir)
    print(f"tensorboard logdir: {writer.log_dir}")

    model = DenseNet(num_output_neurons) # 
    print("model:", model.name)

    net = model.get_model_on_device(True)
    print(net)
    device = model.get_device()

    train_dataset = ImageDataset(datadir / "train.json", datadir / "train", is_regression, granularity)
    val_dataset = ImageDataset(datadir / "val.json", datadir / "val", is_regression, granularity)

    optimizer = optim.Adam(net.parameters())
    criterion = nn.CrossEntropyLoss() if not is_regression else nn.SmoothL1Loss()
    criterion_l1_loss_reg = nn.L1Loss()

    def criterion_l1_loss_classif(a, b):
        ax = a.argmax(1).float()
        bx = b.float()

        # ax[torch.isnan(ax)] = 0
        # bx[torch.isnan(bx)] = 0
        return criterion_l1_loss_reg(ax, bx) * granularity
    def criterion_rel_error(pred, truth):
        # https://en.wikipedia.org/wiki/Approximation_error
        ret = torch.abs(1 - pred / truth)
        ret[torch.isnan(ret)] = 0 # if truth = 0 relative error is undefined
        return torch.mean(ret)
    trainable_params, total_params = count_parameters(net)
    print(f"Parameters: {trainable_params} trainable, {total_params} total")
    running_losses = defaultdict(list)
    batch_idx = 0
    loss_fns = {
        "loss": criterion,
        "l1": criterion_l1_loss_reg if is_regression else criterion_l1_loss_classif,
        "rel_error": criterion_rel_error if is_regression else None, # todo: impl for classif
    }
    for epoch in range(1, epochs + 1):
        if epoch == 3:
            for param in net.parameters():
                param.requires_grad = True
            trainable_params, total_params = count_parameters(net)
            print('all params unfreezed')
            print(f"Parameters: {trainable_params} trainable, {total_params} total")

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4
        )

        for epoch_batch_idx, data in enumerate(train_loader, 0):
            batch_idx += 1
            image_ongpu = data["image"].to(device)
            optimizer.zero_grad()

            outputs = net(image_ongpu)

            # print("out", outputs.shape)

            kcal = data["kcal"].to(device) if is_regression else data["kcal"].squeeze().to(device)
            for loss_name, loss_fn in loss_fns.items():
                loss_value = loss_fn(outputs, kcal)
                if loss_name == "loss":
                    loss_value.backward()
                    optimizer.step()
                # technically, this is not 100% correct because it assumes all batches are the same size
                running_losses[loss_name].append(float(loss_value.item()))

            if batch_idx % validate_every == 0:
                write_losses(
                    writer=writer,
                    running_losses=running_losses,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    epoch_batch_idx=epoch_batch_idx,
                )
                running_losses = defaultdict(list)

            if batch_idx % validate_every == 0:
                val_error = defaultdict(list)
                # validation loop
                with torch.no_grad():
                    for data in islice(val_loader, validate_batches):
                        image = data["image"].to(device)
                        # print(data["image"], type(data["image"]))
                        kcal_cpu = data["kcal"] if is_regression else data["kcal"].squeeze()
                        kcal = kcal_cpu.to(device)

                        output = net(image)
                        for loss_name, loss_fn in loss_fns.items():
                            val_error[loss_name].append(float(loss_fn(output, kcal).item()))
                        
                        truth, pred = (
                            kcal_cpu.squeeze().numpy(),
                            output.cpu().squeeze().numpy() if is_regression else torch.argmax(output.cpu(), 1).numpy(),
                        )
                    # only run this on last batch from val loop (truth, pred will be from last iteration)
                    images_cpu = (
                        image.view(-1, 3, 224, 224)[:show_img_count].cpu().numpy()
                    )

                    images_cpu = put_text(
                        images_cpu,
                        [
                            (f"truth: {t:.0f}kcal, pred: {p:.0f}kcal" if is_regression else f"truth: {t*granularity}kcal, pred: {p*granularity}kcal")
                            for t, p in zip(truth, pred)
                        ],
                    )
                    writer.add_images("YOOO", images_cpu)
                write_losses(
                    writer=writer,
                    running_losses=val_error,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    epoch_batch_idx=epoch_batch_idx,
                    prefix="val_",
                )

    writer.close()
    model.save()


if __name__ == "__main__":
    with torch.autograd.detect_anomaly():
        train()
