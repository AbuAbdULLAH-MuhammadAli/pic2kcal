from nn.dataset import ImageDataset
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

# https://github.com/microsoft/ptvsd/issues/943
import multiprocessing

from nn.models.res_nutritional_net50 import ResNutritionalNet50

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
    args = parser.parse_args()
    batch_size = 2
    epochs = 1
    shuffle = True
    validate_every = 100
    validate_batches = 50
    show_img_count = 16

    logdir = (
        "runs/"
        + datetime.datetime.now().replace(microsecond=0).isoformat().replace(":", ".")
        + "-"
        + args.runname
    )
    writer = SummaryWriter(logdir)
    print(f"tensorboard logdir: {writer.log_dir}")

    model = ResNutritionalNet50()


    net = model.get_model_on_device()
    device = model.ged_device()

    train_dataset = ImageDataset("train.json", "train")
    val_dataset = ImageDataset("val.json", "val")

    optimizer = optim.Adam(net.parameters())
    criterion = nn.CrossEntropyLoss()
    __criterion_l1_loss = nn.L1Loss()

    def criterion_l1_loss(a, b):
        ax = a.argmax(1).float()
        bx = b.float()

        # ax[torch.isnan(ax)] = 0
        # bx[torch.isnan(bx)] = 0
        return __criterion_l1_loss(ax, bx) * 50

    gpu = torch.device("cuda:0")
    trainable_params, total_params = count_parameters(net)
    print(f"Parameters: {trainable_params} trainable, {total_params} total")
    running_losses = defaultdict(list)
    batch_idx = 0
    for epoch in range(1, epochs + 1):
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4
        )

        for epoch_batch_idx, data in enumerate(train_loader, 0):
            # print(data["kcal"].shape)
            # print(data["kcal"].squeeze().shape)
            # print("sq2", data["kcal"].squeeze())
            batch_idx += 1
            image_ongpu = data["image"].to(device)
            optimizer.zero_grad()

            outputs = net(image_ongpu)

            # print("out", outputs.shape)

            kcal = data["kcal"].squeeze().to(device)
            loss = criterion(outputs, kcal)
            l1_loss = criterion_l1_loss(outputs, kcal)

            loss.backward()
            optimizer.step()

            running_losses["loss"].append(float(loss.item()))
            running_losses["l1"].append(float(l1_loss.item()))

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
                        kcal_i = data["kcal"].squeeze()
                        kcal = kcal_i.to(device)

                        output = net(image)
                        val_error["loss"].append(criterion(output, kcal).item())
                        l1_loss = criterion_l1_loss(output, kcal)

                        truth, pred = (
                            kcal_i.numpy(),
                            torch.argmax(output.cpu(), 1).numpy(),
                        )
                        val_error["l1"].append(float(l1_loss.item()))
                    # only run this on last batch from val loop
                    images_cpu = (
                        image.view(-1, 3, 224, 224)[:show_img_count].cpu().numpy()
                    )

                    images_cpu = put_text(
                        images_cpu,
                        [
                            f"truth: {t*50}kcal, pred: {p*50}kcal"
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
