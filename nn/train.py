from nn.dataset import FoodDataset
import math
import random
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
from nn.models.pretrained_model import PretrainedModel
from nn.models.baseline_model import BaselineModel
from pathlib import Path
import json

# https://github.com/microsoft/ptvsd/issues/943
# import multiprocessing

# multiprocessing.set_start_method("spawn", True)


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
        font = ImageFont.truetype("/usr/share/fonts/TTF/LiberationSans-Regular.ttf", 12)
        # draw.text((x, y),"Sample Text",(r,g,b))
        draw.text((0, 0), text, (255, 255, 255), font=font)
        result[i] = (np.asarray(img).astype("float32") / 255).transpose((2, 0, 1))
        result[i] = (result[i] - result[i].min()) / (result[i].max() - result[i].min())
    return result


def write_imgs_to_dir(batch, dir, imgs, texts, meta):
    for i, (img, text, datapoint) in enumerate(zip(imgs, texts, meta)):
        from PIL import Image

        img = Image.fromarray((img.transpose((1, 2, 0)) * 255).astype("uint8"), "RGB")
        batchdir = dir / f"{batch:05}"
        batchdir.mkdir(parents=True, exist_ok=True)
        img.save(batchdir / f"{i:02}.png")
        (batchdir / f"{i:02}.txt").write_text(text)
        (batchdir / f"{i:02}.json").write_text(json.dumps(datapoint, indent=3))


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


def draw_val_images(
    *,
    output,
    data,
    image,
    device,
    prediction_keys,
    writer,
    ingredient_names,
    logdir,
    epoch,
    batch_idx,
    epoch_batch_idx,
):
    show_img_count = 16
    # generate and write pictures to tensorboard

    ings_pred = (torch.sigmoid(output) > 0.5).cpu().numpy()
    ings_truth = data["ingredients"].numpy()
    meta = [
        {
            "truth": {
                "ings": [
                    ingredient_names[inx].replace(",", "")
                    for inx in ing_truth.nonzero()[0]
                ]
            },
            "pred": {
                "ings": [
                    ingredient_names[inx].replace(",", "")
                    for inx in ing_pred.nonzero()[0]
                ]
            },
            "fname": fname,
        }
        for ing_truth, ing_pred, fname in zip(ings_truth, ings_pred, data["fname"])
    ]

    for pred_inx, pred_key in enumerate(prediction_keys):
        kcal_cpu = data[pred_key]

        truth, pred = (
            kcal_cpu.squeeze().numpy(),
            output[:, pred_inx : pred_inx + 1].cpu().squeeze().numpy(),
        )
        pred_key_p = "carbs" if pred_key == "carbohydrates" else pred_key
        for i, (t, p) in enumerate(zip(truth, pred)):
            meta[i]["truth"][pred_key_p] = t.astype(float)
            meta[i]["pred"][pred_key_p] = p.astype(float)

    pred_arrs = ["" for _ in meta]
    for i, m in enumerate(meta):
        text = ""
        for k in m["truth"]:
            t = m["truth"][k]
            p = m["pred"][k]
            if k == "ings":
                tru_str = ", ".join(t)
                pred_str = ", ".join(p)
                text += f"\n\nings truth: {tru_str}\nings pred: {pred_str}"
            elif k == "kcal":
                text += f"\ntruth: {t:.0f}kcal, pred: {p:.0f}kcal"
            else:
                text += f"\ntruth: {t:.0f}g {k}, pred: {p:.0f}g {k}"
        pred_arrs[i] = text
    images_cpu = image.view(-1, 3, 224, 224)[:show_img_count].cpu().numpy()
    pred_arrs = [s.strip() for s in pred_arrs]

    # write images to out dir
    imgdir = Path(logdir) / "val_images"
    write_imgs_to_dir(batch_idx, imgdir, images_cpu, pred_arrs, meta)

    # write images to tensorboard
    images_cpu = put_text(images_cpu, pred_arrs,)
    writer.add_images("val examples", images_cpu, global_step=batch_idx)


def MyLoader(datadir: str, ds: str, batch_size: int):
    # ds = train, test, val
    dataset = FoodDataset(
        calories_file=datadir / f"{ds}.json",
        image_dir=datadir / ds,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True
    )


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runname", help="name this experiment", required=True)
    parser.add_argument(
        "--datadir",
        help="input data dir generated by data/split.py (contains e.g. train.json and train/",
        required=True,
    )
    parser.add_argument(
        "--train-type",
        required=True,
        choices=["kcal", "kcal+nut", "kcal+nut+topings",],
    )
    parser.add_argument(
        "--bce-weight",
        required=True,
        type=int,
        help="set to 400 for per 100g, 2000 for per recipe, ?? for per portion",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "resnet50",
            "resnet101",
            "resnet152",
            "densenet121",
            "densenet201",
            "resnext50_32x4d",
        ],
    )
    parser.add_argument(
        "--test", required=True, choices=["train", "train+test", "test"]
    )
    parser.add_argument("--weights", required=False)
    parser.add_argument("--no-predict-portion-size", dest='predict_portion_size', required=False, default=True, type=bool, action='store_false')

    args = parser.parse_args()
    datadir = Path(args.datadir)
    batch_size = 50
    epochs = 40

    if args.test == "test":
        epochs = 0

    validate_every = 200
    validate_batches = 50

    training_type = args.train_type
    predict_portion_size = args.predict_portion_size

    # regression settings
    num_top_ingredients = 100

    def criterion_rel_error(pred, truth):
        # https://en.wikipedia.org/wiki/Approximation_error
        ret = torch.abs(1 - pred / truth)
        ret[torch.isnan(ret)] = 0  # if truth = 0 relative error is undefined
        return torch.mean(ret)

    ings_start_idx = 4
    if predict_portion_size:
        ings_start_idx = 5

    def loss_top_ingredients(pred, data):
        from torch.nn.functional import smooth_l1_loss, binary_cross_entropy_with_logits

        # todo: loop over enumerate(prediction_keys) here
        l1 = smooth_l1_loss(pred[:, 0:1], data["kcal"])
        l1 += smooth_l1_loss(pred[:, 1:2], data["protein"])
        l1 += smooth_l1_loss(pred[:, 2:3], data["fat"])
        l1 += smooth_l1_loss(pred[:, 3:4], data["carbohydrates"])
        if predict_portion_size:
            l1 += smooth_l1_loss(pred[:, 4:5], data["mass_per_portion"])
        if training_type == "kcal+nut+topings":
            # todo: adjust the 400 factor to 2000 if per recipe etc
            bce = (
                binary_cross_entropy_with_logits(pred[:, ings_start_idx:], data["ingredients"])
                * args.bce_weight
            )
            if random.random() < 0.02:
                print(
                    "l1 vs bce weight (should be around the same)",
                    float(l1),
                    float(bce),
                )
            return l1 + bce
        return l1

    loss_fns = {}

    prediction_keys = ["kcal"]

    num_output_neurons = 1

    loss_fns["loss"] = lambda pred, data: nn.functional.smooth_l1_loss(
        pred, data["kcal"]
    )
    loss_fns["l1_kcal"] = lambda pred, data: nn.functional.l1_loss(pred, data["kcal"])
    loss_fns["rel_error_kcal"] = lambda pred, data: criterion_rel_error(
        pred, data["kcal"]
    )

    if training_type.startswith("kcal+nut"):
        prediction_keys = ["kcal", "protein", "fat", "carbohydrates"]
        if predict_portion_size:
            prediction_keys += ["mass_per_portion"]
        num_output_neurons = len(prediction_keys)
        loss_fns["loss"] = loss_top_ingredients
        from torch.nn.functional import l1_loss

        def mk_loss(i, k, fn):
            def fuck_python(pred, data):
                return fn(pred[:, i : (i + 1)], data[k])

            return fuck_python

        for i, k in enumerate(prediction_keys):
            loss_fns[f"l1_{k}"] = mk_loss(i, k, l1_loss)
            loss_fns[f"rel_error_{k}"] = mk_loss(i, k, criterion_rel_error)
        i = 999
    if training_type == "kcal+nut+topings":
        num_output_neurons += num_top_ingredients

    logdir = (
        "runs/"
        + datetime.datetime.now().replace(microsecond=0).isoformat().replace(":", ".")
        + "-"
        + args.runname
    )
    writer = SummaryWriter(logdir)
    print(f"tensorboard logdir: {writer.log_dir}")

    model = PretrainedModel(num_output_neurons, pytorch_model=args.model)  #
    print("model:", model.name)

    net = model.get_model_on_device(True)
    print(net)
    device = model.get_device()

    train_loader = MyLoader(datadir, "train", batch_size)
    val_loader = MyLoader(datadir, "val", batch_size)

    optimizer = optim.Adam(net.parameters())

    trainable_params, total_params = count_parameters(net)
    print(f"Parameters: {trainable_params} trainable, {total_params} total")
    running_losses = defaultdict(list)
    batch_idx = -1

    for epoch in range(1, epochs + 1):
        if epoch == 3:
            for param in net.parameters():
                param.requires_grad = True
            trainable_params, total_params = count_parameters(net)
            print("all params unfreezed")
            print(f"Parameters: {trainable_params} trainable, {total_params} total")

        for epoch_batch_idx, data in enumerate(train_loader, 0):
            batch_idx += 1
            #print("batch idx", batch_idx)
            #if batch_idx > 100:
            #   return
            # print("is pinned", data["image"].is_pinned())
            image_ongpu = data["image"].to(device)
            optimizer.zero_grad()

            outputs = net(image_ongpu)

            # print("out", outputs.shape)

            target_data = {
                k: v.to(device) for k, v in data.items() if k not in ["image", "fname"]
            }

            for loss_name, loss_fn in loss_fns.items():
                loss_value = loss_fn(outputs, target_data)
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
                    for val_batch_inx, data in enumerate(
                        islice(val_loader, validate_batches)
                    ):
                        image = data["image"].to(device)

                        target_data = {
                            k: v.to(device)
                            for k, v in data.items()
                            if k not in ["image", "fname"]
                        }

                        output = net(image)
                        for loss_name, loss_fn in loss_fns.items():
                            val_error[loss_name].append(
                                float(loss_fn(output, target_data).item())
                            )

                        if val_batch_inx == 0:
                            # only run this on last batch from val loop
                            draw_val_images(
                                writer=writer,
                                output=output[:, ings_start_idx:],
                                data=data,
                                image=image,
                                device=device,
                                prediction_keys=prediction_keys,
                                ingredient_names=train_loader.dataset.ingredient_names,
                                logdir=logdir,
                                epoch=epoch,
                                batch_idx=batch_idx,
                                epoch_batch_idx=epoch_batch_idx,
                            )
                write_losses(
                    writer=writer,
                    running_losses=val_error,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    epoch_batch_idx=epoch_batch_idx,
                    prefix="val_",
                )

        model.save(net, f"{args.runname}-epoch-{epoch:02d}", logdir)

    writer.close()
    if args.test != "test":
        model.save(net, args.runname, logdir)

    if args.test != "train":

        if args.test == "test":
            model.load(args.weights)

        test_loader = MyLoader(datadir, "test", batch_size)

        with torch.no_grad():
            for epoch_batch_idx, data in enumerate(test_loader, 0):
                image_ongpu = data["image"].to(device)

                outputs = net(image_ongpu)

                target_data = {k: v.to(device) for k, v in data.items() if k != "image"}

                for loss_name, loss_fn in loss_fns.items():
                    loss_value = loss_fn(outputs, target_data)
                    running_losses[loss_name].append(float(loss_value.item()))

                    print(loss_value)


if __name__ == "__main__":
    with torch.autograd.detect_anomaly():
        train()
