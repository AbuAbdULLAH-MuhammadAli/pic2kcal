import argparse
from dataset import food_image_transform
from skimage import io
from nn.train import Model, get_args, draw_val_images
import torch

args = get_args(predict=True)
indicator = "__".join(args.weights.split("/")[-2:]) + "__" + args.input_file.split("/")[-1]

model = Model(args)  #

img_tensor = food_image_transform(io.imread(args.input_file))

img_tensor = img_tensor.reshape((1, *img_tensor.shape))
with torch.no_grad():
    output = model.net(img_tensor.to(model.device))

draw_val_images(
    output=output,
    data={
        "image": img_tensor,
        "ingredients": torch.zeros(1, 100).detach(),
        "kcal": torch.zeros(1, 1).detach(),
        "protein": torch.zeros(1, 1).detach(),
        "carbohydrates": torch.zeros(1, 1).detach(),
        "fat": torch.zeros(1, 1).detach(),
        "mass_per_portion": torch.zeros(1, 1).detach(),
        "fname": [args.input_file],
    },
    ings_start_idx=model.ings_start_idx,
    image=img_tensor,
    device=model.device,
    prediction_keys=model.prediction_keys,
    ingredient_names=[f"{i}" for i in range(1, 101)],
    logdir="predict/"+indicator,
    epoch=0,
    batch_idx=0,
    epoch_batch_idx=0,
)
