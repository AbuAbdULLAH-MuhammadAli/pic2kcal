from torch.utils.data import Dataset
from torchvision import transforms
import json
import os
import numpy as np
from skimage import io


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):

        # image = image.transpose((2, 0, 1))
        return image.convert("RGB")


def transform_data(element, is_regression=False, granularity=50):
    if is_regression:
        return np.array(
            [element], dtype=np.float32
        )
    else:
        return np.array(
            [np.floor(element / granularity)], dtype=np.int64
        )


class FoodDataset(Dataset):
    def __init__(
        self,
        calories_file,
        image_dir,
        is_regression,
        granularity,
        include_nutritional_data=False,
        include_top_ingredients=False,
        transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
                #transforms.Resize((224, 224)),
                # imageNet normalization
                transforms.ToTensor(),
                transforms.Normalize(
                   mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    ):

        with open(calories_file) as json_file:
            self.calorie_image_tuples = json.load(json_file)["data"]
        self.image_dir = image_dir
        self.transform = transform
        self.granularity = granularity
        self.is_regression = is_regression
        self.include_nutritional_data = include_nutritional_data
        self.include_top_ingredients = include_top_ingredients

    def __len__(self):
        return len(self.calorie_image_tuples)

    def __getitem__(self, idx):
        element = self.calorie_image_tuples[idx]

        img_name = os.path.join(self.image_dir, element["name"])

        image = io.imread(img_name)
        kcal = transform_data(element["kcal"], self.is_regression, self.granularity)

        sample = {"image": image, "kcal": kcal}

        if self.include_nutritional_data:
            protein = transform_data(element["protein"], self.is_regression, self.granularity)
            carbohydrates = transform_data(element["carbohydrates"], self.is_regression, self.granularity)
            fat = transform_data(element["fat"], self.is_regression, self.granularity)
            sample.update({"protein": protein, "fat": fat, "carbohydrates": carbohydrates})

        if self.include_top_ingredients:
            ingredients = element["ingredients"]
            sample.update({"ingredients": ingredients})

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample