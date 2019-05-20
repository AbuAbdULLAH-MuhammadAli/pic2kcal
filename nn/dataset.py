from torch.utils.data import Dataset
from skimage import io, transform
import json
import os

ROOT = "./data/"


class ImageCaloriesDataset(Dataset):

    def __init__(self, calories_file, image_dir, transform=None):

        with open(ROOT + calories_file) as json_file:
            self.calorie_image_tuples = json.load(json_file)["data"]
        self.image_dir = ROOT + image_dir
        self.transform = transform

    def __len__(self):
        return len(self.calorie_image_tuples)

    def __getitem__(self, idx):
        element = self.calorie_image_tuples[idx]

        img_name = os.path.join(self.image_dir, element['name'])

        image = io.imread(img_name)
        kcal = element['kcal']

        return {'image': image, 'kcal': kcal}
