import json
from sklearn.model_selection import train_test_split
import os
import numpy as np
from random import sample
import pandas as pd
from collections import defaultdict


def filter_outliers(data: list, *, factor=2, key=lambda x: x):
    # https://www.kdnuggets.com/2017/02/removing-outliers-standard-deviation-python.html
    vals = [key(e) for e in data]
    mean = np.mean(vals)
    stddev = np.std(vals)
    filt_min = mean - factor * stddev
    filt_max = mean + factor * stddev
    return (
        [ele for val, ele in zip(vals, data) if val >= filt_min and val <= filt_max],
        filt_min,
        filt_max,
    )


# read json
with open("per_portion_data.json") as f:
    data = json.load(f)


bef_count = len(data)
data, filt_min, filt_max = filter_outliers(data, key=lambda p: p["kcal_per_portion"])
print(f"filtering kcal to [{filt_min}, {filt_max}]")

print(f"removed {bef_count - len(data)} of {bef_count}")

df = pd.DataFrame(data)

# train, test and val split
train, val, test = np.split(
    df.sample(frac=1), [int(0.7 * len(df)), int(0.85 * len(df))]
)

# save selected receipe names into distinct subfolders)
os.makedirs("train", exist_ok=True)
os.makedirs("val", exist_ok=True)
os.makedirs("test", exist_ok=True)

output = defaultdict(list)

for dataset, dsname in ((train, "train"), (val, "val"), (test, "test")):
    for idx, row in dataset.iterrows():
        for i, img_name in enumerate(row["picture_files"]):
            # print(img_name)
            name = row["id"] + "_" + str(i) + ".jpg"
            output[dsname].append({"name": name, "kcal": row["kcal_per_portion"]})
            src = "../img/" + img_name
            dest = dsname + "/" + name
            os.symlink(src, dest)
    with open(dsname + ".json", "w") as file:
        json.dump({"data": output[dsname]}, file)
