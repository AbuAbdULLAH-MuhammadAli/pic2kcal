import json
from sklearn.model_selection import train_test_split
import os
import numpy as np
from random import sample
import pandas as pd
from collections import defaultdict
from pathlib import Path
out_root = Path("tomove")
if out_root.exists():
    raise Exception("out dir already exists")

def filter_outliers(data: list, *, factor=2, key=lambda x: x):
    l = len(data)
    while True:
        # https://www.kdnuggets.com/2017/02/removing-outliers-standard-deviation-python.html
        vals = [key(e) for e in data]
        mean = np.mean(vals)
        stddev = np.std(vals)
        print("mean", mean)
        print("std", stddev)
        filt_min = mean - factor * stddev
        filt_max = mean + factor * stddev
        data = [ele for val, ele in zip(vals, data) if val >= filt_min and val <= filt_max]
        if len(data) == l:
            break
        l = len(data)
    return (
        data,
        filt_min,
        filt_max,
    )


mode = "matched"
if mode == "usergiven":
    inp_file = "../../data/recipes/processed_data.json"
    def get_recipe_outs(r):
        return r["kcal_per_portion"]
elif mode == "matched":
    inp_file = "../../data/recipes/recipes_matched.json"
    def get_recipe_outs(r):
        if r["portions"] < 2:
            return None
        nut = r["nutritional_values"]
        if nut is None:
            return None
        if "Kalorien" not in nut["per_portion"]:
            return None
        return nut["per_portion"]["Kalorien"]["Menge"]
else:
    raise Exception("noee")
# read json
with open(inp_file) as f:
    data = json.load(f)

data = [d for d in data if get_recipe_outs(d) is not None and len(d["picture_files"]) > 0]
bef_count = len(data)
data, filt_min, filt_max = filter_outliers(data, key=get_recipe_outs)
print(f"filtering kcal to [{filt_min}, {filt_max}]")

print(f"removed {bef_count - len(data)} of {bef_count}")

df = pd.DataFrame(data)

# train, test and val split
train, val, test = np.split(
    df.sample(frac=1, random_state=42), [int(0.7 * len(df)), int(0.85 * len(df))]
)

# save selected receipe names into distinct subfolders)
os.makedirs(out_root / "train", exist_ok=True)
os.makedirs(out_root / "val", exist_ok=True)
os.makedirs(out_root / "test", exist_ok=True)

output = defaultdict(list)

for dataset, dsname in ((train, "train"), (val, "val"), (test, "test")):
    for idx, row in dataset.iterrows():
        for i, img_name in enumerate(row["picture_files"]):
            # print(img_name)
            name = row["id"] + "_" + str(i) + ".jpg"
            output[dsname].append({"name": name, "kcal": get_recipe_outs(row)})
            src = "../../img/" + img_name
            dest = out_root / dsname / name
            os.symlink(src, dest)
    with open(out_root / (dsname + ".json"), "w") as file:
        json.dump({"data": output[dsname]}, file)
