import json
from sklearn.model_selection import train_test_split
import os
import numpy as np
from random import sample
import pandas as pd
from collections import defaultdict
from pathlib import Path
import argparse
from tqdm import tqdm
from itertools import islice

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True, choices=["usergiven", "matched"])
parser.add_argument(
    "--kcal-mode", choices=["per_portion", "per_100g", "per_recipe"], required=True
    )
parser.add_argument("--ing-count", type=int, default=50)
parser.add_argument("--out-dir", type=str, required=True)
args = parser.parse_args()
out_root = Path(args.out_dir)
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
        data = [
            ele for val, ele in zip(vals, data) if val >= filt_min and val <= filt_max
        ]
        if len(data) == l:
            break
        l = len(data)
    return (data, filt_min, filt_max)


if args.mode == "usergiven":
    inp_file = "../../data/recipes/processed_data.json"

    def get_recipe_outs(r):
        return {"kcal": r["kcal_per_portion"], "recipe_id": r["id"]}
    with open(inp_file) as f:
        data = json.load(f)
    def get_meta():
        return {}
elif args.mode == "matched":
    inp_file = "../../data/recipes/recipes_matched.jsonl"
    f = open(inp_file, encoding='utf-8')
    data = (json.loads(line) for line in tqdm(f))#islice(tqdm(f), 1000))
    with open("../../data/recipes/ingredients_common.json") as fi:
        start = 1
        stop = args.ing_count + start
        ings_common = json.load(fi)[start:stop]
    def get_recipe_outs(r):
        # print(type(r), r)
        try:
            kcal_mode = args.kcal_mode
            if kcal_mode == "per_portion" and r["portions"] < 2:
                return None
            nut = r["nutritional_values"]
            if nut is None:
                return None
            if "Kalorien" not in nut["per_portion"]:
                return None
            matched_ingredients = [
                ingredient for ingredient in r["ingredients"]
                if ingredient["type"] == "ingredient"
                and ingredient["matched"]["matched"]
            ]
            if kcal_mode == "per_100g":
                # just
                total_mass = sum(
                    ingredient["matched"]["normal"]["count"]
                    for ingredient in matched_ingredients
                )
                kcal_src = {
                    k: {**v, "Menge": v["Menge"] / total_mass * 100}
                    for k, v in nut["per_recipe"].items()
                }
            else:
                kcal_src = nut[kcal_mode]
            return {
                "kcal": kcal_src["Kalorien"]["Menge"],
                "protein": kcal_src["Protein"]["Menge"],
                "fat": kcal_src["Fett"]["Menge"],
                "carbohydrates": kcal_src["Kohlenhydrate"]["Menge"],
                "recipe_id": r["id"],
                "ingredients": [any(x for x in matched_ingredients if x["matched"]["id"] == ing_common["id"]) for ing_common in ings_common]
            }
        except Exception as e:
            print("at recipe", r["id"])
            raise e
    def get_meta():
        return {"ingredient_names": [ing["most_common_matches"][0] for ing in ings_common]}
else:
    raise Exception("noee")


#if "__len__" in data:
#    print("before removing", len(data), "recipes")

def o_iter():
    rem_no_pics = 0
    rem_no_ings = 0
    for d in data:
        if len(d["picture_files"]) == 0:
            rem_no_pics += 1
            continue
        r_out = get_recipe_outs(d)
        if r_out is None:
            rem_no_ings += 1
            continue
        yield {**d, "r_out": r_out}
    print("removed no pics", rem_no_pics)
    print("removed no ings", rem_no_ings)
data = [d for d in o_iter()]
print("after removing", len(data), "recipes")
bef_count = len(data)
data, filt_min, filt_max = filter_outliers(
    data, key=lambda r: r["r_out"]["kcal"]
)
print(f"filtering kcal to [{filt_min}, {filt_max}]")

print(f"outliers: removed {bef_count - len(data)} of {bef_count}")

random = np.random.RandomState(seed=42)
random.shuffle(data)
# train, test and val split
train, val, test = np.split(
    data, [int(0.7 * len(data)), int(0.85 * len(data))]
)

# save selected receipe names into distinct subfolders)
os.makedirs(out_root / "train", exist_ok=True)
os.makedirs(out_root / "val", exist_ok=True)
os.makedirs(out_root / "test", exist_ok=True)

output = defaultdict(list)

for dataset, dsname in ((train, "train"), (val, "val"), (test, "test")):
    for row in dataset:
        for i, img_name in enumerate(row["picture_files"]):
            # print(img_name)
            name = row["id"] + "_" + str(i) + ".jpg"
            output[dsname].append({"name": name, **get_recipe_outs(row)})
            src = "../../img/" + img_name
            dest = out_root / dsname / name
            os.symlink(src, dest)
    with open(out_root / (dsname + ".json"), "w") as file:
        json.dump({**get_meta(), "data": output[dsname]}, file, indent="\t")
