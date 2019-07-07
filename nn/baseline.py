from pathlib import Path
import json
import numpy as np

dir = Path("data/kcal_given_by_user")

with open(dir / "train.json") as f:
    train = json.load(f)["data"]


kcals = [e["kcal"] for e in train]

mean = np.mean(kcals)
std = np.std(kcals)
print("mean", mean)
print("std", std)

with open(dir / "val.json") as f:
    test = json.load(f)["data"]

kcals = [e["kcal"] for e in test]
print("l1", np.mean(np.abs(kcals - mean)))