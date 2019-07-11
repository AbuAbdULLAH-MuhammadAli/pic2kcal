from pathlib import Path
import json
import numpy as np

dir = Path("data/extracted_v1_per_recipe")

with open(dir / "train.json") as f:
    train = json.load(f)["data"]


kcals = [e["kcal"] for e in train]

mean = np.mean(kcals)
std = np.std(kcals)
print(f"mean: {mean:.1f}")
print(f"std: {std:.1f}")

with open(dir / "val.json") as f:
    test = json.load(f)["data"]

kcals_val = [e["kcal"] for e in test]
l1_train = np.mean(np.abs(kcals - mean))
l1_val = np.mean(np.abs(kcals_val - mean))
print(f"l1_train: {l1_train:.1f}")
print(f"l1_val: {l1_val:.1f}")