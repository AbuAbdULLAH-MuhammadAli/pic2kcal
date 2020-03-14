# Transform the recipe1m(+) dataset to our own format

# vscode python notebook

# %%
import json
import numpy as np
from pprint import pprint
from pathlib import Path
from tqdm import tqdm

# %%

include_google_images = False

# %%
datadir = Path("../../data/recipe1m/dl")
l1 = json.load(open(datadir / "layer1.json"))
l2 = json.load(open(datadir / "layer2.json"))
l2p = json.load(open(datadir / "layer2+.json"))


# %%
l2[0]

# %%
nut = json.load(open(datadir / "recipes_with_nutritional_info.json"))


# %%
l1d = {l["id"]: l for l in l1}
l2d = {l["id"]: l for l in l2}
l2pd = {l["id"]: l for l in l2p}
nutd = {l["id"]: l for l in nut}

# %%

pprint("l1")
pprint(l1d[nut[0]["id"]])

pprint("l2")
pprint(l2d[nut[0]["id"]])

pprint("nut")
pprint(nut[0])

# %%


def get_imgname(partition, id):
    return f"{partition}/{id[0]}/{id[1]}/{id[2]}/{id[3]}/{id}"


def nutr_to_our(nut, multi=1):
    if "sugars" in nut:
        # long form
        nut["sug"] = nut["sugars"]
        nut["pro"] = nut["protein"]
        nut["nrg"] = nut["energy"]
    return {
        "Kohlenhydrate": {"Menge": nut["sug"] * multi, "Einheit": "g"},
        "Kalorien": {"Menge": nut["nrg"] * multi, "Einheit": "kcal"},
        "Protein": {"Menge": nut["pro"] * multi, "Einheit": "g"},
        "Fett": {"Menge": nut["fat"] * multi, "Einheit": "g"},
    }


def get_converted():
    for recipe in tqdm(l1):
        id = recipe["id"]
        l2 = l2d.get(id, {"images": []}) if not include_google_images else l2pd[id]
        nut = nutd.get(id, None)
        nuti = [(None, None, None, None, None) for i in recipe["ingredients"]]
        if nut:
            nuti = zip(
                nut["ingredients"],
                nut["quantity"],
                nut["unit"],
                nut["weight_per_ingr"],
                nut["nutr_per_ingredient"],
            )
        total_weight = sum(nut["weight_per_ingr"]) if nut else None
        yield {
            **recipe,
            "subtitle": "",
            "recipe_text": "\n".join(e["text"] for e in recipe["instructions"]),
            "picture_urls": [i["url"] for i in l2["images"]],
            "canonical_url": recipe["url"],
            "picture_files": [
                get_imgname(recipe["partition"], i["id"]) for i in l2["images"]
            ],
            "portions": 1,
            "nutritional_values": {
                "per_portion": nutr_to_our(
                    nut["nutr_values_per100g"], 1 / 100 * total_weight
                ),  # dont know portion count
                "per_recipe": nutr_to_our(
                    nut["nutr_values_per100g"], 1 / 100 * total_weight
                ),
                # per 100g is calculated in split.py
            }
            if nut
            else None,
            "ingredients": [
                {
                    "type": "ingredient",
                    "original": {"ingredient": orig["text"], "amount": "",},
                    "matched": {
                        "id": matched["text"],
                        "name": matched["text"],
                        "weird": {
                            "count": weird_amount["text"],  # TODO: to number
                            "unit": weird_unit["text"],
                        },
                        "normal": {"count": grams, "unit": "g"},
                        "matched": True,
                        "nutritional_values": nutr_to_our(nutr),
                    }
                    if matched is not None
                    else {"matched": False},
                }
                for orig, (matched, weird_amount, weird_unit, grams, nutr) in zip(
                    recipe["ingredients"], nuti
                )
            ],
        }


# %%

pprint(l2d["23a1411b2c"]["images"][7])

# %%

pprint(Counter(e["partition"] for e in l1))
# %%

with open(datadir / ".." / "recipe1m_extracted.jsonl", "w") as f:
    for l in get_converted():
        f.write(json.dumps(l))
        f.write("\n")


# %%

# print(l2pd["000018c8a5"])

# %%
