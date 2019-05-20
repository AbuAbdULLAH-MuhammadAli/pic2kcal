# %%
import gensim
from gensim.utils import tokenize
import json
from pathlib import Path
import re
import numpy as np
import random
import heapq
from operator import itemgetter
from preprocessing.util import parse_jsonlines

# %%
data_dir = Path("data")

# model.save("oo")

# gensim.models.fasttext.FastText.load("oo", mmap="r")
print("loading out names")


# jq 'to_entries[] | .value.name=.key | .value' nutrient_data.json > nutrient_data.jsonl
# jq 'select(.Quelle | contains("ebensmittelschlüssel"))|.name' data/nutrient_data.jsonl |jq -s > data/bls_nutrient_names.json
with open(data_dir / "bls/bls_nutrient_names.json") as f:
    out_names = json.load(f)

# cat processed_data.jsonl | jq '.ingredients |.[] | select(.ingredient) | .ingredient' | jq -s unique > ingredients.jsonl
with open(data_dir / "ingredients.json") as f:
    in_names = json.load(f)


def normalize_ingredient(ing: str):
    ing = re.sub(r"\([^)]+\)", "", ing)  # remove stuff in parens
    ing = re.sub(r"(\d+,)?\d+ k?g\b", "", ing)  # remove xyz gram
    ing = re.sub(r",.*", "", ing)
    ing = re.sub(r"\bzum .*", "", ing)
    ing = ing.strip()
    return ing


in_names = list({normalize_ingredient(ing) for ing in in_names})

print(len(in_names))
# print(in_names)


# %%
print("loading model")
model = gensim.models.fasttext.load_facebook_vectors(
    data_dir / "fasttext/cc.de.300.bin.gz"
)

# %%


def normalize(v):
    return v / np.linalg.norm(v)


def get_sentence_vector(ingredient: str):
    # shitty, but they apparently did something similar https://github.com/facebookresearch/fastText/issues/323#issuecomment-353167113
    v = np.mean(
        [normalize(model.get_vector(word)) for word in tokenize(ingredient)], axis=0
    )
    return v / np.linalg.norm(v)


out_vecs = [(ingredient, get_sentence_vector(ingredient)) for ingredient in out_names]


# %%


def get_match(ingredient: str):
    ingredient = normalize_ingredient(ingredient)
    search = get_sentence_vector(ingredient)
    it = ((v[0], np.dot(v[1], search)) for v in out_vecs)
    res_list = heapq.nlargest(10, it, key=itemgetter(1))
    return res_list


for ingredient in random.sample(in_names, 100):
    res_list = get_match(ingredient)
    print(f"{ingredient} -> {res_list[0]}")

# %% load recipes

# can't stream json reading apparently in python. wtf
txt = (data_dir / "recipes" / "processed_data.jsonl").read_text()
recipe_generator = parse_jsonlines(txt)

recipes = [recipe for recipe in recipe_generator if len(recipe["picture_files"]) > 0]

# %%

for recipe in random.sample(recipes, 3):
    print("-" * 10)
    print(recipe["title"])
    for ingredient in recipe["ingredients"]:
        if "subtitle" in ingredient:
            continue
        name = ingredient["ingredient"]
        (best_match, quality), *_ = get_match(name)
        print(f"{100 * quality:.0f}% match: {name} -> {best_match}")


#%%
