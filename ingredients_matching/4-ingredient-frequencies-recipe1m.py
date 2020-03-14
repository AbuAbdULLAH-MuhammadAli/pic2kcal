# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().run_line_magic("cd", "..")


# %%
get_ipython().system("pip install ijson")


# %%
import ijson
from tqdm import tqdm_notebook as tqdm


# %%
import json

# %%
ingredients = []
with open("../data/recipe1m/recipe1m_extracted.jsonl", encoding="utf-8",) as f:
    for line in tqdm(f):
        recipe = json.loads(line)
        ingredients += [
            ingredient
            for ingredient in recipe["ingredients"]
            if ingredient["type"] == "ingredient" and ingredient["matched"]["matched"]
        ]


# %%
from collections import defaultdict, Counter

orig_frequencies = defaultdict(Counter)
frequencies = Counter()
ings = {}
for ingredient in tqdm(ingredients):
    orig_text = ingredient["original"]["ingredient"]
    mid = ingredient["matched"]["id"]
    frequencies.update([mid])
    ings[mid] = ingredient["matched"]
    orig_frequencies[mid].update([orig_text])


# %%
for id, count in frequencies.most_common(100):
    origs = orig_frequencies[id].most_common(5)
    print((str(count) + "x").rjust(10), "", ings[id]["name"], [o for o, count in origs])


# %%
ing_freqs = []
for id, count in frequencies.most_common():
    if count < 5:
        continue
    origs = orig_frequencies[id].most_common(5)
    common_source_names = [o for o, count in origs]
    ing_freqs.append({**ings[id], "most_common_matches": common_source_names})

with open("../data/recipe1m/ingredients_common.json", "w") as f:
    json.dump(ing_freqs, f)


# %%

