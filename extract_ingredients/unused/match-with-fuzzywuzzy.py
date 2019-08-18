import json
from tqdm import tqdm
from fuzzywuzzy import process
from multiprocessing import Pool

fddb_product_dict = {}
ingredient_list = []

# generate file with this command
# jq '.[]' fddb_data.json > fddb_data.jsonl
# jq 'select((.Bewertungen|tonumber) > 0) | {key:.Id, value: .name}' fddb_data.jsonl | jq -s from_entries > fddb_names.json
with open("./data/fddb_names.json") as file:
    fddb_product_dict = json.load(file)

# jq '.ingredients[]|.ingredient' processed_data.jsonl | sort -u | jq -s > ingredients.json
with open("./data/recipes/ingredients.json") as json_file:
    ingredient_list = json.load(json_file)


def extract(ingredient):
    return ingredient, process.extract(ingredient, fddb_product_dict, limit=3)


with open("./data/ingredient-matching/matches.jsonl", "w") as f:
    with Pool(8) as p:
        for extracted in p.imap(extract, tqdm(ingredient_list), chunksize=20):
            json.dump(extracted, f)

