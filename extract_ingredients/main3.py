import json

from fuzzywuzzy import process
from joblib import Parallel, delayed

fddb_product_dict = {}
ingredient_list = []

# generate file with this command
# jq -c   '{ key:(.name? // "asdf"), value: .name }  '  fddb.jsonl  | jq -s from_entries > fddb.json
with open('./data/fddb.json') as file:
    fddb_product_dict = json.load(file)

with open('./data/out.json') as json_file:
    ingredient_list = json.load(json_file)


def extract(ingredient):
    print("{} : {}".format(ingredient, process.extract(ingredient, fddb_product_dict, limit=3)))


Parallel(n_jobs=4)(delayed(extract)(ingredient) for ingredient in ingredient_list)
