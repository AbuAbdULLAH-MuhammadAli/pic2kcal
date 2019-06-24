import json
from fuzzywuzzy import fuzz, process

fddb_product_list = []
ingredient_list = []

with open('./data/fddb.json') as file:
    for fddb_product in file:
        fddb_product_list.append(fddb_product.replace("\"", "").replace("\n", ""))

with open('./data/out.json') as json_file:
    ingredient_list = json.load(json_file)


for ingredient in ingredient_list:
    score = -1
    matched_ingredient = ''
    for to_match in fddb_product_list:
        new_score = fuzz.token_sort_ratio(ingredient, to_match)
        if new_score > score:
            score = new_score
            matched_ingredient = to_match

    print("{} - {}  score: {}".format(ingredient, matched_ingredient, score))
