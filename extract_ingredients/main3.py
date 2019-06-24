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
    print("{} : {}".format(ingredient, process.extract(ingredient, fddb_product_list, limit=3)))
