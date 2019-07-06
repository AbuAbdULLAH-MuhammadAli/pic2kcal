import json
import re
from collections import OrderedDict

TAKE_N = 10000

ingredient_dict = {}

recipe_num_ingredients = {}

with open('./data/data.json') as json_file:
    data = json.load(json_file)

    for recipe in data:
        recipe_id = recipe['id']
        recipe_num_ingredients[recipe_id] = len(recipe['ingredients'])

        for ingredient in recipe['ingredients']:
            if 'ingredient' in ingredient:
                ingredient = ingredient['ingredient']
                if ingredient not in ingredient_dict:
                    ingredient_dict[ingredient] = []
                ingredient_dict[ingredient].append(recipe_id)

sorted_element = OrderedDict(sorted(ingredient_dict.items(), key=lambda kv: len(kv[1]), reverse=True))

moreThanTwo = [(k, v) for k, v in sorted_element.items() if len(v) > 2]
most_frequent = moreThanTwo[:TAKE_N]

recipe_num_ingredients_most_frequent = {}

for k, v in most_frequent:
    for rec in v:
        if rec not in recipe_num_ingredients_most_frequent:
            recipe_num_ingredients_most_frequent[rec] = 0

        recipe_num_ingredients_most_frequent[rec] += 1

num_lost = 0
for r, n in recipe_num_ingredients.items():
    if r in recipe_num_ingredients_most_frequent:
        if recipe_num_ingredients_most_frequent[r] != n:
            num_lost += 1

print('lost recipes: {}/{}'.format(num_lost, len(recipe_num_ingredients)))
print('recipes: {}'.format(len(recipe_num_ingredients) - num_lost))

ingredients_to_process = []

for x, y in most_frequent:
    name = x
    name = name.lower()
    name = re.sub(r"[^a-z]", "", name)
    ingredients_to_process.append(x)
    print(name)


with open('./data/out.json', 'w') as out_file:
    json.dump(ingredients_to_process, out_file)
