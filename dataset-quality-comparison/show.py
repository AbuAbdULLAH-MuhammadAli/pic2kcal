from recipe_scrapers import scrape_me
import json
import random
import pandas

with open("../data/recipe1m/dl/recipes_with_nutritional_info.json") as f:
    recipes = json.load(f)

for i in range(0, 200):
    print()
    print()
    recipe = random.choice(recipes)
    print(i, recipe["url"])

    ings_parsed = [
        " ".join(i["text"] for i in e)
        for e in zip(recipe["quantity"], recipe["unit"], recipe["ingredients"])
    ]
    try:
        scraped = scrape_me(recipe["url"])
    except Exception as e:
        print("recipe ", i, "skipped")
        continue
    try:
        df = pandas.DataFrame(dict(orig=scraped.ingredients(), r1m_parsed=ings_parsed))
        print(df)
    except ValueError as e:
        print("differing ings lengths")
        print("orig", scraped.ingredients())
        print("r1m_parsed", ings_parsed)
