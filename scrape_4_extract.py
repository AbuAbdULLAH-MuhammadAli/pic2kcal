# %%
from multiprocessing import Pool
from scrape_1_index import db, get_or_retry, data_dir
from bs4 import BeautifulSoup
from scrape_3_images import get_filename as get_image_filename, get_image_urls
from util import sqlite_db
import re
import json

odb = sqlite_db(data_dir / "processed_data.sqlite3")


def write_meta(recipe_id):
    print("get_images", recipe_id)
    index_html, detail_html = db.execute(
        "select index_html,detail_html from recipes where id=?", (recipe_id,)
    ).fetchone()

    soup = BeautifulSoup(detail_html, "lxml")
    image_filenames = [get_image_filename(url) for url in get_image_urls(soup)]
    for basename in image_filenames:
        filename = data_dir / "img" / basename
        if not filename.exists():
            print(f"warning: missing image for recipe {recipe_id}: {filename}")

    subtitle = soup.select_one("#content > p > strong")
    ing_tbl = soup.select_one("#content table.incredients > tbody > tr")
    ingredients = [[col.get_text() for col in row.select("> td")] for row in ing_tbl]
    processed_data = {
        "title": soup.select_one("#content > h1").get_text(),
        "subtitle": subtitle.get_text() if subtitle else None,
        "tags": [],  # annoying to get these from the print page
        "rating": 1,
        "rating_count": 23,
        "portions": float(
            re.match(
                r"Zutaten für (\d+) Portion",
                soup.select_one("#content > .content-right > h3"),
            ).group(1)
        ),
        "ingredients": ingredients,
        "recipe_text": "",
        "author": 1,
        "worktime_min": 1,
        "cookingtime_min": 1,
        "difficulty": "normal",
        "kcal_per_portion": 123,
    }
    print(json.dumps(processed_data))


# %%
if __name__ == "__main__":
    # order by length(detail_html) desc
    missing_recipes = db.execute(
        "select id from recipes where detail_html is not null limit 20"
    ).fetchall()

    print(f"getting {len(missing_recipes)} recipes")
    id_list = [recipe["id"] for recipe in missing_recipes]

    with Pool(20) as p:
        p.map(write_meta, id_list)
