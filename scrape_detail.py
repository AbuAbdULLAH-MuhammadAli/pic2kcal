# %%
from multiprocessing import Pool
from scrape_index import db, get_or_retry
from bs4 import BeautifulSoup

def get_detail_page(recipe_id):
    print("get_detail_page", recipe_id)
    url = f"https://www.chefkoch.de/rezepte/drucken/{recipe_id}"
    html = get_or_retry(url)
    #soup = BeautifulSoup(html, "lxml")
    #lis = soup.select("article.rsel-item")

    with db:
        res = db.execute("update recipes set detail_html = :html where id = :id and detail_html is null", {"html": html, "id": recipe_id})
        if res.rowcount != 1:
            raise Exception(f"error: did not update recipe {recipe_id}")

# %%
if __name__ == "__main__":
    missing_recipes = db.execute("select * from recipes where detail_html is null").fetchall()

    print(f"getting {len(missing_recipes)} detail pages")
    id_list = [recipe["id"] for recipe in missing_recipes]

    with Pool(30) as p:
        p.map(get_detail_page, id_list)