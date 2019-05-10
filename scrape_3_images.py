# %%
from multiprocessing import Pool
from scrape_1_index import db, get_or_retry, data_dir
from bs4 import BeautifulSoup

img_dir = data_dir / "img"
print(img_dir)


def get_filename(url):
    return (
        url.replace("https://img.chefkoch-cdn.de/ck.de/rezepte/", "")
        .replace("/", "_")
        .replace("_", "/", 1)
    )


def get_images(recipe_id):
    print("get_images", recipe_id)
    html = db.execute(
        "select detail_html from recipes where id=?", (recipe_id,)
    ).fetchone()[0]
    soup = BeautifulSoup(html, "lxml")
    images = soup.select("#morepictures .gallery-imagewrapper img")
    top_pic = soup.select_one("#top-picture")
    image_urls = [top_pic["src"]] if top_pic else []
    image_urls += [img["data-bigimage"] for img in images]
    # lis = soup.select("article.rsel-item")
    for image_url in image_urls:
        filename = img_dir / get_filename(image_url)
        filename.parent.mkdir(parents=True, exist_ok=True)
        if not filename.exists():
            image = get_or_retry(image_url).content
            with open(filename, "wb") as f:
                f.write(image)


# %%
if __name__ == "__main__":
    # order by length(detail_html) desc
    missing_recipes = db.execute(
        "select id from recipes where detail_html is not null"
    ).fetchall()

    print(f"getting {len(missing_recipes)} recipes")
    id_list = [recipe["id"] for recipe in missing_recipes]

    with Pool(20) as p:
        p.map(get_images, id_list)
