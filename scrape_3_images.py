# %%
from multiprocessing import Pool
from scrape_1_index import db, get_or_retry, data_dir
from bs4 import BeautifulSoup
from tqdm import tqdm
import requests as _requests

img_dir = data_dir / "img"


def get_filename(url):
    return (
        url.replace("https://img.chefkoch-cdn.de/ck.de/rezepte/", "")
        .replace("/", "_")
        .replace("_", "/", 1)
    )


def get_image_urls(soup):
    images = soup.select("#morepictures .gallery-imagewrapper img")
    top_pic = soup.select_one("#top-picture")
    image_urls = [top_pic["src"]] if top_pic else []
    image_urls += [img["data-bigimage"] for img in images]
    return image_urls


def get_images(recipe_id):
    print("get_images", recipe_id)
    html = db.execute(
        "select detail_html from recipes where id=?", (recipe_id,)
    ).fetchone()[0]
    soup = BeautifulSoup(html, "lxml")
    image_urls = get_image_urls(soup)
    for image_url in image_urls:
        filename = img_dir / get_filename(image_url)
        filename.parent.mkdir(parents=True, exist_ok=True)
        if not filename.exists():
            try:
                print("GETTING URL", image_url)
                image = get_or_retry(image_url).content
                with open(filename, "wb") as f:
                    f.write(image)
            except _requests.exceptions.RequestException as e:
                print(f"Error getting image {image_url}: {filename}: {e}")


# %%
if __name__ == "__main__":
    # order by length(detail_html) desc
    missing_recipes = db.execute("select id from recipes").fetchall()

    print(f"getting {len(missing_recipes)} recipes")
    id_list = [recipe["id"] for recipe in missing_recipes]

    with Pool(40) as p:
        p.map(get_images, tqdm(id_list), chunksize=100)
