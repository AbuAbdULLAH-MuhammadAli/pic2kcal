#
# fetches a list of all recipe urls / basic metadata into a sqlite db
# download [sqlitebrowser](https://sqlitebrowser.org/)
#
# this file is in Jupyter Text Notebook format,
# use [VSCode Python plugin](https://code.visualstudio.com/docs/python/jupyter-support)
# for the best experience :)
#
#
# somewhat adapted from https://github.com/Murgio/Food-Recipe-CNN/blob/master/python-files/01_rezepte_download.py

# %% imports

from time import sleep
from time import time
from random import choice
import requests as _requests
from bs4 import BeautifulSoup
import math
from pathlib import Path
from util import desktop_agents, sqlite_db

# %% setup sqlite db

data_dir = Path("data")

db = sqlite_db(data_dir / "raw_data.sqlite3")

db.executescript(
    """
    create table if not exists index_pages (
        url text primary key not null,
        fetched boolean not null
    );
    create index if not exists idx_pages_fetched on index_pages (fetched);

    create table if not exists recipes (
        id text primary key not null,
        canonical_url text not null,
        index_html text not null,
        detail_html text,
        data json
    );
    create index if not exists idx_rd on recipes(detail_html) where detail_html is null;
    """
)

# %% setup requests


def random_headers():
    return {
        "User-Agent": choice(desktop_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    }


proxies = [f"socks5://localhost:173{i:02}" for i in range(0, 5)]  # todo: don't hardcode
reqsessions = []
for proxy in proxies:
    reqsession = _requests.Session()
    reqsession.proxies = {"http": proxy, "https": proxy}
    reqsessions.append(reqsession)
# %%


def get_or_retry(url):
    reqsession = choice(reqsessions)
    i = 5
    while i > 0:
        try:
            reqsession.cookies.clear()
            page = reqsession.get(url, headers=random_headers())
            page.raise_for_status()
            return page
        except _requests.exceptions.RequestException as e:
            print("Could not fetch", url, e, "retrying")
            sleep(10)
            i -= 1

    raise Exception("Could not fetch 5 times", url)


def compile_index_url_list():
    START_URL = "http://www.chefkoch.de/rs/s"
    CATEGORY = "/Rezepte.html"

    # get index url list
    def _get_total_pages():
        return math.ceil(334360 / 30)  # todo: dont hardcode

    total_pages = _get_total_pages()
    print("Total pages: ", total_pages)

    # Liste von allen einzelnen Rezepteurls bei Chefkoch im folgenden Format:
    # 1. Seite: http://www.chefkoch.de/rs/s**0**o3/Rezepte.html
    # 2. Seite: http://www.chefkoch.de/rs/s**30**o3/Rezepte.html
    # 3. Seite: http://www.chefkoch.de/rs/s**60**o3/Rezepte.html
    # 4. Seite: ...
    #
    # Auf einer Seite erhält man 30 Rezepte. Um jede Seite aufrufen zu können, muss man nur die Zahl zwischen **s** und **o3** um 30 erhöhen.

    url_list = [
        START_URL + str(i * 30) + "o3" + CATEGORY for i in range(0, total_pages + 1)
    ]
    with db:
        db.executemany(
            "insert into index_pages (url, fetched) values (?, ?)",
            [(url, False) for url in url_list],
        )


# %%


def get_index_page(url):
    print("get_index_page", url)
    html = get_or_retry(url).text
    soup = BeautifulSoup(html, "lxml")
    lis = soup.select("article.rsel-item")

    with db:
        for index, ele in enumerate(lis):
            a = ele.select_one("> a")
            id = a["data-vars-recipe-id"]
            title = a["data-vars-recipe-title"]
            info = {"id": id, "canonical_url": a["href"], "index_html": str(ele)}
            res = db.execute(
                """insert into recipes (id, canonical_url, index_html)
                    values (:id, :canonical_url, :index_html)
                    on conflict(id) do nothing""",
                info,
            )
            if res.rowcount == 0:
                # index pages are ordered by date, but in descending order
                # → it should be impossible to miss recipes, but we can see duplicates as new recipes are added
                print(
                    f"warning: skipping duplicate recipe {id} (title={title}, url={a['href']})"
                )
        db.execute("update index_pages set fetched=true where url=?", (url,))


# %% scrape all index pages that have not yet been scraped

if __name__ == "__main__":
    if db.execute("select count(*) from index_pages").fetchone()[0] == 0:
        compile_index_url_list()

    index_url_list = db.execute(
        "select url from index_pages where not fetched"
    ).fetchall()
    index_url_list = [e["url"] for e in index_url_list]

    start_time = time()
    print(f"getting {len(index_url_list)} index pages")
    for url in index_url_list:
        get_index_page(url)
        # sleep(randint(1, 2))

    # with Pool(1) as p:
    #    p.map(scrap_main, url_list)
    print("--- %s seconds ---" % (time() - start_time))

