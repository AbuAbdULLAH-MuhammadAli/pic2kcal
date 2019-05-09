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
import sqlite3
import math
from util import desktop_agents

# %% setup sqlite db

db = sqlite3.connect("data/output.sqlite3")
db.row_factory = sqlite3.Row

# unnecessary stuff for 𝐏𝐄𝐀𝐊 𝐏𝐄𝐑𝐅𝐎𝐑𝐌𝐀𝐍𝐂𝐄
db.execute("pragma page_size = 32768;")
db.execute("pragma temp_store = memory;")
db.execute("pragma journal_mode = WAL;")
db.execute("pragma synchronous = normal;")
db.execute(f"pragma mmap_size={30 * 1000 * 1e6};")
db.execute("pragma auto_vacuum = incremental;")
db.execute("pragma incremental_vacuum;")
db.execute("pragma optimize;")

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
    """
)

# %% setup requests


def random_headers():
    return {
        "User-Agent": choice(desktop_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    }


reqsession = _requests.Session()
reqsession.proxies = { # todo: don't hardcode
    "http": "socks5://localhost:8080",
    "https": "socks5://localhost:8080",
}
# %%
# Alle 300k Rezepte sortiert nach Datum: http://www.chefkoch.de/rs/s30o3/Rezepte.html


# Chefkoch.de Seite
CHEFKOCH_URL = "http://www.chefkoch.de"
START_URL = "http://www.chefkoch.de/rs/s"
CATEGORY = "/Rezepte.html"

category_url = START_URL + "0o3" + CATEGORY


def get_or_retry(url):
    page = ""
    while page == "":
        try:
            page = reqsession.get(url, headers=random_headers())
            reqsession.cookies.clear()
        except Exception as e:
            print("Connection refused", e)
            sleep(10)
            continue
    return page.text


def compile_index_url_list():
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


if db.execute("select count(*) from index_pages").fetchone()[0] == 0:
    compile_index_url_list()

url_list = db.execute("select url from index_pages where not fetched").fetchall()
url_list = [e["url"] for e in url_list]

from pprint import pprint

# Die ersten 30 Seiten:
pprint(url_list[:30])

# %%


def get_index_page(url):
    print("get_index_page", url)
    html = get_or_retry(url)
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


# url_list = url_list[0:10]
start_time = time()
for url in url_list:
    get_index_page(url)
    # sleep(randint(1, 2))

# with Pool(1) as p:
#    p.map(scrap_main, url_list)
print("--- %s seconds ---" % (time() - start_time))

