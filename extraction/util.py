import json
import itertools
import collections
from pathlib import Path

def sqlite_db(fname: str):
    import sqlite3

    db = sqlite3.connect(fname, timeout=10)
    db.row_factory = sqlite3.Row

    # unnecessary stuff for 𝐏𝐄𝐀𝐊 𝐏𝐄𝐑𝐅𝐎𝐑𝐌𝐀𝐍𝐂𝐄
    db.execute("pragma page_size = 32768;")
    db.execute("pragma temp_store = memory;")
    db.execute("pragma journal_mode = WAL;")
    db.execute("pragma synchronous = off;")
    db.execute(f"pragma mmap_size={30 * 1000 * 1e6};")
    db.execute("pragma cache_size=-30000")
    db.execute("pragma auto_vacuum = incremental;")
    db.execute("pragma incremental_vacuum;")
    db.execute("pragma optimize;")
    return db


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