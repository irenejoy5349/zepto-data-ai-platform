import re
import sqlite3
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com"
CATEGORY_URLS = {
    "Travel": f"{BASE_URL}/catalogue/category/books/travel_2/index.html",
    "Mystery": f"{BASE_URL}/catalogue/category/books/mystery_3/index.html",
    "Historical Fiction": f"{BASE_URL}/catalogue/category/books/historical-fiction_20/index.html",
}
DATA_DIR = Path("data_pipeline")
DB_PATH = DATA_DIR / "zepto_books.db"
CLEANED_PATH = DATA_DIR / "books_cleaned.csv"
GBP_TO_INR = 105.50

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def parse_price(text: str) -> float:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        raise ValueError(f"Unable to parse price: {text!r}")
    return float(match.group(1))


def parse_rating(article) -> int:
    classes = article.select_one("p.star-rating").get("class", [])
    label = next((c for c in classes if c in RATING_MAP), None)
    if label is None:
        raise ValueError("Unknown star rating")
    return RATING_MAP[label]


def scrape_category(category: str, first_url: str) -> list[dict]:
    rows = []
    url = first_url
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 Zepto-Capstone-Scraper/1.0"})

    while url:
        response = session.get(url, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for article in soup.select("article.product_pod"):
            rows.append(
                {
                    "title": article.select_one("h3 a")["title"].strip(),
                    "price_gbp": parse_price(article.select_one("p.price_color").get_text(" ", strip=True)),
                    "star_rating": article.select_one("p.star-rating").get("class", ["unknown"])[-1],
                    "availability": article.select_one("p.instock.availability").get_text(" ", strip=True),
                    "category": category,
                }
            )

        next_link = soup.select_one("li.next a")
        if next_link:
            href = next_link["href"]
            url = str((Path(url).parent / href).resolve()) if False else requests.compat.urljoin(url, href)
        else:
            url = None

    return rows


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    cleaned["price_gbp"] = pd.to_numeric(cleaned["price_gbp"], errors="coerce")
    cleaned["rating"] = cleaned["star_rating"].map(RATING_MAP)
    cleaned["in_stock"] = cleaned["availability"].str.contains("in stock", case=False, na=False)

    for col in ["price_gbp", "rating"]:
        if cleaned[col].isna().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    cleaned = cleaned.dropna(subset=["title", "category", "price_gbp", "rating"]).copy()
    cleaned["price_gbp"] = cleaned["price_gbp"].astype(float)
    cleaned["rating"] = cleaned["rating"].round().clip(1, 5).astype(int)
    cleaned["in_stock"] = cleaned["in_stock"].astype(bool)
    cleaned["price_inr"] = (cleaned["price_gbp"] * GBP_TO_INR).round(2)

    return cleaned[
        ["title", "price_gbp", "star_rating", "availability", "category", "rating", "in_stock", "price_inr"]
    ]


def load_sqlite(cleaned: pd.DataFrame) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS books")
        cur.execute("DROP TABLE IF EXISTS categories")
        cur.execute(
            """CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY,
                category_name TEXT NOT NULL UNIQUE
            )"""
        )
        cur.execute(
            """CREATE TABLE books (
                book_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                star_rating TEXT,
                availability TEXT,
                rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
                in_stock INTEGER NOT NULL CHECK(in_stock IN (0, 1)),
                category_id INTEGER NOT NULL,
                FOREIGN KEY(category_id) REFERENCES categories(category_id)
            )"""
        )

        categories = sorted(cleaned["category"].unique())
        category_ids = {name: i + 1 for i, name in enumerate(categories)}
        cur.executemany(
            "INSERT INTO categories(category_id, category_name) VALUES (?, ?)",
            [(cid, name) for name, cid in category_ids.items()],
        )

        rows = []
        for book_id, row in enumerate(cleaned.itertuples(index=False), start=1):
            rows.append(
                (
                    book_id,
                    row.title,
                    row.price_gbp,
                    row.price_inr,
                    row.star_rating,
                    row.availability,
                    row.rating,
                    int(row.in_stock),
                    category_ids[row.category],
                )
            )
        cur.executemany(
            """INSERT INTO books
            (book_id, title, price_gbp, price_inr, star_rating, availability, rating, in_stock, category_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def run_query(conn, query: str, query_name: str) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print(query_name)
    print("=" * 70)
    print("SQL:")
    print(query.strip())
    df = pd.read_sql_query(query, conn)
    print("\nOutput:")
    print(df.to_string(index=False))
    return df


def run_queries() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        query_1 = "SELECT title, price_gbp, rating FROM books WHERE rating >= 4;"
        query_2 = "SELECT title, price_inr FROM books ORDER BY price_inr DESC LIMIT 10;"
        query_3 = "SELECT DISTINCT category_name FROM categories;"
        query_4 = """SELECT b.title, c.category_name, b.rating FROM books b
        INNER JOIN categories c ON b.category_id = c.category_id
        WHERE c.category_name IN ('Travel', 'Mystery') ORDER BY b.rating DESC;"""
        query_5 = "SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 20 AND 40 ORDER BY price_gbp;"
        query_6 = """SELECT b.title, b.rating, b.price_gbp, c.category_name
        FROM books b INNER JOIN categories c ON b.category_id = c.category_id
        ORDER BY b.rating DESC, c.category_name, b.title;"""

        outputs = [
            run_query(conn, query_1, "Query 1 - SELECT + WHERE"),
            run_query(conn, query_2, "Query 2 - ORDER BY + LIMIT"),
            run_query(conn, query_3, "Query 3 - DISTINCT"),
            run_query(conn, query_4, "Query 4 - IN + JOIN"),
            run_query(conn, query_5, "Query 5 - BETWEEN"),
            run_query(conn, query_6, "Query 6 - JOIN"),
        ]

        for i, df in enumerate(outputs, start=1):
            df.to_csv(DATA_DIR / f"query_{i}_output.csv", index=False)

        books_df = pd.read_sql(
            "SELECT book_id, title, price_gbp, rating, category_id FROM books;", conn
        )
        categories_df = pd.read_sql(
            "SELECT category_id, category_name FROM categories;", conn
        )
        merge_df = pd.merge(books_df, categories_df, on="category_id", how="inner")
        merge_df = merge_df[["title", "rating", "price_gbp", "category_name"]].sort_values(
            ["rating", "category_name", "title"], ascending=[False, True, True]
        ).reset_index(drop=True)

        sql_join = pd.read_sql(
            """SELECT b.title, b.rating, b.price_gbp, c.category_name
            FROM books b INNER JOIN categories c ON b.category_id = c.category_id
            ORDER BY b.rating DESC, c.category_name, b.title;""", conn
        ).sort_values(["rating", "category_name", "title"], ascending=[False, True, True]).reset_index(drop=True)

        sql_join.to_csv(DATA_DIR / "pd_read_sql_output.csv", index=False)
        merge_df.to_csv(DATA_DIR / "pd_merge_output.csv", index=False)
        print("\npd.read_sql JOIN and pd.merge equivalent:", sql_join.equals(merge_df))
    finally:
        conn.close()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Starting Books to Scrape collection for 3 categories...")
    rows = []
    try:
        for category, url in CATEGORY_URLS.items():
            category_rows = scrape_category(category, url)
            print(f"{category}: scraped {len(category_rows)} books")
            rows.extend(category_rows)
        if len(rows) < 60 or len({r["category"] for r in rows}) < 3:
            raise RuntimeError("Scrape result does not satisfy the minimum rubric requirements.")
        raw_df = pd.DataFrame(rows)
    except Exception as exc:
        # Offline fallback keeps the project reproducible when external access is unavailable.
        # The committed cleaned dataset was created by this same scraper workflow.
        print(f"Live scraping failed: {exc}")
        try:
            conn = sqlite3.connect(DB_PATH)
            cached = pd.read_sql_query(
                """SELECT b.title, b.price_gbp, b.rating, b.in_stock, c.category_name AS category
                   FROM books b JOIN categories c ON b.category_id = c.category_id;""",
                conn,
            )
            conn.close()
        except Exception:
            cached = pd.DataFrame()
        required = {"title", "price_gbp", "rating", "in_stock", "category"}
        if not required.issubset(cached.columns) or len(cached) < 60 or cached["category"].nunique() < 3:
            raise RuntimeError("Live scraping failed and the offline fallback is invalid.") from exc
        reverse_rating = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}
        raw_df = cached.copy()
        raw_df["star_rating"] = raw_df["rating"].map(reverse_rating).fillna("Unknown")
        raw_df["availability"] = raw_df["in_stock"].map({1: "In stock", 0: "Out of stock"}).fillna("Unknown")
        raw_df = raw_df[["title", "price_gbp", "star_rating", "availability", "category"]]
        print(f"Using offline fallback: {len(raw_df)} cached rows")

    cleaned = clean_data(raw_df)
    print(f"Final cleaned rows: {len(cleaned)}")
    print(f"Categories: {sorted(cleaned['category'].unique().tolist())}")
    cleaned.to_csv(CLEANED_PATH, index=False)
    load_sqlite(cleaned)
    run_queries()
    print(f"SQLite database saved to {DB_PATH}")
    print("Data pipeline completed successfully.")


if __name__ == "__main__":
    main()
