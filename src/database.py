import sqlite3
import pandas as pd
from pathlib import Path

DB_FILE = "data/db/reddit_sentiment.sqlite"
TABLE_NAME = "sentiments"
def save_db(df, db_file=DB_FILE, table_name=TABLE_NAME):
    Path(db_file).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_file)
    try:
        df.to_sql(table_name, conn, if_exists="append", index=False)
        print(f"saved {len(df)} rows into {table_name} table in {db_file}")
    finally:
        conn.close()
def load_from_db(db_file=DB_FILE, table_name=TABLE_NAME):
    conn = sqlite3.connect(db_file)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

if __name__ == "__main__":
    input_file = Path("data/sentiment/anime_posts_sentiment.csv")
    if input_file.exists():
        df = pd.read_csv(input_file)
        save_db(df)

        df_from_db = load_from_db()
        print("First 5 rows from Database : ")
        print(df_from_db.head())
    else:
        print(f"{input_file} not found!!")
