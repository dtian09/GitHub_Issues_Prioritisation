import os, re, csv, mysql.connector
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from summarize_label_types_priorities import summarize_40w, label_type, label_priority
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

load_dotenv(override=True)

# ---------- Config ----------
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
SPLIT = "train"
STREAMING = True
def parse_args():
    parser = argparse.ArgumentParser(description="Fetch and process GitHub issues from Hugging Face starting at a user-specified index.")
    parser.add_argument('--start', type=int, default=1000, help='Start index (default: 1000)')
    parser.add_argument('--n', type=int, default=1000, help='Number of issues to fetch (default: 1000)')
    return parser.parse_args()

MODEL = os.getenv("MODEL", "gpt-4o")
TEMP = float(os.getenv("TEMPERATURE", "0.2"))

# ---------- MySQL Config ----------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "david")
DB_PASS = os.getenv("DB_PASS", "david")
DB_NAME = os.getenv("DB_NAME", "github_issues_db")

# ---------- Control-char cleaner ----------
_CTRL_EXCEPT_TNL = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")
def clean_text(x):
    if x is None:
        return ""
    if not isinstance(x, str):
        try:
            x = str(x)
        except Exception:
            return ""
    x = x.replace("\r\n", "\n").replace("\r", "\n")
    x = x.replace("\x00", "")
    x = _CTRL_EXCEPT_TNL.sub("", x)
    return x.strip()

# ---------- Load dataset ----------
def fetch_rows(start, n):
    ds = load_dataset(
        "bigcode/the-stack-github-issues",
        split=SPLIT,
        streaming=STREAMING,
        token=HF_TOKEN
    )
    rows = []
    for i, ex in enumerate(tqdm(ds, desc=f"Fetching issues {start} to {start+n-1}")):
        if i < start:
            continue
        if i >= start + n:
            break
        rows.append({
            "issue_id": ex.get("issue_id"),
            "content": clean_text(ex.get("content"))
        })
    return rows

# ---------- Summarize, label, and insert into DB ----------
def insert_issues_batch(cursor, batch):
    sql = """
        INSERT INTO issue (issue_id, content, summary, type, priority)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            content=VALUES(content), summary=VALUES(summary), type=VALUES(type), priority=VALUES(priority)
    """
    cursor.executemany(sql, batch)

def process_row(row):
    issue_id = row["issue_id"]
    content = row["content"]
    try:
        summary = summarize_40w(content, MODEL, TEMP)
    except Exception:
        summary = ""
    try:
        type_ = label_type(summary, MODEL) if summary else ""
    except Exception:
        type_ = ""
    try:
        priority = label_priority(summary, MODEL) if summary else ""
    except Exception:
        priority = ""
    return (issue_id, content, summary, type_, priority)

def main():
    args = parse_args()
    start = args.start
    n = args.n
    rows = fetch_rows(start, n)

    conn = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME
    )
    cursor = conn.cursor()

    batch_size = 20
    batch = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_row = {executor.submit(process_row, row): row for row in rows}
        for future in tqdm(as_completed(future_to_row), total=len(rows), desc="Summarize + label + insert"):
            result = future.result()
            batch.append(result)
            if len(batch) >= batch_size:
                insert_issues_batch(cursor, batch)
                conn.commit()
                batch = []
        # Insert any remaining
        if batch:
            insert_issues_batch(cursor, batch)
            conn.commit()

    cursor.close()
    conn.close()
    print(f"✅ Inserted {len(rows)} issues into DB.")

if __name__ == "__main__":
    main()