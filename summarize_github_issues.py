#!/usr/bin/env python3
# summarize_github_issues_20w.py

import os
import time
import re
import pandas as pd
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
#from dotenv import load_dotenv
from openai import OpenAI

#load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==== Config ====
INPUT_CSV = os.getenv("INPUT_CSV", "the_stack_github_issues_first100.csv")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "the_stack_github_issues_first100_summaries.csv")
# Use a GPT‑4 class model; override via env MODEL if you wish (e.g., gpt-4o, gpt-4.1, gpt-4o-mini)
MODEL = os.getenv("MODEL", "gpt-4o")

TEMPERATURE = 0.2
MAX_TOKENS = 120  # plenty for 20 words + safety
RATE_LIMIT_SLEEP = float(os.getenv("RATE_LIMIT_SLEEP", "0.5"))  # seconds between calls

# ==== Helpers ====
_word_re = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)

def coerce_to_20_words(text: str) -> str:
    """
    Force the output to exactly 20 space-separated words:
    - Tokenize by word boundaries.
    - If >20, truncate to first 20.
    - If <20, keep as-is (model is instructed to output exactly 20).
    """
    tokens = _word_re.findall(text)
    if not tokens:
        return ""
    if len(tokens) >= 20:
        tokens = tokens[:20]
    else:
        # If model under-shoots, don't fabricate words; return what we have.
        # (Typically the prompt enforces 20 exactly.)
        pass
    return " ".join(tokens)

def build_messages(issue_text: str):
    sys = (
        "You are a concise technical writing assistant. "
        "Summarize GitHub issues crisply for an engineering audience."
    )
    user = (
        "Summarize the following GitHub issue in EXACTLY 20 words. "
        "No preamble, no numbering, no quotes. Focus on the core problem and requested action.\n\n"
        f"--- ISSUE TEXT START ---\n{issue_text}\n--- ISSUE TEXT END ---"
    )
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]

class TransientOpenAIError(Exception):
    pass

@retry(
    retry=retry_if_exception_type(TransientOpenAIError),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    stop=stop_after_attempt(5),
    reraise=True,
)
def summarize_issue(text: str) -> str:
    if not text or str(text).strip() == "" or str(text).lower() == "nan":
        return ""

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=build_messages(str(text)),
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    except Exception as e:
        # Treat many API failures as transient (rate limits, timeouts, 5xx)
        msg = str(e).lower()
        if any(s in msg for s in ["rate limit", "timeout", "server error", "overloaded", "502", "503", "504"]):
            raise TransientOpenAIError(e)
        raise

    out = resp.choices[0].message.content.strip()
    return coerce_to_20_words(out)

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Input file '{INPUT_CSV}' not found. "
            "Set INPUT_CSV env var or place the file in the working directory."
        )

    df = pd.read_csv(INPUT_CSV)
    # Pick the content column your CSV has; your earlier script used 'content'
    text_col_candidates = ["content", "contents", "body", "text", "issue_text"]
    text_col = next((c for c in text_col_candidates if c in df.columns), None)
    if text_col is None:
        raise ValueError(f"Could not find a text column among {text_col_candidates}. Found columns: {list(df.columns)}")

    summaries = []
    for txt in tqdm(df[text_col].fillna(""), desc="Summarizing issues"):
        try:
            s = summarize_issue(txt)
        except Exception as e:
            s = f""  # leave blank on hard failure
        summaries.append(s)
        time.sleep(RATE_LIMIT_SLEEP)  # gentle pacing

    df["summary_20w"] = summaries
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV} with {len(df)} rows.")

if __name__ == "__main__":
    main()
