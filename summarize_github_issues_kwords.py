#!/usr/bin/env python3
# summarize_github_issues_Kw.py

import os, re, time, argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

# ----- Helpers -----
# We still strip issue_start / issue_comment tags, but we DO NOT ignore issue_closed.
START_COMMENT_TAGS = re.compile(r"</?issue_start>|</?issue_comment>", re.IGNORECASE)
CLOSED_TAG         = re.compile(r"</?issue_closed>", re.IGNORECASE)
WORD_RE            = re.compile(r"\b[\w'-]+\b", re.UNICODE)

# ----- Args & config -----
def parse_args():
    p = argparse.ArgumentParser(description="Summarize GitHub issues to K words.")
    p.add_argument("--input", default=os.getenv("INPUT_CSV", "the_stack_github_issues_first100.csv"))
    p.add_argument("--output", default=os.getenv("OUTPUT_CSV", "the_stack_github_issues_first100_summaries.csv"))
    p.add_argument("--model", default=os.getenv("MODEL", "gpt-4o"))
    p.add_argument("--k", type=int, default=int(os.getenv("WORDS", "40")), help="words per summary")
    p.add_argument("--sleep", type=float, default=float(os.getenv("RATE_LIMIT_SLEEP", "0.5")), help="seconds between calls")
    p.add_argument("--temp", type=float, default=float(os.getenv("TEMPERATURE", "0.2")))
    return p.parse_args()

def extract_and_clean_issue(text: str):
    """
    Detect <issue_closed>, then clean tags.
    Returns (cleaned_text, is_closed: bool)
    """
    s = "" if not isinstance(text, str) else text
    is_closed = bool(CLOSED_TAG.search(s))
    # remove all our tags from the text sent to the model
    s = START_COMMENT_TAGS.sub("", s)
    s = CLOSED_TAG.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s, is_closed

def clamp_to_k_words(text: str, k: int) -> str:
    toks = WORD_RE.findall(text or "")
    return " ".join(toks[:k])

def build_messages(issue_text: str, k: int, is_closed: bool) -> list:
    sys = ("You are a concise technical writing assistant. "
           "Summarize GitHub issues crisply for an engineering audience.")
    status_hint = " The issue is CLOSED; explicitly state that it is closed in the summary." if is_closed else ""
    user = (f"Summarize the following GitHub issue in EXACTLY {k} words.{status_hint} "
            "No preamble, numbering, or quotes. Focus on the core problem and requested action.\n\n"
            f"--- ISSUE TEXT START ---\n{issue_text}\n--- ISSUE TEXT END ---")
    return [{"role": "system", "content": sys},
            {"role": "user", "content": user}]

def summarize_issue(text: str, k: int, model: str, temperature: float, max_tokens: int, is_closed: bool) -> str:
    if not text.strip():
        return ""
    resp = client.chat.completions.create(
        model=model,
        messages=build_messages(text, k, is_closed),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    out = (resp.choices[0].message.content or "").strip()
    return clamp_to_k_words(out, k)

# ----- Main -----
def main():
    args = parse_args()
    if args.k <= 0:
        raise ValueError("--k / WORDS must be a positive integer")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Missing {args.input}")

    # Rough token budget: ~3 tokens/word + overhead
    max_tokens = max(60, int(args.k * 3 + 40))

    df = pd.read_csv(args.input)
    if "content" not in df.columns:
        raise ValueError("CSV must have a 'content' column.")

    col_name = f"summary_{args.k}w"
    summaries = []

    for raw in tqdm(df["content"].fillna("").astype(str), desc=f"Summarizing to {args.k} words"):
        try:
            cleaned_text, is_closed = extract_and_clean_issue(raw)
            s = summarize_issue(cleaned_text, args.k, args.model, args.temp, max_tokens, is_closed)
        except Exception:
            s = ""  # keep row alignment on failure
        summaries.append(s)
        if args.sleep > 0:
            time.sleep(args.sleep)

    df[col_name] = summaries
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(df)} rows and '{col_name}' filled.")

if __name__ == "__main__":
    main()