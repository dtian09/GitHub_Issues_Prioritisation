#!/usr/bin/env python3
# summarize_github_issues_Kw_groq_llama.py
#
# Summarize each row's `content` into K words using Llama via the Groq API.
#
# Setup:
#   pip install groq pandas tqdm
#   export GROQ_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
#
# Example:
#   python summarize_github_issues_Kw_groq_llama.py --k 30 --model llama3-70b-8192

import os, re, time, argparse
import pandas as pd
from tqdm import tqdm
from groq import Groq

# ----- Args & config -----
def parse_args():
    p = argparse.ArgumentParser(description="Summarize GitHub issues to K words with Llama (Groq API).")
    p.add_argument("--input",  default=os.getenv("INPUT_CSV", "the_stack_github_issues_first100.csv"))
    p.add_argument("--output", default=os.getenv("OUTPUT_CSV", "the_stack_github_issues_first100_summaries.csv"))
    p.add_argument("--model",  default=os.getenv("MODEL", "llama3-70b-8192"),
                   help="Groq model id (e.g., llama3-70b-8192, llama3-8b-8192)")
    p.add_argument("--k",      type=int, default=int(os.getenv("WORDS", "40")), help="words per summary")
    p.add_argument("--sleep",  type=float, default=float(os.getenv("RATE_LIMIT_SLEEP", "0.25")), help="seconds between calls")
    p.add_argument("--temp",   type=float, default=float(os.getenv("TEMPERATURE", "0.2")))
    return p.parse_args()

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GROQ_API_KEY")

client = Groq(api_key=API_KEY)

# ----- Helpers -----
TAG_RE   = re.compile(r"</?issue_start>|</?issue_comment>", re.IGNORECASE)
WORD_RE  = re.compile(r"\b[\w'-]+\b", re.UNICODE)

def clean_issue_text(x: str) -> str:
    s = "" if not isinstance(x, str) else x
    s = TAG_RE.sub("", s)
    return re.sub(r"\s+", " ", s).strip()

def clamp_to_k_words(text: str, k: int) -> str:
    toks = WORD_RE.findall(text or "")
    return " ".join(toks[:k])

def build_messages(issue_text: str, k: int) -> list:
    sys = ("You are a concise technical writing assistant. "
           "Summarize GitHub issues crisply for an engineering audience.")
    user = (f"Summarize the following GitHub issue in EXACTLY {k} words. "
            "No preamble, numbering, or quotes. Focus on the core problem and requested action.\n\n"
            f"--- ISSUE TEXT START ---\n{issue_text}\n--- ISSUE TEXT END ---")
    return [{"role": "system", "content": sys},
            {"role": "user",   "content": user}]

def summarize_issue(text: str, k: int, model: str, temperature: float, max_tokens: int) -> str:
    if not text.strip():
        return ""
    resp = client.chat.completions.create(
        model=model,
        messages=build_messages(text, k),
        temperature=temperature,
        max_tokens=max_tokens,  # Groq supports OpenAI-style param name
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

    # Rough output token budget for Llama: ~3 tokens/word + overhead
    max_tokens = max(60, int(args.k * 3 + 40))

    df = pd.read_csv(args.input)
    if "content" not in df.columns:
        raise ValueError("CSV must have a 'content' column.")

    contents = df["content"].fillna("").astype(str).map(clean_issue_text)
    col_name = f"summary_{args.k}w"

    summaries = []
    for txt in tqdm(contents, desc=f"Summarizing to {args.k} words (Groq Llama)"):
        try:
            s = summarize_issue(txt, args.k, args.model, args.temp, max_tokens)
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