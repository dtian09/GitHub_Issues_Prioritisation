#!/usr/bin/env python3
# summarize_github_issues_Kw_bart.py

import os, re, time, argparse
from typing import List
import pandas as pd
from tqdm import tqdm

# ---- HF Transformers (BART) ----
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# ----- Args & config -----
def parse_args():
    p = argparse.ArgumentParser(description="Summarize GitHub issues to K words with BART.")
    p.add_argument("--input",  default=os.getenv("INPUT_CSV", "the_stack_github_issues_first100.csv"))
    p.add_argument("--output", default=os.getenv("OUTPUT_CSV", "the_stack_github_issues_first100_summaries.csv"))
    p.add_argument("--model",  default=os.getenv("HF_MODEL", "facebook/bart-large-cnn"),
                   help="Any seq2seq summarization model, e.g. facebook/bart-large-cnn or google/pegasus-cnn_dailymail")
    p.add_argument("--k",      type=int, default=int(os.getenv("WORDS", "40")), help="words per summary")
    p.add_argument("--batch",  type=int, default=int(os.getenv("BATCH_SIZE", "8")), help="batch size for pipeline")
    p.add_argument("--sleep",  type=float, default=float(os.getenv("RATE_LIMIT_SLEEP", "0.0")), help="seconds between batches")
    p.add_argument("--device", type=str, default=os.getenv("DEVICE", "auto"),
                   help="'auto' (GPU if available) or explicit device id, e.g. '0' for CUDA:0 or '-1' for CPU")
    return p.parse_args()

# ----- Helpers -----
TAG_RE  = re.compile(r"</?issue_start>|</?issue_comment>", re.IGNORECASE)
WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)

def clean_issue_text(x: str) -> str:
    s = "" if not isinstance(x, str) else x
    s = TAG_RE.sub("", s)
    return re.sub(r"\s+", " ", s).strip()

def clamp_to_k_words(text: str, k: int) -> str:
    toks = WORD_RE.findall(text or "")
    return " ".join(toks[:k])

def chunk(lst: List[str], size: int):
    for i in range(0, len(lst), size):
        yield lst[i:i+size], i, min(i+size, len(lst))

def resolve_device(arg: str) -> int:
    if arg != "auto":
        return int(arg)
    try:
        import torch
        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1

def main():
    args = parse_args()
    if args.k <= 0:
        raise ValueError("--k / WORDS must be a positive integer")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Missing {args.input}")

    df = pd.read_csv(args.input)
    if "content" not in df.columns:
        raise ValueError("CSV must have a 'content' column.")

    # Prepare texts
    texts = df["content"].fillna("").astype(str).map(clean_issue_text).tolist()
    n = len(texts)
    out_col = f"summary_{args.k}w"
    results = [""] * n

    # Load BART (or compatible) summarizer
    device = resolve_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    summarizer = pipeline(
        "summarization",
        model=mdl,
        tokenizer=tok,
        device=device,
        truncation=True  # truncate long issues to model's max input length (e.g., 1024 tokens for BART)
    )

    # Heuristic lengths (BART uses tokens, not words). Keep it modest; we'll clamp to K words post hoc.
    # You can tweak these if you want longer/shorter drafts before clamping.
    max_new_tokens = max(48, int(args.k * 3))  # ~3 tokens/word
    min_new_tokens = max(12, int(args.k * 1))  # ensure non-empty outputs

    # Batch inference
    for batch, s, e in tqdm(list(chunk(texts, args.batch)), total=(n + args.batch - 1)//args.batch,
                            desc=f"Summarizing to {args.k} words (BART)"):
        try:
            outs = summarizer(
                batch,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=False,
                clean_up_tokenization_spaces=True
            )
            for j, o in enumerate(outs):
                results[s + j] = clamp_to_k_words(o.get("summary_text", "").strip(), args.k)
        except Exception:
            # On hard failure, leave blanks for this batch to preserve alignment
            pass

        if args.sleep > 0:
            time.sleep(args.sleep)

    df[out_col] = results
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(df)} rows and '{out_col}' filled.")

if __name__ == "__main__":
    main()