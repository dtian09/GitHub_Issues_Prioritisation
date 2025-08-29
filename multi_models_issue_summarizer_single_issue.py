#!/usr/bin/env python3
"""
multi_models_issue_summarizer.py

Summarization + (Type, Priority) classification with cosine similarity,
**standardized through OpenAI-style senders only** for ALL models.

all models called via OpenAI-compatible Chat Completions:

python multi_models_issue_summarizer_single_issue.py --input "C:/Users/dtian/GitHub_Issues_Prioritisation/shortest_issues/issue_id_705332485.txt" --output C:/Users/dtian/GitHub_Issues_Prioritisation/shortest_issues/predictions_705332485.csv --models gpt-4o grok-4 claude-3-5-sonnet-latest gemini-2.0-flash deepseek-chat llama-3.3-70b-versatile

temperature parameter:

The temperature parameter controls the randomness or creativity of the model's output:

Low temperature (e.g., 0.0–0.2): Output is more deterministic, focused, and conservative.
High temperature (e.g., 0.7–1.0): Output is more diverse, creative, and varied.

For summarization and classification tasks, a low temperature is usually preferred for consistency and accuracy.

Install:
  pip install openai sentence-transformers pandas tqdm python-dotenv numpy
"""
import os, re, json, argparse
from typing import Optional

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from openai import OpenAI  # used for ALL models via OpenAI-compatible endpoints

# Import local helper utilities from user's file
from summarize_label_types_priorities_single_issue_input import (
    clean_and_detect_closed,
    count_tokens,
    SUMMARIZE_GUARDRAILS,
    TYPE_LABELS, TYPE_CONTEXT,
    PRIORITY_LABELS, PRIORITY_CONTEXT
)

from sentence_transformers import SentenceTransformer

load_dotenv(override=True)

# Disable OpenTelemetry to avoid connection errors
os.environ["OTEL_SDK_DISABLED"] = "true"

# ---------- helpers ----------
def extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{(?:[^{}]|(?R))*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

# ---------- prompts ----------
def build_summary_prompt(text: str):
    cleaned, is_closed = clean_and_detect_closed(text)
    sys = ("You are a concise technical writing assistant for software issues. "
           "Produce grounded, non-speculative summaries that strictly follow the rules below.\n\n"
           + SUMMARIZE_GUARDRAILS)
    hint = " The issue is CLOSED; explicitly say it is closed." if is_closed else ""
    user = (f"Summarize the following GitHub issue in EXACTLY 40 words.{hint} "
            "No preamble, numbering, or quotes. Focus on the core problem and requested action.\n\n"
            f"--- ISSUE TEXT START ---\n{cleaned}\n--- ISSUE TEXT END ---")
    return sys, user

def build_type_prompt(summary: str) -> str:
    return f"""
Classify the following issue summary into exactly ONE of these 24 types:
{TYPE_LABELS}

Here are common engineer words/phrases that usually indicate each type:
{TYPE_CONTEXT}

Return ONLY JSON exactly like:
{{"predicted_type":"<one_of_the_types>"}}

Summary:
\"\"\"{summary}\"\"\"""".strip()

def build_priority_prompt(summary: str) -> str:
    return f"""
Classify the following issue summary into exactly ONE of these canonical priorities:
{PRIORITY_LABELS}

Here are common engineer words/phrases that usually indicate each priority:
{PRIORITY_CONTEXT}

Return ONLY JSON exactly like:
{{"predicted_priority":"<one_of_the_priorities>"}}

Summary:
\"\"\"{summary}\"\"\"""".strip()

# ---------- main orchestration ----------
def parse_args():
    p = argparse.ArgumentParser(description="Summarize + classify issues via OpenAI-style API for ALL Models.")
    p.add_argument("--input", required=True, help="CSV with at least a 'content' column.")
    p.add_argument("--output", default="results_openai_only.csv")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--log-cleaned", action="store_true",
                   help="If set, save cleaned issue texts into cleaned_issues.log")
    p.add_argument("--models", nargs="+",
                   default=["gpt-4o","claude-3-5-sonnet-latest","gemini-2.0-flash","grok-4","llama3-70b-8192", "deepseek-chat"],
                   help="Subset of: gpt-4o claude-3-5-sonnet-latest gemini-2.0-flash grok-4 llama3-70b-8192 deepseek-chat")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Missing input file: {args.input}")

    # Read plain text from file (1 issue only)
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except UnicodeDecodeError:
        # Try different encodings if UTF-8 fails
        for encoding in ["utf-16", "utf-16-le", "utf-16-be", "latin-1", "cp1252"]:
            try:
                with open(args.input, "r", encoding=encoding) as f:
                    content = f.read().strip()
                print(f"[INFO] Successfully read file using {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise UnicodeDecodeError("Unable to decode file with any common encoding")

    # Use summarize_label_types_priorities_single_issue_input.py functions
    from summarize_label_types_priorities_single_issue_input import summarize_40w, label_type, label_priority
    import anthropic
    from groq import Groq

    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    emb_content = st_model.encode([content], normalize_embeddings=True)[0]
    
    # Get cleaned content word count for tracking
    cleaned_content, _ = clean_and_detect_closed(content, log=args.log_cleaned)
    cleaned_content_word_count = count_tokens(cleaned_content)

    records = []
    for model in args.models:
        # Initialize appropriate client for each model
        if model == "gpt-4o" or model == "gpt-5":
            client = OpenAI()
        elif model == "claude3-opus" or model == "claude-3-5-sonnet-latest":
            client = anthropic.Anthropic()
        elif model == "gemini-2.5-pro" or model == "gemini-2.0-flash":
            client = OpenAI(api_key=os.environ["GOOGLE_API_KEY"], base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        elif model == "grok-4":
            from xai_sdk import Client
            client = Client(api_key=os.environ["XAI_API_KEY"])
        elif model == "llama-3.3-70b-versatile":
            client = Groq()
        elif model == "deepseek-v2" or model == "deepseek-chat":
            client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com/v1")
        else:
            import sys
            sys.exit(f"Unknown model: {model}")

        summary = summarize_40w(content, model, args.temperature, client, log=args.log_cleaned, count_tokens_flag=False)      

        try:
            # Type
            predicted_type = label_type(summary, model, client) if summary else ""
        except Exception as e:
            print(f"Error labeling type with {model}: {e}")
            predicted_type = ""

        try:
            # Priority
            predicted_priority = label_priority(summary, model, client) if summary else ""
        except Exception as e:
            print(f"Error labeling priority with {model}: {e}")
            predicted_priority = ""

        # Cosine similarity
        try:
            emb_summary = st_model.encode([summary], normalize_embeddings=True)[0] if summary else np.zeros_like(emb_content)
            cos = cosine_similarity(emb_summary, emb_content)
        except Exception:
            cos = 0.0

        records.append({
            "Model": model,
            "Summary_40w": summary,
            "Predicted Type": predicted_type,
            "Predicted Priority": predicted_priority,
            "Cosine_Similarity_Score(Summary, Cleaned Content)": round(cos, 2),
            "Cleaned_Content_Word_Count": cleaned_content_word_count,
        })

    # Save output
    pd.DataFrame.from_records(records).to_csv(args.output, index=False)
    print(f"Done. Wrote {args.output}")


if __name__ == '__main__': main()
