#!/usr/bin/env python3
"""
multi_models_issue_summarizer.py

Summarization + (Type, Priority) classification with cosine similarity,
**standardized through OpenAI-style senders only** for ALL models.

all models called via OpenAI-compatible Chat Completions:

python multi_models_issue_summarizer_single_issue.py --input issue_id_549374190.txt --output predictions.csv --models llama3-70b

temperature parameter:

The temperature parameter controls the randomness or creativity of the model's output:

Low temperature (e.g., 0.0–0.2): Output is more deterministic, focused, and conservative.
High temperature (e.g., 0.7–1.0): Output is more diverse, creative, and varied.

For summarization and classification tasks, a low temperature is usually preferred for consistency and accuracy.

Install:
  pip install openai sentence-transformers pandas tqdm python-dotenv numpy
"""
import os, re, json, argparse
from typing import Dict, Tuple, Callable, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

from openai import OpenAI  # used for ALL models via OpenAI-compatible endpoints

# Import local helper utilities from user's file
from summarize_label_types_priorities_single_issue_input import (
    clamp_words,
    clean_and_detect_closed,
    SUMMARIZE_GUARDRAILS,
    TYPE_LABELS, TYPE_CONTEXT,
    PRIORITY_LABELS, PRIORITY_CONTEXT
)

from sentence_transformers import SentenceTransformer

load_dotenv(override=True)

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

def snap_to_allowed(label: str, allowed: list) -> str:
    if not label:
        return "Unknown"
    low = label.strip().lower()
    by_lower = {a.lower(): a for a in allowed}
    if low in by_lower:
        return by_lower[low]
    for a in allowed:
        if low in a.lower() or a.lower() in low:
            return a
    return "Unknown"

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

# ---------- OpenAI-style sender ----------
SenderFn = Callable[[str, str, str, float, int, bool], str]

def make_openai_style_sender(api_key: str, base_url: Optional[str] = None) -> SenderFn:
    if not api_key:
        raise RuntimeError("Missing API key for OpenAI-style sender.")
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    def _send(model_name: str, system: str, user: str, temperature: float, max_tokens: int, json_mode: bool) -> str:
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return (resp.choices[0].message.content or "").strip()
    return _send

# ---------- generic summarizer/classifiers ----------
def summarize_with(sender: SenderFn, model: str, text: str, temperature: float = 0.2) -> str:
    sys, usr = build_summary_prompt(text)
    out = sender(model, sys, usr, temperature, 160, False)
    return clamp_words(out, 40)

def classify_issue_type(sender: SenderFn, model: str, summary_40w: str):
    sys = "You are a strict JSON-only classifier."
    raw = sender(model, sys, build_type_prompt(summary_40w), 0.0, 200, True)
    data = extract_json(raw) or {}
    label = snap_to_allowed(data.get("predicted_type",""), TYPE_LABELS)
    return label

def classify_issue_priority(sender: SenderFn, model: str, summary_40w: str):
    sys = "You are a strict JSON-only classifier."
    raw = sender(model, sys, build_priority_prompt(summary_40w), 0.0, 200, True)
    data = extract_json(raw) or {}
    label = snap_to_allowed(data.get("predicted_priority",""), PRIORITY_LABELS)
    return label

# ---------- main orchestration ----------
def parse_args():
    p = argparse.ArgumentParser(description="Summarize + classify issues via OpenAI-style API for ALL Models.")
    p.add_argument("--input", required=True, help="CSV with at least a 'content' column.")
    p.add_argument("--output", default="results_openai_only.csv")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--models", nargs="+",
                   default=["gpt-4o","claude3-opus","gemini-2.5-pro","grok-4","llama3-70b", "deepseek-v2"],
                   help="Subset of: gpt-4o claude3-opus gemini-2.5-pro grok-4 llama3-70b deepseek-v2")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Missing input file: {args.input}")

    # Read plain text from file (1 issue only)
    with open(args.input, "r", encoding="utf-8") as f:
        content = f.read().strip()

    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    emb_content = st_model.encode([content], normalize_embeddings=True)[0]

    models_map: Dict[str, Tuple[SenderFn, str]] = {}

    if "gpt-4o" in args.models:
        models_map["gpt-4o"] = (
            make_openai_style_sender(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")),
            "gpt-4o"
        )
    if "claude3-opus" in args.models:
        models_map["claude3-opus"] = (
            make_openai_style_sender(api_key=os.getenv("CLAUDE_API_KEY"), base_url=os.getenv("CLAUDE_BASE_URL")),
            "claude-3-opus"
        )
    if "gemini-2.5-pro" in args.models:
        models_map["gemini-2.5-pro"] = (
            make_openai_style_sender(api_key=os.getenv("GOOGLE_API_KEY"), base_url=os.getenv("GEMINI_BASE_URL")),
            "gemini-2.5-pro"
        )
    if "grok-4" in args.models:
        models_map["grok-4"] = (
            make_openai_style_sender(api_key=os.getenv("XAI_API_KEY"), base_url=os.getenv("GROK_BASE_URL")),
            "grok-4-latest"
        )
    if "llama3-70b" in args.models:
        models_map["llama3-70b"] = (
            make_openai_style_sender(api_key=os.getenv("GROQ_API_KEY"), base_url=os.getenv("GROQ_BASE_URL")),
            "llama3-70b-8192"
        )
    if "deepseek-v2" in args.models:
        models_map["deepseek-v2"] = (
            make_openai_style_sender(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("DEEPSEEK_BASE_URL")),
            "deepseek-v2"
        )

    records = []
    for alias, (sender, model_name) in models_map.items():
        try:
            s40 = summarize_with(sender, model_name, content, args.temperature)
        except Exception:
            s40 = ""

        try:
            t_label, _ = classify_issue_type(sender, model_name, s40) if s40 else ("Unknown", 0.0)
        except Exception:
            t_label = "Unknown"

        try:
            p_label = classify_issue_priority(sender, model_name, s40) if s40 else "Unknown"
        except Exception:
            p_label = "Unknown"

        try:
            emb_summary = st_model.encode([s40], normalize_embeddings=True)[0] if s40 else np.zeros_like(emb_content)
            cos = cosine_similarity(emb_summary, emb_content)
        except Exception:
            cos = 0.0

        records.append({
            "Model": alias,
            "Summary_40w": s40,
            "Predicted Type": t_label,
            "Predicted Priority": p_label,
            "Cosine_Similarity_Score(Summary, Content)": round(cos, 2),
        })

    pd.DataFrame.from_records(records).to_csv(args.output, index=False)
    print(f"Done. Wrote {args.output}")


if __name__ == '__main__': main()
