
#!/usr/bin/env python3
import os, re, json, argparse, logging
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from summarize_label_types_priorities_single_issue_input import (
    summarize_40w,
    label_type,
    label_priority
)
'''
python multi_models_issue_summarizer_single_issue_with_logging.py --input issue_id_549374190.txt --output predictions.csv --models gpt-4o claude3-opus llama3-70b --log-errors
''' 

load_dotenv(override=True)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def setup_error_logging(enabled: bool):
    if enabled:
        logging.basicConfig(
            filename="multi_models_issue_errors.log",
            level=logging.WARNING,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )

def parse_args():
    p = argparse.ArgumentParser(description="Summarize + classify single issue using multiple LLMs via OpenAI-compatible API.")
    p.add_argument("--input", required=True, help="Plain text file containing a single GitHub issue.")
    p.add_argument("--output", default="results_openai_only.csv")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--log-errors", action="store_true", help="Enable logging of model errors.")
    p.add_argument("--models", nargs="+", default=["gpt-4o", "claude3", "gemini-2.5-pro", "grok-4", "llama3-70b", "deepseek-v2"])
    return p.parse_args()


try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

def make_claude_sender(api_key: str):
    if not Anthropic:
        raise ImportError("Anthropic SDK not installed")
    client = Anthropic(api_key=api_key)
    def _send(model_name: str, system: str, user: str, temperature: float, max_tokens: int, json_mode: bool) -> str:
        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}]
        )
        return response.content[0].text.strip()
    return _send


def main():
    args = parse_args()
    setup_error_logging(args.log_errors)

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Missing input file: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        content = f.read().strip()

    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    emb_content = st_model.encode([content], normalize_embeddings=True)[0]

    from openai import OpenAI
    def make_openai_style_sender(api_key: str, base_url: str):
        client = OpenAI(api_key=api_key, base_url=base_url)
        def _send(model_name: str, system: str, user: str, temperature: float, max_tokens: int, json_mode: bool) -> str:
            kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return (resp.choices[0].message.content or "").strip()
        return _send

    models_map = {        "gpt-4o":        (make_openai_style_sender(os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_BASE_URL")), "gpt-4o"),
        "claude3":  (make_claude_sender(os.getenv("CLAUDE_API_KEY")), "claude-3-5-sonnet-latest"),
        "gemini-2.5-pro":(make_openai_style_sender(os.getenv("GOOGLE_API_KEY"), os.getenv("GEMINI_BASE_URL")), "gemini-2.5-pro"),
        "grok-4":        (make_openai_style_sender(os.getenv("XAI_API_KEY"), os.getenv("GROK_BASE_URL")), "grok-4-latest"),
        "llama3-70b":    (make_openai_style_sender(os.getenv("GROQ_API_KEY"), os.getenv("GROQ_BASE_URL")), "llama3-70b"),
        "deepseek-v2":   (make_openai_style_sender(os.getenv("DEEPSEEK_API_KEY"), os.getenv("DEEPSEEK_BASE_URL")), "deepseek-v2")
    }

    records = []
    for alias in args.models:
        sender, model_name = models_map.get(alias, (None, None))
        if sender is None:
            continue

        try:
            summary = summarize_40w(content, model_name, args.temperature)
        except Exception as e:
            if args.log_errors:
                logging.warning(f"[{alias}] summarize_40w failed: {e}")
            summary = ""

        try:
            t_label = label_type(summary, model_name) if summary else "Unknown"
        except Exception as e:
            if args.log_errors:
                logging.warning(f"[{alias}] label_type failed: {e}")
            t_label = "Unknown"

        try:
            p_label = label_priority(summary, model_name) if summary else "Unknown"
        except Exception as e:
            if args.log_errors:
                logging.warning(f"[{alias}] label_priority failed: {e}")
            p_label = "Unknown"

        try:
            emb_summary = st_model.encode([summary], normalize_embeddings=True)[0] if summary else np.zeros_like(emb_content)
            cos = cosine_similarity(emb_summary, emb_content)
        except Exception as e:
            if args.log_errors:
                logging.warning(f"[{alias}] cosine similarity failed: {e}")
            cos = 0.0

        records.append({
            "Model": alias,
            "Summary_40w": summary,
            "Predicted Type": t_label,
            "Predicted Priority": p_label,
            "Cosine_Similarity_Score(Summary, Content)": round(cos, 6),
        })

    pd.DataFrame.from_records(records).to_csv(args.output, index=False)
    print(f"Done. Wrote {args.output}")

if __name__ == "__main__":
    main()
