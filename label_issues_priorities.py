#!/usr/bin/env python3
# classify_issue_priorities_ranked.py

import os, json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

# --- Setup ---
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

# --- Canonical Priority Keywords ---
priority_keywords = {
    "Blocker": ["urgent", "production down", "showstopper", "cannot proceed", "system outage", "highest priority"],
    "Critical": ["critical", "security hole", "severe", "immediate fix", "data loss", "vulnerability"],
    "Major": ["major bug", "important", "high impact", "core feature broken"],
    "High": ["high priority", "affects many", "serious issue", "needs attention soon"],
    "Medium": ["medium priority", "moderate impact", "normal fix", "not urgent"],
    "Minor": ["minor bug", "low impact", "cosmetic", "typo"],
    "Trivial": ["trivial issue", "nitpick", "very low priority", "small glitch"],
    "Low": ["low priority", "backlog", "non urgent", "future fix"],
    "Lowest": ["lowest priority", "icebox", "almost irrelevant"],
    "None/To be reviewed": ["untriaged", "to review", "awaiting decision", "not set"]
}

label_set = list(priority_keywords.keys())
context_str = json.dumps(priority_keywords, indent=2)

# --- Prompt Builder ---
def build_prompt(summary):
    return f"""
Classify the following GitHub issue summary into exactly ONE of these canonical priorities:
{label_set}

Here are common engineer words/phrases that usually indicate each priority:
{context_str}

Return ONLY JSON in this format:
{{"predicted_priority": "<one_of_the_priorities>"}}

Summary:
\"\"\"{summary}\"\"\"
"""

# --- Load Data ---
df = pd.read_csv("the_stack_github_issues_first100_summaries_gpt4o.csv")

summaries = df["summary_40w"].fillna("").astype(str)
predictions = []

for s in tqdm(summaries, desc="Classifying priorities"):
    prompt = build_prompt(s)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a strict JSON-only classifier."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=50,
            response_format={"type": "json_object"}  # enforce JSON
        )
        out = resp.choices[0].message.content.strip()
        predicted = json.loads(out).get("predicted_priority", "")
    except Exception as e:
        print("Parse error:", e)
        predicted = ""
    predictions.append(predicted)

df["predicted_priority"] = predictions

df.to_csv("classified_issue_priorities.csv", index=False)
print("Wrote classified_issue_priorities.csv with predictions.")
