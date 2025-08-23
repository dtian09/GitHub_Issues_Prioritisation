#!/usr/bin/env python3
# classify_issues_with_keywords_fixed.py

import os, re, time, argparse, json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

# ==== Setup ====
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

# ==== Issue type vocabulary ====
type_keywords = {
    "Bug": ["error","crash","fail","unexpected","fix","defect"],
    "Story": ["user","requirement","story","acceptance","criteria"],
    "Improvement": ["optimize","enhance","improve","performance","refactor"],
    "Technical task": ["backend","api","refactor","code","implementation","infra"],
    "Epic": ["initiative","large","milestone","project","feature group"],
    "Task": ["todo","work item","small task","assign"],
    "New Feature": ["add","support","implement","introduce","feature"],
    "Sub-task": ["child task","subtask","breakdown","dependency"],
    "Technical Debt": ["legacy","cleanup","refactor","tech debt","overdue"],
    "Documentation": ["docs","manual","guide","documentation","write"],
    "Wish": ["wish","nice to have","future","idea"],
    "Test": ["unit test","integration test","coverage","qa"],
    "Suggestion": ["suggest","recommend","proposal","feedback"],
    "Support Request": ["help","support","request","assist"],
    "Public Security Vulnerability": ["security","vulnerability","exploit","patch","CVE"],
    "Test Task": ["test","verify","validation","qa task"],
    "Milestone": ["milestone","release goal","deadline","phase"],
    "Release": ["release","deploy","publish","ship"],
    "Investigation": ["investigate","root cause","analysis","diagnose"],
    "Question": ["how","why","what","clarify","question"],
    "Build Failure": ["build failed","ci","pipeline","compile error"],
    "Problem Ticket": ["ticket","issue","incident","problem"],
    "Incident": ["outage","incident","downtime","alert","sev"],
    "Enhancement Request": ["enhance","feature request","improvement","extend"]
}

label_set = list(type_keywords.keys())
context_str = json.dumps(type_keywords, indent=2)

# ==== Prompt builder ====
def build_prompt(summary):
    return f"""
Classify the following GitHub issue summary into exactly ONE of these 24 types:
{label_set}

Here are common engineer words/phrases that usually indicate each type:
{context_str}

Return ONLY JSON like:
{{"predicted_type": "<one_of_the_types>"}}

Summary:
\"\"\"{summary}\"\"\"
"""

# ==== Load data ====
df = pd.read_csv("the_stack_github_issues_first100_summaries_gpt4o.csv")

summaries = df["summary_40w"].fillna("").astype(str)
predictions = []

for s in tqdm(summaries, desc="Classifying issues"):
    prompt = build_prompt(s)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"system","content":"You are a strict JSON-only classifier."},
                      {"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=50,
            response_format={"type": "json_object"}  # enforce JSON
        )
        out = resp.choices[0].message.content.strip()
        predicted = json.loads(out).get("predicted_type","")
    except Exception as e:
        print("Parse error:", e)
        predicted = ""
    predictions.append(predicted)

df["predicted_type"] = predictions
df.to_csv("classified_issues.csv", index=False)
print("Wrote classified_issues.csv with predictions.")
