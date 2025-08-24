#!/usr/bin/env python3
import os, re, time, argparse, json
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

SUMMARIZE_GUARDRAILS = (
    "Context: summaries will be used for type/priority classification.\n"
    "Grounding Rules:\n"
    "1) Use ONLY information present in the issue text; do NOT invent facts, error names, versions, or numbers.\n"
    "2) Prefer paraphrasing over inference; do NOT guess causes, fixes, or impact unless explicitly stated.\n"
    "3) If the text references code, logs, or links, refer to them GENERICALLY (e.g., 'stack trace shows error', 'code sample provided').\n"
    "4) Exclude greetings, usernames, labels, and pleasantries. Keep it factual and concise.\n"
    "5) Output plain text only, no lists or quotes. Cap at 40 words.\n"
)

load_dotenv(override=True)

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Summarize issues (40w) and label Type & Priority with GPT-4o.")
    p.add_argument("--input",  default=os.getenv("INPUT_CSV", "the_stack_github_issues_first100.csv"))
    p.add_argument("--output", default=os.getenv("OUTPUT_CSV", "issues_summaries_types_priorities.csv"))
    p.add_argument("--model",  default=os.getenv("MODEL", "gpt-4o"))
    p.add_argument("--sleep",  type=float, default=float(os.getenv("RATE_LIMIT_SLEEP", "0.25")))
    p.add_argument("--temp",   type=float, default=float(os.getenv("TEMPERATURE", "0.2")))
    return p.parse_args()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# -------------------- Shared utils --------------------
WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
START_COMMENT_TAGS = re.compile(r"</?issue_start>|</?issue_comment>", re.IGNORECASE)
CLOSED_TAG = re.compile(r"</?issue_closed>", re.IGNORECASE)

def clamp_words(text: str, k: int) -> str:
    toks = WORD_RE.findall(text or "")
    return " ".join(toks[:k])

def clean_and_detect_closed(text: str):
    """Remove helper tags but remember if <issue_closed> is present."""
    s = "" if not isinstance(text, str) else text
    is_closed = bool(CLOSED_TAG.search(s))
    s = START_COMMENT_TAGS.sub("", s)
    s = CLOSED_TAG.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s, is_closed

# -------------------- Summarize (40 words) --------------------
def summarize_40w(raw_text: str, model: str, temperature: float) -> str:
    cleaned, is_closed = clean_and_detect_closed(raw_text)
    sys = (
        "You are a concise technical writing assistant for software issues. "
        "Produce grounded, non-speculative summaries that strictly follow the rules below.\n\n"
        + SUMMARIZE_GUARDRAILS
    )    
    hint = " The issue is CLOSED; explicitly say it is closed." if is_closed else ""
    user = (
        f"Summarize the following GitHub issue in EXACTLY 40 words.{hint} "
        "No preamble, numbering, or quotes. Focus on the core problem and requested action.\n\n"
        f"--- ISSUE TEXT START ---\n{cleaned}\n--- ISSUE TEXT END ---"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
        temperature=temperature,
        max_tokens=160,
    )
    out = (resp.choices[0].message.content or "").strip()
    return clamp_words(out, 40)

# -------------------- Type labeling context --------------------
TYPE_KEYWORDS = {
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
TYPE_LABELS = list(TYPE_KEYWORDS.keys())
TYPE_CONTEXT = json.dumps(TYPE_KEYWORDS, indent=2)

def label_type(summary: str, model: str) -> str:
    prompt = f"""
Classify the following issue summary into exactly ONE of these 24 types:
{TYPE_LABELS}

Here are common engineer words/phrases that usually indicate each type:
{TYPE_CONTEXT}

Return ONLY JSON like:
{{"predicted_type": "<one_of_the_types>"}}

Summary:
\"\"\"{summary}\"\"\""""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":"You are a strict JSON-only classifier."},
                  {"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=60,
        response_format={"type":"json_object"},
    )
    return json.loads(resp.choices[0].message.content)["predicted_type"]

# -------------------- Priority labeling context --------------------
PRIORITY_KEYWORDS = {
    "Blocker": ["urgent","production down","showstopper","cannot proceed","system outage","highest priority"],
    "Critical": ["critical","security hole","severe","immediate fix","data loss","vulnerability"],
    "Major": ["major bug","important","high impact","core feature broken"],
    "High": ["high priority","affects many","serious issue","needs attention soon"],
    "Medium": ["medium priority","moderate impact","normal fix","not urgent"],
    "Minor": ["minor bug","low impact","cosmetic","typo"],
    "Trivial": ["trivial issue","nitpick","very low priority","small glitch"],
    "Low": ["low priority","backlog","non urgent","future fix"],
    "Lowest": ["lowest priority","icebox","almost irrelevant"],
    "None/To be reviewed": ["untriaged","to review","awaiting decision","not set"]
}
PRIORITY_LABELS = list(PRIORITY_KEYWORDS.keys())
PRIORITY_CONTEXT = json.dumps(PRIORITY_KEYWORDS, indent=2)

def label_priority(summary: str, model: str) -> str:
    prompt = f"""
Classify the following issue summary into exactly ONE of these canonical priorities:
{PRIORITY_LABELS}

Here are common engineer words/phrases that usually indicate each priority:
{PRIORITY_CONTEXT}

Return ONLY JSON in this format:
{{"predicted_priority": "<one_of_the_priorities>"}}

Summary:
\"\"\"{summary}\"\"\""""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":"You are a strict JSON-only classifier."},
                  {"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=60,
        response_format={"type":"json_object"},
    )
    return json.loads(resp.choices[0].message.content)["predicted_priority"]

# -------------------- Orchestrate --------------------
def main():
    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Missing {args.input}")

    df = pd.read_csv(args.input)
    if "content" not in df.columns or "issue_id" not in df.columns:
        raise ValueError("CSV must contain 'content' and 'issue_id' columns.")

    summaries, types, priorities = [], [], []

    for text in tqdm(df["content"].fillna("").astype(str), desc="Summarize + label"):
        try:
            s40 = summarize_40w(text, args.model, args.temp)
        except Exception:
            s40 = ""
        summaries.append(s40)

        # Type & priority only use the generated 40-word summary
        try:
            t = label_type(s40, args.model) if s40 else ""
        except Exception:
            t = ""
        types.append(t)

        try:
            p = label_priority(s40, args.model) if s40 else ""
        except Exception:
            p = ""
        priorities.append(p)

        if args.sleep > 0:
            time.sleep(args.sleep)

    out = pd.DataFrame({
        "issue_id": df["issue_id"],
        "summary_40w": summaries,
        "predicted_type": types,
        "predicted_priority": priorities,
    })
    out.to_csv(args.output, index=False)
    print(f"Done. Wrote {args.output} with {len(out)} rows.")

if __name__ == "__main__":
    main()
