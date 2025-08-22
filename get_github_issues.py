# get_github_issues_clean.py
# pip install datasets pandas tqdm python-dotenv
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import os, re, csv

load_dotenv(override=True)

# ---------- Config ----------
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
SPLIT = "train"
STREAMING = True
N = 310000 #100                  # set to an int for first N; or use None for all_data
OUT_ALL = "the_stack_github_issues_alldata.csv"
OUT_N   = lambda n: f"the_stack_github_issues_first{n}.csv"

# ---------- Control-char cleaner ----------
# Remove C0 controls except tab/newline; also strip NULLs explicitly.
_CTRL_EXCEPT_TNL = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")
def clean_text(x):
    if x is None:
        return ""
    if not isinstance(x, str):
        try:
            x = str(x)
        except Exception:
            return ""
    # Normalize Windows newlines to \n
    x = x.replace("\r\n", "\n").replace("\r", "\n")
    # Strip NULLs and other control chars (keep \n and \t)
    x = x.replace("\x00", "")
    x = _CTRL_EXCEPT_TNL.sub("", x)
    # Optionally collapse long runs of whitespace but keep newlines
    # x = re.sub(r"[ \t]+", " ", x)
    return x.strip()

# ---------- Load dataset ----------
ds = load_dataset(
    "bigcode/the-stack-github-issues",
    split=SPLIT,
    streaming=STREAMING,
    token=HF_TOKEN
)

rows = []

if N is None:
    # Unknown total → tqdm without total
    for ex in tqdm(ds, desc="Fetching issues (all)"):
        rows.append({
            "issue_id": ex.get("issue_id"),
            "content": clean_text(ex.get("content"))
        })
    out_path = OUT_ALL
else:
    for i, ex in enumerate(tqdm(ds, total=N, desc=f"Fetching first {N} issues")):
        if i >= N:
            break
        rows.append({
            "issue_id": ex.get("issue_id"),
            "content": clean_text(ex.get("content"))
        })
    out_path = OUT_N(N)

# ---------- Build DataFrame & write CSV safely ----------
df = pd.DataFrame(rows, columns=["issue_id", "content"])

# If you want to double-sanitize any stray non-UTF8 sequences during write:
# (Pandas will handle utf-8; if upstream text is odd, you can defensively re-encode.)
df["content"] = df["content"].astype(str).map(lambda s: s.encode("utf-8", "replace").decode("utf-8"))

df.to_csv(
    out_path,
    index=False,
    escapechar="\\",
    quoting=csv.QUOTE_ALL,
    encoding="utf-8",
    lineterminator="\n"  # consistent newlines
)

print(f"✅ Saved {out_path} with {len(df)} issues")