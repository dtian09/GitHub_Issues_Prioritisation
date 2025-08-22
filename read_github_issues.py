# pip install datasets pandas
from datasets import load_dataset
import pandas as pd
import os

#cmd: export HF_TOKEN=token

tok = os.getenv("HF_TOKEN")
ds = load_dataset("bigcode/the-stack-github-issues", split="train", streaming=True, token=tok)

rows = []
for i, ex in enumerate(ds):
    if i >= 100:
        break

    # Pick a representative datetime:
    # Prefer the 'opened' event; otherwise fall back to the first event's datetime if present.
    dt = None
    if ex.get("events"):
        # find 'opened'
        opened = next((ev for ev in ex["events"] if ev.get("action") == "opened"), None)
        dt = (opened or ex["events"][0]).get("datetime")

    rows.append({
        "content": ex.get("content"),
        "datetime": dt,
        "issue_id": ex.get("issue_id"),
        "pull_request": ex.get("pull_request"),  # may be None if it's a regular issue
        "repo": ex.get("repo"),
        "usernames": ex.get("usernames"),
    })

df = pd.DataFrame(rows, columns=["content", "datetime", "issue_id", "pull_request", "repo", "usernames"])
df.to_csv("the_stack_github_issues_first100.csv", index=False)
print(df.head(10))
