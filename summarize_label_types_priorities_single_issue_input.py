#!/usr/bin/env python3
import os, re, time, argparse, json
import anthropic
from groq import Groq
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
import json
'''

python summarize_label_types_priorities_single_issue_input.py --input "C:/Users/dtian/GitHub_Issues_Prioritisation/longest_issues/issue_id_354702553.txt" --output prediction.txt --model claude-3-opus-20240229 --log-cleaned

'''
SUMMARIZE_GUARDRAILS = (
    "Context: summaries will be used for type/priority classification.\n"
    "Grounding Rules:\n"
    "1) Use ONLY information present in the issue text; do NOT invent facts, error names, versions, or numbers.\n"
    "2) Prefer paraphrasing over inference; do NOT guess causes, fixes, or impact unless explicitly stated.\n"
    "3) If the text references code, logs, or links, refer to them GENERICALLY (e.g., 'stack trace shows error', 'code sample provided').\n"
    "4) Exclude greetings, usernames, labels, and pleasantries. Keep it factual and concise.\n"
    "5) Output plain text only, no lists or quotes. Cap at 40 words.\n"
)

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

# -------------------- Shared utils --------------------
WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
START_COMMENT_TAGS = re.compile(r"</?issue_start>|</?issue_comment>", re.IGNORECASE)
CLOSED_TAG = re.compile(r"</?issue_closed>", re.IGNORECASE)

load_dotenv(override=True)

def parse_args():
    p = argparse.ArgumentParser(description="Summarize issues (40w) and label Type & Priority with a LLM (default: GPT-4o).")
    p.add_argument("--input",  default=os.getenv("INPUT_CSV", "issue_id_549374190.txt"))
    p.add_argument("--output", default=os.getenv("OUTPUT_CSV", "single_issue_summaries_types_priorities.txt"))
    p.add_argument("--model",  default=os.getenv("MODEL", "gpt-4o"))
    p.add_argument("--sleep",  type=float, default=float(os.getenv("RATE_LIMIT_SLEEP", "0.25")))
    p.add_argument("--temp",   type=float, default=float(os.getenv("TEMPERATURE", "0.2")))
    p.add_argument("--log-cleaned", action="store_true",
                   help="If set, save cleaned issue texts into cleaned_issues.log")
    p.add_argument("--count-tokens", action="store_true",
                   help="If set, count and display tokens in cleaned data")
    return p.parse_args()

def clamp_words(text: str, k: int) -> str:
    toks = WORD_RE.findall(text or "")
    return " ".join(toks[:k])

def count_tokens(text: str) -> int:
    """Count tokens using a simple word-based approach."""
    return len(WORD_RE.findall(text or ""))

def clean_and_detect_closed(text: str, log: bool = False, count_tokens_flag: bool = False):
    """
    Remove helper tags, detect <issue_closed>, and normalize non-conversational content
    by replacing it with placeholders so summarization/classification stays focused.

    Placeholders introduced:
      [CODE_BLOCK]       fenced code blocks (```...``` or ~~~...~~~)
      [CODE]             inline code `...`
      [STACK_TRACE]      multi-line Python traceback blocks
      [STACK_TRACE_LINE] single stack-trace style lines (e.g., "at ...", "Exception ...")
      [URL]              http(s)://... and www.... links
      [FILE_PATH]        Unix/Windows file paths
      [COMMIT_HASH]      git SHAs / long hex identifiers (8–40 hex chars)
      [VERSION]          semantic-like version strings (e.g., 1.2.3, v3.11.2, 2.0.0-rc1)
      [ERROR_CODE]       common error code tokens (E1234, ERR_FOO, HTTP 500, ERROR 404)
      [ENV_INFO_LINE]    environment/config lines (OS:, CPU:, Python:, Node:, CUDA:, etc.)
      [EMAIL]            email addresses
      [USER]             @mentions
      [LOG_LINE]         timestamped or log-level lines (INFO/WARN/DEBUG/ERROR ...)

    Returns:
      normalized_text, is_closed
    """
    s = "" if not isinstance(text, str) else text

    # detect closed tag first
    is_closed = bool(CLOSED_TAG.search(s))

    # remove helper tags
    s = START_COMMENT_TAGS.sub("", s)
    s = CLOSED_TAG.sub("", s)

    # --- Replace fenced code blocks (```...``` or ~~~...~~~) ---
    s = re.sub(r"```.*?```", "[CODE_BLOCK]", s, flags=re.DOTALL)
    s = re.sub(r"~~~.*?~~~", "[CODE_BLOCK]", s, flags=re.DOTALL)

    # --- Replace inline code ---
    s = re.sub(r"`[^`]+`", "[CODE]", s)

    # --- Python traceback blocks ---
    s = re.sub(
        r"(Traceback \(most recent call last\):[\s\S]+?)(?=\n\S|\Z)",
        "[STACK_TRACE]",
        s,
        flags=re.MULTILINE
    )

    # --- Java/.NET style stack lines ---
    s = re.sub(r"(?m)^\s*at\s+.+$", "[STACK_TRACE_LINE]", s)
    # Generic "Exception ..." lines
    s = re.sub(r"(?mi)^[^\n]*exception[^\n]*$", "[STACK_TRACE_LINE]", s)

    # --- URLs (http/https + www.) ---
    s = re.sub(r"https?://\S+", "[URL]", s)
    s = re.sub(r"\bwww\.[^\s)]+", "[URL]", s)

    # --- File paths (Unix and Windows) ---
    s = re.sub(r"(?:(?:[A-Za-z]:\\|/)[^\s'\"`()<>\[\]]+)", "[FILE_PATH]", s)

    # --- Commit hashes / long hex IDs (8–40 hex) ---
    s = re.sub(r"\b[0-9a-f]{8,40}\b", "[COMMIT_HASH]", s, flags=re.IGNORECASE)

    # --- Version strings (semver-ish) ---
    s = re.sub(r"\bv?\d+(?:\.\d+){1,3}(?:-[A-Za-z0-9.\-]+)?\b", "[VERSION]", s)

    # --- Error codes ---
    s = re.sub(r"\bE\d{3,}\b", "[ERROR_CODE]", s)
    s = re.sub(r"\bERR[_\-][A-Z0-9_\-]+\b", "[ERROR_CODE]", s)
    s = re.sub(r"\bERROR\s*\d{3,}\b", "[ERROR_CODE]", s, flags=re.IGNORECASE)
    s = re.sub(r"\bHTTP\s*(?:4|5)\d{2}\b", "[ERROR_CODE]", s, flags=re.IGNORECASE)

    # --- Environment/config lines (one line each) ---
    env_keys = r"(OS|CPU|GPU|Memory|RAM|Processor|Platform|Kernel|Distro|Node\.?js?|Python|Java|Go|Rust|npm|yarn|pip|conda|CUDA|cuDNN|Driver|Browser|Chrome|Firefox|Safari|Edge|Android|iOS|macOS|Windows|Ubuntu|Debian)"
    s = re.sub(rf"(?mi)^\s*{env_keys}\s*[:=].*$", "[ENV_INFO_LINE]", s)

    # --- Emails and @mentions ---
    s = re.sub(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "[EMAIL]", s)
    s = re.sub(r"(?<!\w)@[A-Za-z0-9_\-]+", "[USER]", s)

    # --- Log-like lines (timestamps, log levels) ---
    s = re.sub(r"(?m)^\s*(?:\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?|\d{2}:\d{2}:\d{2}|INFO|WARN|WARNING|DEBUG|ERROR)\b.*$", "[LOG_LINE]", s)

    # --- Clean table/markdown formatting characters ---
    # Remove table separators like | ------- | ---------------\ 
    s = re.sub(r"\|\s*-+\s*\\?\s*\|?", "", s)
    s = re.sub(r"\|\s*-+\s*\\?\s*", "", s)
    
    # Remove empty markdown link/image syntax []()
    s = re.sub(r"\[\]\(\)", "", s)
    
    # Remove :x: emoji pattern
    s = re.sub(r":x:", "", s)
    
    # Clean up remaining table pipes and markdown artifacts
    s = re.sub(r"\|\s*\|", "", s)  # Remove empty table cells
    s = re.sub(r"^\s*\|\s*", "", s, flags=re.MULTILINE)  # Remove leading pipes
    s = re.sub(r"\s*\|\s*$", "", s, flags=re.MULTILINE)  # Remove trailing pipes

    # Collapse repeated placeholders
    s = re.sub(r"(?:\[CODE_BLOCK\]\s*){2,}", "[CODE_BLOCK] ", s)
    s = re.sub(r"(?:\[STACK_TRACE(?:_LINE)?\]\s*){2,}", "[STACK_TRACE] ", s)
    s = re.sub(r"(?:\[LOG_LINE\]\s*){2,}", "[LOG_LINE] ", s)
    s = re.sub(r"(?:\[ENV_INFO_LINE\]\s*){2,}", "[ENV_INFO_LINE] ", s)

    # --- Replace repeated patterns with counts ---
    def replace_repeated_patterns(text):
        """Replace sequences of the same pattern separated by delimiters with a count."""
        # Split text into tokens
        tokens = text.split()
        if len(tokens) <= 1:
            return text
        
        result = []
        i = 0
        while i < len(tokens):
            current_token = tokens[i]
            count = 1
            
            # Count consecutive occurrences of the same token
            j = i + 1
            while j < len(tokens) and tokens[j] == current_token:
                count += 1
                j += 1
            
            # If we found repetitions, replace with count notation
            if count > 2:  # Only replace if there are more than 2 occurrences
                result.append(f"{current_token}(x{count})")
            else:
                # Add tokens individually if not enough repetitions
                for k in range(count):
                    result.append(current_token)
            
            i = j
        
        return " ".join(result)
    
    s = replace_repeated_patterns(s)

    # --- Replace non-English characters ---
    def replace_non_english_chars(text):
        """Replace sequences of non-English characters with a note."""
        # Define pattern for non-ASCII characters (outside basic English)
        non_english_pattern = r'[^\x00-\x7F]+'
        
        # Find all non-English character sequences
        matches = re.findall(non_english_pattern, text)
        if matches:
            # Replace all non-English character sequences with a note
            text = re.sub(non_english_pattern, '[NON_ENGLISH_CHARS]', text)
        
        return text
    
    s = replace_non_english_chars(s)

    # Final whitespace normalize
    s = re.sub(r"\s+", " ", s).strip()      
    
    if log:
        try:
            with open("cleaned_issues.log", "w", encoding="utf-8") as f:
                f.write(s + "\n---\n")
                token_count = count_tokens(s)
                f.write(f"[INFO] Token count in cleaned data: {token_count}")
        except Exception as e:
            print(f"[WARN] Could not log cleaned text: {e}")

    return s, is_closed

def make_llm_call(prompt: str, system_msg: str, model: str, temperature: float, client: OpenAI, max_tokens: int = 160) -> str:
    """Helper function to make LLM calls with consistent interface across different models."""
    if model == "claude-3-5-sonnet-latest":
        parts = [{"type": "text", "text": prompt}]
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_msg,
            temperature=temperature,
            messages=[{"role": "user", "content": parts}]
        )
        return "".join([block.text for block in resp.content])
    elif model == "grok-4":
        from xai_sdk.chat import user, system
        chat = client.chat.create(model=model, temperature=temperature)
        chat.append(system(system_msg))
        chat.append(user(prompt))
        response = chat.sample()
        return response.content        
    else:        
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system_msg},{"role":"user","content":prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return (resp.choices[0].message.content or "").strip()


def summarize_40w(raw_text: str, model: str, temperature: float, client: OpenAI, log: bool = False, count_tokens_flag: bool = False) -> str:
    cleaned, is_closed = clean_and_detect_closed(raw_text, log=log, count_tokens_flag=count_tokens_flag)
    
    # Check if content is too large (>= 20k tokens)
    token_count = count_tokens(cleaned)
    
    if token_count >= 20000:
        # Split content into roughly equal parts
        words = cleaned.split()
        total_words = len(words)

        # Calculate number of chunks (aim for ~20k tokens per chunk to be safe)
        num_chunks = max(2, (token_count // 20000) + 1)
        words_per_chunk = total_words // num_chunks
        
        chunks = []
        for i in range(num_chunks):
            start_idx = i * words_per_chunk
            if i == num_chunks - 1:  # Last chunk gets remaining words
                end_idx = total_words
            else:
                end_idx = (i + 1) * words_per_chunk
            
            chunk = " ".join(words[start_idx:end_idx])
            chunks.append(chunk)
        
        # Summarize each chunk
        chunk_summaries = []
        sys_chunk = (
            "You are a technical writing assistant. Summarize the following GitHub issue content section "
            "in 20-30 words. Focus on key problems, errors, and requested actions. "
            "Be factual and concise.\n\n" + SUMMARIZE_GUARDRAILS
        )
        
        for i, chunk in enumerate(chunks):
            user_chunk = (
                f"Summarize this section ({i+1}/{num_chunks}) of a GitHub issue in 20-30 words. "
                "Focus on the main problems and actions mentioned:\n\n"
                f"--- SECTION START ---\n{chunk}\n--- SECTION END ---"
            )
            
            try:
                chunk_summary = make_llm_call(user_chunk, sys_chunk, model, temperature, client, max_tokens=120)
                chunk_summaries.append(chunk_summary.strip())
            except Exception:
                chunk_summaries.append(f"[Error summarizing section {i+1}]")
        
        # Merge chunk summaries into final summary
        merged_content = " ".join(chunk_summaries)
        
        sys_final = (
            "You are a concise technical writing assistant for software issues. "
            "Merge the following section summaries into a single coherent 40-word summary.\n\n"
            + SUMMARIZE_GUARDRAILS
        )
        hint = " The issue is CLOSED; explicitly say it is closed." if is_closed else ""
        user_final = (
            f"Merge these section summaries into EXACTLY 40 words.{hint} "
            "Create a coherent summary focusing on the main problem and requested action:\n\n"
            f"--- SECTION SUMMARIES ---\n{merged_content}\n--- END SUMMARIES ---"
        )
        
        try:
            out = make_llm_call(user_final, sys_final, model, temperature, client, max_tokens=160)
        except Exception:
            # Fallback: truncate merged content if final summarization fails
            out = clamp_words(merged_content, 40)
            
    else:
        # Original logic for content < 20k tokens
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
        
        try:
            out = make_llm_call(user, sys, model, temperature, client, max_tokens=160)
        except Exception:
            out = ""

    return clamp_words(out, 40)

def extract_predicted_value(text: str, key: str = "predicted_type") -> str:
    """
    Extracts a value from a JSON-like text string under a given key.
    Returns 'Unknown' if parsing fails or key is missing.
    """
    try:
        data = json.loads(text.strip())
        return data.get(key, "Unknown")
    except json.JSONDecodeError:
        return "Unknown"

def label_type(summary: str, model: str, client: OpenAI) -> str:
    prompt = f"""
Classify the following issue summary into exactly ONE of these 24 types:
{TYPE_LABELS}

Here are common engineer words/phrases that usually indicate each type:
{TYPE_CONTEXT}

Return ONLY JSON like:
{{"predicted_type": "<one_of_the_types>"}}

Summary:
\"\"\"{summary}\"\"\""""
    if model == "claude-3-5-sonnet-latest":
        parts = [{"type": "text", "text": prompt}]
        resp = client.messages.create(
            model=model,
            max_tokens=160,
            temperature=0.0,
            messages=[{"role": "user", "content": parts}]
        )
        return extract_predicted_value(resp.content[0].text,key="predicted_type")
    elif model == "grok-4":
        from xai_sdk.chat import user
        chat = client.chat.create(model=model, temperature=0)
        chat.append(user(prompt))
        response = chat.sample()
        return extract_predicted_value(response.content,key="predicted_type")
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":"You are a strict JSON-only classifier."},
                    {"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=60,
            response_format={"type":"json_object"},
        )
        return json.loads(resp.choices[0].message.content)["predicted_type"]

def label_priority(summary: str, model: str, client: OpenAI) -> str:
    prompt = f"""
Classify the following issue summary into exactly ONE of these canonical priorities:
{PRIORITY_LABELS}

Here are common engineer words/phrases that usually indicate each priority:
{PRIORITY_CONTEXT}

One-shot example:
Summary: "This issue has been resolved and is closed. No further action required."
Expected output: {{"predicted_priority": "None/To be reviewed"}}

Now classify the new summary below.

Return ONLY JSON in this format:
{{"predicted_priority": "<one_of_the_priorities>"}}

Summary:
\"\"\"{summary}\"\"\""""
    if model == "claude-3-5-sonnet-latest" or model == "claude-opus-4-20250514":
       parts = [{"type": "text", "text": prompt}]
       resp = client.messages.create(
           model=model,
           max_tokens=160,
           temperature=0.0,
           messages=[{"role": "user", "content": parts}]
       )
       return extract_predicted_value(resp.content[0].text,key="predicted_priority")
    elif model == "grok-4":
        from xai_sdk.chat import user 
        chat = client.chat.create(model=model, temperature=0)
        chat.append(user(prompt))
        response = chat.sample()
        return extract_predicted_value(response.content,key="predicted_priority")
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":"You are a strict JSON-only classifier."},
                      {"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=60,
            response_format={"type":"json_object"}
        )
        return json.loads(resp.choices[0].message.content)["predicted_priority"]

# -------------------- Orchestrate --------------------

def main():
    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Missing {args.input}")

    if args.input.endswith(".txt"):
        # Handle plain text input
        with open(args.input, "r", encoding="utf-8") as f:
            issue_text = f.read()
        df = pd.DataFrame([{"issue_id": 1, "content": issue_text}])
    else:
        # Assume CSV input
        df = pd.read_csv(args.input)
        if "content" not in df.columns or "issue_id" not in df.columns:
            raise ValueError("CSV must contain 'content' and 'issue_id' columns.")
    
    if args.model == "gpt-4o" or args.model == "gpt-5":
        client = OpenAI()
    elif args.model == "claude-3-5-sonnet-latest" or args.model == "claude-opus-4-20250514":
        client = anthropic.Anthropic()
    elif args.model == "gemini-2.0-flash": #"gemini-1.5-pro-latest"
        client = OpenAI(api_key=os.environ["GOOGLE_API_KEY"], base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    elif args.model == "grok-4":
        from xai_sdk import Client
        client = Client(api_key=os.environ["XAI_API_KEY"])
    elif args.model == "llama-3.3-70b-versatile":
        client = Groq()
    elif args.model == "deepseek-chat":
        client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com/v1")
    else:
        import sys
        sys.exit(f"Unknown model: {args.model}")

    summaries, types, priorities = [], [], []

    for text in tqdm(df["content"].fillna("").astype(str), desc="Summarize + label"):
        try:
            s40 = summarize_40w(text, args.model, args.temp, client, log=args.log_cleaned, count_tokens_flag=args.count_tokens)
        except Exception:
            s40 = ""
        summaries.append(s40)

        try:
            t = label_type(s40, args.model, client) if s40 else ""
        except Exception:
            t = ""
        types.append(t)

        try:
            p = label_priority(s40, args.model, client) if s40 else ""
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
