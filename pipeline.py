'''
Basic usage with default settings
python pipeline.py --input sample_issue_ids.txt

Advanced usage with custom parameters
python pipeline.py --input my_issue_ids.txt --model grok-4 --batch-size 10 --max-workers 4 --sleep 0.5

# Process 1 issue at a time (no parallelism)
python pipeline.py --input sample_issue_ids.txt --max-workers 1

# Process 2 issues at a time
python pipeline.py --input sample_issue_ids.txt --max-workers 2

# Process 4 issues at a time (good for rate limiting)
python pipeline.py --input sample_issue_ids.txt --max-workers 4

# Default: 8 issues at a time
python pipeline.py --input sample_issue_ids.txt

# Conservative approach
python pipeline.py --input sample_issue_ids.txt --max-workers 1 --sleep 1.0

# Moderate approach  
python pipeline.py --input sample_issue_ids.txt --max-workers 2 --sleep 0.5

# Aggressive (might hit rate limits)
python pipeline.py --input sample_issue_ids.txt --max-workers 4 --sleep 0.25
'''

import os, re, csv, mysql.connector
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from xai_sdk import Client

# Import functions from the single issue processor
from summarize_label_types_priorities_single_issue import (
    summarize_40w, label_type, label_priority
)

load_dotenv(override=True)

# ---------- Config ----------
def parse_args():
    parser = argparse.ArgumentParser(description="Process GitHub issues from MySQL database using issue IDs from a text file.")
    parser.add_argument('--input', type=str, default="issue_ids.txt", 
                       help='Text file containing issue IDs (one per line)')
    parser.add_argument('--model', type=str, default="grok-4", 
                       help='Model to use for processing (default: grok-4)')
    parser.add_argument('--batch-size', type=int, default=20, 
                       help='Batch size for database operations (default: 20)')
    parser.add_argument('--max-workers', type=int, default=8, 
                       help='Maximum worker threads (default: 8)')
    parser.add_argument('--sleep', type=float, default=0.25, 
                       help='Sleep between API calls (default: 0.25)')
    return parser.parse_args()

MODEL = "grok-4"  # Default to grok-4 as requested
TEMP = float(os.getenv("TEMPERATURE", "0.2"))

# ---------- MySQL Config ----------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "username")
DB_PASS = os.getenv("DB_PASS", "password")
DB_NAME = os.getenv("DB_NAME", "github_issues_db")

# ---------- Control-char cleaner ----------
_CTRL_EXCEPT_TNL = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")
def clean_text(x):
    if x is None:
        return ""
    if not isinstance(x, str):
        try:
            x = str(x)
        except Exception:
            return ""
    x = x.replace("\r\n", "\n").replace("\r", "\n")
    x = x.replace("\x00", "")
    x = _CTRL_EXCEPT_TNL.sub("", x)
    return x.strip()

# ---------- Read issue IDs from file ----------
def read_issue_ids(file_path):
    """Read issue IDs from a text file, one ID per line."""
    issue_ids = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and line.isdigit():
                    issue_ids.append(int(line))
        print(f"✅ Read {len(issue_ids)} issue IDs from {file_path}")
        return issue_ids
    except FileNotFoundError:
        raise FileNotFoundError(f"Issue IDs file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading issue IDs file: {e}")

# ---------- Fetch issues from MySQL ----------
def fetch_issues_from_db(issue_ids):
    """Fetch issue contents from MySQL database using issue IDs."""
    conn = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME
    )
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Create placeholders for the IN clause
        placeholders = ','.join(['%s'] * len(issue_ids))
        query = f"SELECT issue_id, content FROM issue WHERE issue_id IN ({placeholders})"
        
        cursor.execute(query, issue_ids)
        results = cursor.fetchall()
        
        print(f"✅ Retrieved {len(results)} issues from database")
        return results
    
    finally:
        cursor.close()
        conn.close()

# ---------- Initialize the appropriate client ----------
def get_client(model):
    """Initialize the appropriate client based on the model."""
    if model == "grok-4":
        return Client(api_key=os.environ["XAI_API_KEY"])
    elif model in ["gpt-4o", "gpt-5"]:
        from openai import OpenAI
        return OpenAI()
    elif model in ["claude-3-5-sonnet-latest", "claude-opus-4-20250514"]:
        import anthropic
        return anthropic.Anthropic()
    elif model == "gemini-2.0-flash":
        from openai import OpenAI
        return OpenAI(api_key=os.environ["GOOGLE_API_KEY"], base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    elif model == "llama-3.3-70b-versatile":
        from groq import Groq
        return Groq()
    elif model == "deepseek-chat":
        from openai import OpenAI
        return OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com/v1")
    else:
        raise ValueError(f"Unknown model: {model}")

# ---------- Process single issue ----------
def process_issue(issue_data, model, temp, client, sleep_time):
    """Process a single issue: clean, summarize, label type and priority."""
    issue_id = issue_data["issue_id"]
    content = clean_text(issue_data["content"])
    
    try:
        # Generate summary using grok-4
        summary = summarize_40w(content, model, temp, client, log=False, count_tokens_flag=False)
    except Exception as e:
        print(f"⚠️  Error summarizing issue {issue_id}: {e}")
        summary = ""
    
    try:
        # Label type
        predicted_type = label_type(summary, model, client) if summary else ""
    except Exception as e:
        print(f"⚠️  Error labeling type for issue {issue_id}: {e}")
        predicted_type = ""
    
    try:
        # Label priority
        predicted_priority = label_priority(summary, model, client) if summary else ""
    except Exception as e:
        print(f"⚠️  Error labeling priority for issue {issue_id}: {e}")
        predicted_priority = ""
    
    # Sleep to respect rate limits
    if sleep_time > 0:
        import time
        time.sleep(sleep_time)
    
    return {
        "issue_id": issue_id,
        "summary": summary,
        "type": predicted_type,
        "priority": predicted_priority
    }

# ---------- Update database with results ----------
def update_issues_batch(cursor, batch):
    """Update issues in database with summary, type, and priority."""
    sql = (
        "UPDATE issue SET summary = %s, type = %s, priority = %s "
        "WHERE issue_id = %s"
    )
    
    # Prepare data tuples for batch update
    update_data = []
    for result in batch:
        update_data.append((
            result["summary"],
            result["type"], 
            result["priority"],
            result["issue_id"]
        ))
    
    cursor.executemany(sql, update_data)

def main():
    args = parse_args()
    
    print(f"🚀 Starting pipeline with model: {args.model}")
    print(f"📁 Reading issue IDs from: {args.input}")
    
    # Read issue IDs from file
    issue_ids = read_issue_ids(args.input)
    
    if not issue_ids:
        print("❌ No valid issue IDs found in the input file")
        return
    
    # Fetch issues from database
    issues = fetch_issues_from_db(issue_ids)
    
    if not issues:
        print("❌ No issues found in database for the provided IDs")
        return
    
    # Initialize client for the specified model
    client = get_client(args.model)
    
    # Process issues
    print(f"🔄 Processing {len(issues)} issues...")
    
    conn = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME
    )
    
    try:
        cursor = conn.cursor()
        batch = []
        
        # Process issues with threading for better performance
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all tasks
            future_to_issue = {
                executor.submit(
                    process_issue, 
                    issue, 
                    args.model, 
                    TEMP, 
                    client, 
                    args.sleep
                ): issue for issue in issues
            }
            
            # Collect results and batch update database
            for future in tqdm(as_completed(future_to_issue), total=len(issues), desc="Processing issues"):
                try:
                    result = future.result()
                    batch.append(result)
                    
                    # Update database in batches
                    if len(batch) >= args.batch_size:
                        update_issues_batch(cursor, batch)
                        conn.commit()
                        print(f"✅ Updated batch of {len(batch)} issues")
                        batch = []
                        
                except Exception as e:
                    issue = future_to_issue[future]
                    print(f"❌ Error processing issue {issue['issue_id']}: {e}")
            
            # Update any remaining issues in the final batch
            if batch:
                update_issues_batch(cursor, batch)
                conn.commit()
                print(f"✅ Updated final batch of {len(batch)} issues")
    
    finally:
        cursor.close()
        conn.close()
    
    print(f"🎉 Pipeline completed! Processed {len(issues)} issues with {args.model}")

if __name__ == "__main__":
    main()