#!/usr/bin/env python3
"""
Script to label issue types based on personal judgement using the provided mapping table.
Analyzes summaries in CSV files and adds a 'personal judgement' column with the appropriate issue type.
"""

import pandas as pd
import os
import re
from typing import List, Dict, Tuple
import glob

# Issue type mapping based on the provided table
ISSUE_TYPE_MAPPING = {
    'Bug': {
        'keywords': ['crash', 'error', 'fail', 'unexpected behavior', 'exception', 'broken', 'not working', 'incorrect', 'wrong'],
        'patterns': [r'\berror\b', r'\bfail\b', r'\bcrash\b', r'\bbug\b', r'\bbroken\b', r'\bissue\b', r'\bproblem\b']
    },
    'New Feature': {
        'keywords': ['add', 'new', 'feature', 'implement', 'support', 'introduce', 'create'],
        'patterns': [r'\badd\s+\w+', r'\bnew\s+feature\b', r'\bimplement\b', r'\bsupport\s+for\b', r'\bintroduce\b']
    },
    'Enhancement Request': {
        'keywords': ['improve', 'enhance', 'better', 'upgrade', 'optimize', 'extend'],
        'patterns': [r'\bimprove\b', r'\benhance\b', r'\bupgrade\b', r'\boptimize\b', r'\bbetter\b']
    },
    'Improvement': {
        'keywords': ['ui', 'ux', 'alignment', 'layout', 'design', 'refinement', 'polish'],
        'patterns': [r'\bui\b', r'\bux\b', r'\balignment\b', r'\blayout\b', r'\bdesign\b']
    },
    'Documentation': {
        'keywords': ['documentation', 'doc', 'tutorial', 'guide', 'manual', 'readme', 'help', 'instruction'],
        'patterns': [r'\bdoc\b', r'\bdocumentation\b', r'\btutorial\b', r'\bguide\b', r'\bmanual\b', r'\breadme\b']
    },
    'Technical Debt': {
        'keywords': ['refactor', 'cleanup', 'maintainability', 'debt', 'legacy', 'deprecate', 'remove'],
        'patterns': [r'\brefactor\b', r'\bcleanup\b', r'\bdebt\b', r'\blegacy\b', r'\bdeprecate\b']
    },
    'Question': {
        'keywords': ['question', 'how to', 'help', 'clarification', 'unclear', 'understand'],
        'patterns': [r'\bquestion\b', r'\bhow\s+to\b', r'\bhelp\b', r'\bclarification\b', r'\bunclear\b']
    },
    'Build Failure': {
        'keywords': ['build', 'compile', 'ci', 'pipeline', 'dependency', 'installation'],
        'patterns': [r'\bbuild\s+fail\b', r'\bcompile\s+error\b', r'\bci\b', r'\bpipeline\b', r'\bdependency\b']
    },
    'Support Request': {
        'keywords': ['support', 'help', 'assistance', 'configure', 'setup', 'installation'],
        'patterns': [r'\bsupport\b', r'\bhelp\b', r'\bassistance\b', r'\bconfigure\b', r'\bsetup\b']
    },
    'Suggestion': {
        'keywords': ['suggest', 'proposal', 'idea', 'what if', 'consider', 'maybe'],
        'patterns': [r'\bsuggest\b', r'\bproposal\b', r'\bidea\b', r'\bwhat\s+if\b', r'\bconsider\b']
    },
    'Task': {
        'keywords': ['task', 'todo', 'implement', 'work item', 'action'],
        'patterns': [r'\btask\b', r'\btodo\b', r'\bwork\s+item\b', r'\baction\b']
    }
}

def analyze_text_for_issue_type(text: str) -> str:
    """
    Analyze text content to determine the most likely issue type based on keywords and patterns.
    """
    if not text or pd.isna(text):
        return "Unknown"
    
    text_lower = text.lower()
    scores = {}
    
    # Score each issue type based on keyword and pattern matches
    for issue_type, config in ISSUE_TYPE_MAPPING.items():
        score = 0
        
        # Check keywords
        for keyword in config['keywords']:
            if keyword in text_lower:
                score += 1
        
        # Check patterns
        for pattern in config['patterns']:
            matches = len(re.findall(pattern, text_lower))
            score += matches * 2  # Patterns have higher weight
        
        scores[issue_type] = score
    
    # Return the issue type with highest score, or default classification
    if not scores or max(scores.values()) == 0:
        return classify_by_context(text_lower)
    
    return max(scores, key=scores.get)

def classify_by_context(text: str) -> str:
    """
    Additional context-based classification for edge cases.
    """
    # Check for specific patterns that indicate certain types
    if any(word in text for word in ['warning', 'scheme', 'mismatch', 'backend']):
        return "Bug"
    
    if any(word in text for word in ['url', 'add', 'username']):
        return "New Feature"
    
    if any(word in text for word in ['cannot', 'unable', 'minimal', 'no details']):
        return "Question"
    
    if any(word in text for word in ['broker', 'client', 'configuration', 'parameter']):
        return "Bug"  # Technical issues are often bugs
    
    return "Task"  # Default fallback

def analyze_csv_summaries(csv_path: str) -> str:
    """
    Analyze all summaries in a CSV file to determine the overall issue type.
    """
    try:
        df = pd.read_csv(csv_path)
        
        if 'Summary_40w' not in df.columns:
            return "Unknown"
        
        # Get all summaries
        summaries = df['Summary_40w'].dropna().tolist()
        
        if not summaries:
            return "Unknown"
        
        # Combine all summaries for analysis
        combined_text = " ".join(summaries)
        
        # Analyze the combined text
        issue_type = analyze_text_for_issue_type(combined_text)
        
        # Additional validation based on existing predictions
        if 'Predicted Type' in df.columns:
            predicted_types = df['Predicted Type'].dropna().tolist()
            # If most models agree on a type, consider that
            type_counts = {}
            for ptype in predicted_types:
                type_counts[ptype] = type_counts.get(ptype, 0) + 1
            
            if type_counts:
                most_common = max(type_counts, key=type_counts.get)
                # If there's strong consensus and it's a known type, use it
                if type_counts[most_common] >= len(predicted_types) * 0.6:
                    if most_common in ['Bug', 'New Feature', 'Enhancement Request', 'Documentation']:
                        issue_type = most_common
        
        return issue_type
        
    except Exception as e:
        print(f"Error analyzing {csv_path}: {e}")
        return "Unknown"

def add_personal_judgement_column(csv_path: str) -> bool:
    """
    Add a 'personal judgement' column to the CSV file with the determined issue type.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Determine the issue type
        issue_type = analyze_csv_summaries(csv_path)
        
        # Add the personal judgement column
        df['Personal Judgement'] = issue_type
        
        # Save the updated CSV
        df.to_csv(csv_path, index=False)
        
        print(f"✅ Updated {os.path.basename(csv_path)} - Classified as: {issue_type}")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {csv_path}: {e}")
        return False

def process_directory(directory_path: str) -> Dict[str, int]:
    """
    Process all CSV files in a directory and add personal judgement columns.
    """
    results = {'success': 0, 'error': 0}
    
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return results
    
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return results
    
    print(f"\n📁 Processing directory: {directory_path}")
    print(f"Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        if add_personal_judgement_column(csv_file):
            results['success'] += 1
        else:
            results['error'] += 1
    
    return results

def main():
    """
    Main function to process all directories.
    """
    base_path = r"C:\Users\dtian\GitHub_Issues_Prioritisation"
    
    directories = [
        os.path.join(base_path, "longest_issues"),
        os.path.join(base_path, "shortest_issues"),
        os.path.join(base_path, "random_issues")
    ]
    
    total_results = {'success': 0, 'error': 0}
    
    print("🚀 Starting issue type labeling based on personal judgement mapping...")
    print("=" * 70)
    
    for directory in directories:
        results = process_directory(directory)
        total_results['success'] += results['success']
        total_results['error'] += results['error']
    
    print("\n" + "=" * 70)
    print(f"📊 Summary:")
    print(f"   ✅ Successfully processed: {total_results['success']} files")
    print(f"   ❌ Errors: {total_results['error']} files")
    print(f"   📁 Total files: {total_results['success'] + total_results['error']}")
    
    if total_results['error'] == 0:
        print("\n🎉 All files processed successfully!")
    else:
        print(f"\n⚠️  {total_results['error']} files encountered errors.")

if __name__ == "__main__":
    main()
