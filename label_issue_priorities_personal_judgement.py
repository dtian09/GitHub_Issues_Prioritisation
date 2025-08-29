import pandas as pd
import os
import glob
import re

def analyze_priority_from_summaries(summaries):
    """
    Analyze all summaries in a CSV file and determine the appropriate priority level.
    Returns the most appropriate priority based on content analysis.
    """
    
    # Combine all summaries for analysis
    combined_text = " ".join(summaries).lower()
    
    # Priority keywords and patterns (in order of severity)
    priority_patterns = {
        "Blocker (or Highest)": [
            r"system down", r"production down", r"complete failure", r"total outage",
            r"blocks everything", r"prevents deployment", r"security breach", 
            r"data loss", r"critical vulnerability", r"system crash", r"service unavailable"
        ],
        "Critical": [
            r"critical", r"urgent", r"emergency", r"severe", r"major outage",
            r"affecting all users", r"business critical", r"revenue impact",
            r"security issue", r"data corruption", r"performance degradation"
        ],
        "Major": [
            r"major", r"significant", r"important", r"affects many users",
            r"core functionality", r"main feature", r"widespread impact",
            r"breaking change", r"regression", r"build failure"
        ],
        "High": [
            r"high priority", r"affects users", r"feature broken", r"error",
            r"exception", r"failure", r"not working", r"incorrect behavior",
            r"bug", r"issue", r"problem"
        ],
        "Medium": [
            r"medium", r"moderate", r"enhancement", r"improvement", r"optimization",
            r"refactor", r"update", r"upgrade", r"new feature", r"feature request"
        ],
        "Minor": [
            r"minor", r"small", r"cosmetic", r"ui", r"formatting", r"typo",
            r"documentation", r"comment", r"style", r"cleanup"
        ],
        "Trivial": [
            r"trivial", r"negligible", r"whitespace", r"spelling", r"grammar",
            r"code style", r"formatting only", r"comment fix"
        ],
        "Low": [
            r"low priority", r"nice to have", r"future consideration",
            r"suggestion", r"idea", r"wish", r"question"
        ],
        "Lowest": [
            r"lowest", r"minimal", r"discussion", r"research needed",
            r"investigation", r"exploration"
        ]
    }
    
    # Special case patterns for None/To be reviewed
    none_patterns = [
        r"closed", r"duplicate", r"wontfix", r"invalid", r"obsolete",
        r"already fixed", r"not reproducible", r"working as intended"
    ]
    
    # Count matches for each priority level
    priority_scores = {}
    
    # Check for None/To be reviewed cases first
    none_score = 0
    for pattern in none_patterns:
        none_score += len(re.findall(pattern, combined_text))
    
    if none_score > 0:
        return "None (or To be reviewed)"
    
    # Score each priority level
    for priority, patterns in priority_patterns.items():
        score = 0
        for pattern in patterns:
            matches = len(re.findall(pattern, combined_text))
            score += matches
        priority_scores[priority] = score
    
    # Additional context-based scoring
    # Security-related issues get higher priority
    if any(word in combined_text for word in ["security", "vulnerability", "exploit", "breach"]):
        priority_scores["Critical"] += 3
    
    # Performance issues
    if any(word in combined_text for word in ["slow", "timeout", "performance", "memory leak"]):
        priority_scores["Major"] += 2
    
    # Build/deployment issues
    if any(word in combined_text for word in ["build fail", "deployment", "ci/cd", "pipeline"]):
        priority_scores["Major"] += 2
    
    # Documentation issues typically lower priority
    if any(word in combined_text for word in ["documentation", "readme", "tutorial", "guide"]):
        priority_scores["Minor"] += 1
    
    # Test-related issues
    if any(word in combined_text for word in ["test fail", "test case", "unit test", "integration test"]):
        priority_scores["Medium"] += 1
    
    # Find the priority with the highest score
    if not priority_scores or max(priority_scores.values()) == 0:
        return "Medium"  # Default priority if no patterns match
    
    max_priority = max(priority_scores, key=priority_scores.get)
    max_score = priority_scores[max_priority]
    
    # If multiple priorities have the same high score, prefer higher severity
    priority_order = [
        "Blocker (or Highest)", "Critical", "Major", "High", 
        "Medium", "Minor", "Trivial", "Low", "Lowest"
    ]
    
    for priority in priority_order:
        if priority in priority_scores and priority_scores[priority] == max_score:
            return priority
    
    return max_priority

def label_issue_priorities():
    """
    Label priorities for issues in CSV files based on summary analysis.
    """
    
    # Define directories to process
    directories = [
        "longest_issues",
        "shortest_issues", 
        "random_issues"
    ]
    
    total_files_processed = 0
    total_files_updated = 0
    
    print("Labeling issue priorities based on summary analysis...")
    print("=" * 60)
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Directory '{directory}' does not exist. Skipping...")
            continue
            
        print(f"\nProcessing directory: {directory}")
        
        # Find all CSV files in the directory
        csv_files = glob.glob(os.path.join(directory, "*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in '{directory}'")
            continue
            
        for csv_file in csv_files:
            try:
                # Read the CSV file
                df = pd.read_csv(csv_file)
                total_files_processed += 1
                
                # Check if required columns exist
                if 'Summary_40w' not in df.columns:
                    print(f"  Skipping {os.path.basename(csv_file)} - missing Summary_40w column")
                    continue
                
                print(f"  Processing: {os.path.basename(csv_file)}")
                
                # Get all summaries for analysis
                summaries = df['Summary_40w'].fillna('').tolist()
                
                # Analyze and determine priority
                priority = analyze_priority_from_summaries(summaries)
                print(f"    Assigned Priority: {priority}")
                
                # Add or update the Personal Judgement Priority column
                df['Personal Judgement Priority'] = priority
                
                # Save the updated CSV file
                df.to_csv(csv_file, index=False)
                total_files_updated += 1
                
            except Exception as e:
                print(f"  Error processing {csv_file}: {str(e)}")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {total_files_processed}")
    print(f"Total files updated: {total_files_updated}")

if __name__ == "__main__":
    label_issue_priorities()
