import pandas as pd
import os
import glob
from collections import defaultdict
import re

def normalize_priority(priority_str):
    """
    Normalize priority strings by cleaning and standardizing them.
    """
    if pd.isna(priority_str) or priority_str == '' or priority_str.strip() == '':
        return []
    
    # Convert to string and clean
    priority_str = str(priority_str).strip()
    
    # Handle common variations
    priority_mappings = {
        'none/to be reviewed': 'None (or To be reviewed)',
        'none (or to be reviewed)': 'None (or To be reviewed)',
        'to be reviewed': 'None (or To be reviewed)',
        'none': 'None (or To be reviewed)',
        'blocker (or highest)': 'Blocker (or Highest)',
        'blocker': 'Blocker (or Highest)',
        'highest': 'Blocker (or Highest)',
        'critical': 'Critical',
        'major': 'Major',
        'high': 'High',
        'medium': 'Medium',
        'minor': 'Minor',
        'trivial': 'Trivial',
        'low': 'Low',
        'lowest': 'Lowest'
    }
    
    # Normalize the string
    normalized = priority_str.lower().strip()
    
    # Split by common separators and clean each part
    parts = re.split(r'[,;|&]|\sand\s', normalized)
    result_priorities = []
    
    for part in parts:
        part = part.strip()
        if part in priority_mappings:
            mapped = priority_mappings[part]
            if mapped not in result_priorities:
                result_priorities.append(mapped)
        elif part:  # Keep original if not in mappings but not empty
            # Capitalize first letter of each word
            formatted = ' '.join(word.capitalize() for word in part.split())
            if formatted not in result_priorities:
                result_priorities.append(formatted)
    
    return result_priorities

def check_priority_match(predicted, personal_judgement):
    """
    Check if predicted priority matches any of the personal judgement priorities.
    """
    predicted_normalized = normalize_priority(predicted)
    personal_normalized = normalize_priority(personal_judgement)
    
    if not predicted_normalized or not personal_normalized:
        # If either is empty/None, check if both are empty/None
        return len(predicted_normalized) == 0 and len(personal_normalized) == 0
    
    # Check if any predicted priority matches any personal judgement priority
    for pred in predicted_normalized:
        for pers in personal_normalized:
            if pred.lower() == pers.lower():
                return True
    
    return False

def compute_priority_accuracy():
    """
    Compute the accuracy of each model's priority predictions compared to Personal Judgement Priority.
    """
    
    # Define directories to process
    directories = [
        "longest_issues",
        "shortest_issues", 
        "random_issues"
    ]
    
    # Dictionary to store results for each model
    model_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    total_files_processed = 0
    files_skipped_no_priority = 0
    
    print("Computing model accuracy for priority predictions...")
    print("=" * 70)
    
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
                
                # Check if required columns exist
                if 'Model' not in df.columns or 'Predicted Priority' not in df.columns:
                    print(f"  Skipping {os.path.basename(csv_file)} - missing required columns")
                    continue
                
                # Check if Personal Judgement Priority column exists
                if 'Personal Judgement Priority' not in df.columns:
                    print(f"  Skipping {os.path.basename(csv_file)} - missing Personal Judgement Priority column")
                    files_skipped_no_priority += 1
                    continue
                
                # Get the first value from Personal Judgement Priority column as ground truth
                if len(df) == 0:
                    continue
                    
                ground_truth_priority = df['Personal Judgement Priority'].iloc[0]
                
                total_files_processed += 1
                print(f"  Processing: {os.path.basename(csv_file)}")
                print(f"    Ground truth priority: {ground_truth_priority}")
                
                # Process each row (each model's prediction) against the same ground truth
                for _, row in df.iterrows():
                    model = row['Model']
                    predicted_priority = row['Predicted Priority']
                    
                    # Count total predictions
                    model_stats[model]['total'] += 1
                    
                    # Check if prediction matches the ground truth priority
                    if check_priority_match(predicted_priority, ground_truth_priority):
                        model_stats[model]['correct'] += 1
                        print(f"      ✓ {model}: {predicted_priority} matches {ground_truth_priority}")
                    else:
                        print(f"      ✗ {model}: {predicted_priority} vs {ground_truth_priority}")
                        
            except Exception as e:
                print(f"  Error processing {csv_file}: {str(e)}")
    
    # Calculate and display accuracy for each model
    print(f"\n{'='*70}")
    print("MODEL PRIORITY ACCURACY RESULTS")
    print(f"{'='*70}")
    print(f"{'Model':<30} {'Correct':<10} {'Total':<10} {'Accuracy':<12}")
    print(f"{'-'*70}")
    
    # Sort models by accuracy (descending)
    model_accuracies = []
    
    for model, stats in model_stats.items():
        if stats['total'] > 0:
            accuracy = (stats['correct'] / stats['total']) * 100
            model_accuracies.append((model, accuracy, stats))
        else:
            accuracy = 0.0
            model_accuracies.append((model, accuracy, stats))
    
    # Sort by accuracy descending
    model_accuracies.sort(key=lambda x: x[1], reverse=True)
    
    for model, accuracy, stats in model_accuracies:
        print(f"{model:<30} {stats['correct']:<10} {stats['total']:<10} {accuracy:<12.2f}%")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total CSV files processed: {total_files_processed}")
    print(f"Total CSV files skipped (no priority column): {files_skipped_no_priority}")
    print(f"Total models evaluated: {len(model_stats)}")
    
    if model_accuracies:
        best_model = model_accuracies[0]
        worst_model = model_accuracies[-1]
        
        print(f"Best performing model: {best_model[0]} ({best_model[1]:.2f}%)")
        print(f"Worst performing model: {worst_model[0]} ({worst_model[1]:.2f}%)")
        
        # Calculate overall average accuracy
        total_correct = sum(stats['correct'] for _, _, stats in model_accuracies)
        total_predictions = sum(stats['total'] for _, _, stats in model_accuracies)
        
        if total_predictions > 0:
            overall_accuracy = (total_correct / total_predictions) * 100
            print(f"Overall average accuracy: {overall_accuracy:.2f}%")

if __name__ == "__main__":
    compute_priority_accuracy()
