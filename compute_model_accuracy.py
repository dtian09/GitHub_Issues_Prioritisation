import pandas as pd
import os
import glob
from collections import defaultdict

def compute_model_accuracy():
    """
    Compute the accuracy of each model's type predictions compared to the first Personal Judgement Type.
    For each CSV file, use the first value in Personal Judgement Type column as ground truth.
    Accuracy = (Correct Predictions) / (Total Predictions) excluding CSV files with Unknown Type ground truth.
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
    files_skipped_unknown = 0
    
    print("Computing model accuracy for type predictions...")
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
                
                # Check if required columns exist
                if 'Model' not in df.columns or 'Predicted Type' not in df.columns or 'Personal Judgement Type' not in df.columns:
                    print(f"  Skipping {os.path.basename(csv_file)} - missing required columns")
                    continue
                
                # Get the first value from Personal Judgement Type column as ground truth
                if len(df) == 0:
                    continue
                    
                ground_truth_type = df['Personal Judgement Type'].iloc[0]
                
                total_files_processed += 1
                print(f"  Processing: {os.path.basename(csv_file)}")
                print(f"    Ground truth type: {ground_truth_type}")
                # Skip this CSV file if ground truth is Unknown Type
                if ground_truth_type == 'Unknown Type':
                    print(f"    Skipping - Ground truth is 'Unknown Type'")
                    files_skipped_unknown += 1
                    continue
                
                # Process each row (each model's prediction) against the same ground truth
                for _, row in df.iterrows():
                    model = row['Model']
                    predicted_type = row['Predicted Type']
                    
                    # Count total predictions
                    model_stats[model]['total'] += 1
                    
                    # Check if prediction matches the ground truth (first Personal Judgement Type)
                    if predicted_type == ground_truth_type:
                        model_stats[model]['correct'] += 1
                        
            except Exception as e:
                print(f"  Error processing {csv_file}: {str(e)}")
    
    # Calculate and display accuracy for each model
    print(f"\n{'='*60}")
    print("MODEL ACCURACY RESULTS")
    print(f"{'='*60}")
    print(f"{'Model':<30} {'Correct':<10} {'Total':<10} {'Accuracy':<12}")
    print(f"{'-'*62}")
    
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
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total CSV files processed: {total_files_processed}")
    print(f"Total CSV files skipped (Unknown Type): {files_skipped_unknown}")
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
    compute_model_accuracy()
