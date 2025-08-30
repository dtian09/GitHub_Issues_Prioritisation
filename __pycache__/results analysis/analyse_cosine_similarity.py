#!/usr/bin/env python3
"""
analyze_cosine_similarity.py

Analyzes cosine similarity scores from CSV files in /shortest_issues directory.
Calculates average scores per model and displays results in a table format.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def get_cosine_scores_from_csv(csv_path):
    """Extract cosine similarity scores from a CSV file."""
    try:
        df = pd.read_csv(csv_path)
        # Look for the cosine similarity column
        cosine_col = None
        for col in df.columns:
            if 'cosine' in col.lower() and 'similarity' in col.lower():
                cosine_col = col
                break
        
        if cosine_col is None:
            print(f"No cosine similarity column found in {csv_path}")
            return {}
        
        # Extract scores by model
        scores = {}
        for _, row in df.iterrows():
            model = row.get('Model', '')
            score = row.get(cosine_col, np.nan)
            if model and not pd.isna(score):
                scores[model] = float(score)
        
        return scores
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return {}

def analyze_cosine_similarity():
    """Main function to analyze cosine similarity scores."""
    base_dir = Path(r"C:\Users\dtian\GitHub_Issues_Prioritisation")
    
    # Define directories to search
    search_dirs = [
        base_dir / "random_issues"
    ]
    
    # Collect all CSV files
    csv_files = []
    for search_dir in search_dirs:
        if search_dir.exists():
            csv_files.extend(list(search_dir.glob("*.csv")))
            print(f"Found {len(list(search_dir.glob('*.csv')))} CSV files in {search_dir}")
        else:
            print(f"Directory not found: {search_dir}")
    
    if not csv_files:
        print("No CSV files found in the specified directories.")
        return
    
    # Collect scores by model
    model_scores = defaultdict(list)
    
    print(f"\nProcessing {len(csv_files)} CSV files...")
    for csv_file in csv_files:
        print(f"Processing: {csv_file.name}")
        scores = get_cosine_scores_from_csv(csv_file)
        
        for model, score in scores.items():
            model_scores[model].append(score)
    
    # Calculate statistics for each model
    results = []
    for model, scores in model_scores.items():
        avg_score = np.mean(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        std_score = np.std(scores)
        
        results.append({
            'Model': model,
            'Scores': scores,
            'Score_Count': len(scores),
            'Average_Score': round(avg_score, 4),
            'Min_Score': round(min_score, 4),
            'Max_Score': round(max_score, 4),
            'Std_Dev': round(std_score, 4)
        })
    
    # Sort by average score (descending)
    results.sort(key=lambda x: x['Average_Score'], reverse=True)
    
    # Display results in table format
    print("\n" + "="*120)
    print("COSINE SIMILARITY ANALYSIS RESULTS")
    print("="*120)
    
    # Header
    print(f"{'Model':<25} {'Count':<8} {'Average':<10} {'Min':<8} {'Max':<8} {'Std Dev':<10} {'All Scores'}")
    print("-" * 120)
    
    # Data rows
    for result in results:
        model = result['Model']
        count = result['Score_Count']
        avg = result['Average_Score']
        min_val = result['Min_Score']
        max_val = result['Max_Score']
        std_val = result['Std_Dev']
        scores_str = ', '.join([f"{s:.3f}" for s in result['Scores']])
        
        print(f"{model:<25} {count:<8} {avg:<10} {min_val:<8} {max_val:<8} {std_val:<10} {scores_str}")
    
    # NEW SECTION: Print detailed statistics for each model
    print("\n" + "="*80)
    print("DETAILED STATISTICS BY MODEL")
    print("="*80)
    
    for result in results:
        model = result['Model']
        scores = result['Scores']
        count = result['Score_Count']
        avg = result['Average_Score']
        min_val = result['Min_Score']
        max_val = result['Max_Score']
        std_val = result['Std_Dev']
        
        print(f"\n{model}:")
        print(f"  Count: {count}")
        print(f"  Average: {avg:.4f}")
        print(f"  Minimum: {min_val:.4f}")
        print(f"  Maximum: {max_val:.4f}")
        print(f"  Std Dev: {std_val:.4f}")
        print(f"  Range: {max_val - min_val:.4f}")
        print(f"  All Scores: {scores}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    all_scores = [score for scores_list in model_scores.values() for score in scores_list]
    if all_scores:
        print(f"Total CSV files processed: {len(csv_files)}")
        print(f"Total scores collected: {len(all_scores)}")
        print(f"Overall average cosine similarity: {np.mean(all_scores):.4f}")
        print(f"Overall standard deviation: {np.std(all_scores):.4f}")
        print(f"Overall min score: {np.min(all_scores):.4f}")
        print(f"Overall max score: {np.max(all_scores):.4f}")
    
    # Model performance ranking
    print("\n" + "="*60)
    print("MODEL RANKING BY AVERAGE COSINE SIMILARITY")
    print("="*60)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['Model']:<25} {result['Average_Score']:.4f}")
    
    # Additional analysis: Most consistent models (lowest std dev)
    print("\n" + "="*60)
    print("MODEL RANKING BY CONSISTENCY (LOWEST STD DEV)")
    print("="*60)
    consistency_ranking = sorted(results, key=lambda x: x['Std_Dev'])
    for i, result in enumerate(consistency_ranking, 1):
        print(f"{i}. {result['Model']:<25} {result['Std_Dev']:.4f}")
    
    # Save to CSV for further analysis
    output_path = base_dir / "cosine_similarity_analysis.csv"
    summary_df = pd.DataFrame([
        {
            'Model': result['Model'],
            'Score_Count': result['Score_Count'],
            'Average_Score': result['Average_Score'],
            'Min_Score': result['Min_Score'],
            'Max_Score': result['Max_Score'],
            'Std_Dev': result['Std_Dev'],
            'All_Scores': ', '.join([str(s) for s in result['Scores']])
        }
        for result in results
    ])
    
    summary_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")

if __name__ == "__main__":
    analyze_cosine_similarity()