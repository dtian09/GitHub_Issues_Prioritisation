#!/usr/bin/env python3
"""
batch_process_issues.py

Process all .txt files in the shortest_issues folder using multi_models_issue_summarizer_single_issue.py
with all available LLMs.

Usage:
python batch_process_issues.py
"""

import os
import subprocess
import glob
from pathlib import Path

def main():
    # Configuration
    input_folder = r"C:\Users\dtian\GitHub_Issues_Prioritisation\random_issues\2"
    output_folder = r"C:\Users\dtian\GitHub_Issues_Prioritisation\random_issues\2\predictions"
    script_path = r"C:\Users\dtian\GitHub_Issues_Prioritisation\multi_models_issue_summarizer_single_issue.py"
    
    # All available models
    models = [
        "gpt-4o",
        "claude-3-5-sonnet-latest", 
        "gemini-2.0-flash",
        "grok-4",
        "llama-3.3-70b-versatile",
        "deepseek-chat"
    ]
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all .txt files in the input folder
    txt_files = glob.glob(os.path.join(input_folder, "*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in {input_folder}")
        return
    
    print(f"Found {len(txt_files)} .txt files to process")
    
    # Process each file
    success_count = 0
    error_count = 0
    
    for txt_file in txt_files:
        file_name = Path(txt_file).stem  # Get filename without extension
        output_file = os.path.join(output_folder, f"predictions_{file_name}.csv")
        
        print(f"\n{'='*60}")
        print(f"Processing: {file_name}")
        print(f"Input: {txt_file}")
        print(f"Output: {output_file}")
        
        # Build command
        cmd = [
            "python", script_path,
            "--input", txt_file,
            "--output", output_file,
            "--models"
        ] + models + [
            "--temperature", "0.2"
        ]
        
        try:
            # Run the command
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            print(f"✅ SUCCESS: {file_name}")
            if result.stdout:
                print(f"Output: {result.stdout}")
            success_count += 1
            
        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR processing {file_name}")
            print(f"Return code: {e.returncode}")
            if e.stdout:
                print(f"Stdout: {e.stdout}")
            if e.stderr:
                print(f"Stderr: {e.stderr}")
            error_count += 1
            
        except Exception as e:
            print(f"❌ UNEXPECTED ERROR processing {file_name}: {e}")
            error_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total files processed: {len(txt_files)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Output folder: {output_folder}")
    
    if success_count > 0:
        print(f"\n✅ Successfully processed {success_count} files!")
        print(f"📁 Check the predictions in: {output_folder}")
    
    if error_count > 0:
        print(f"\n⚠️  {error_count} files had errors - check the logs above")

if __name__ == "__main__":
    main()