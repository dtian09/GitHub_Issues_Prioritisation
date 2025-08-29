import pandas as pd
import os
import glob

def replace_column_headers_in_csv_files():
    """
    Replace 'Personal Judegment', 'personal judgement', or 'Personal Judgement' 
    with 'Personal Judgement Type' in all CSV files under specified directories.
    """
    
    # Define directories to process
    directories = [
        "longest_issues",
        "shortest_issues", 
        "random_issues",
        "random_issues2"  # Including this based on workspace structure
    ]
    
    # Column name variations to replace
    old_column_names = [
        "Personal Judegment",      # Typo version
        "personal judgement",      # Lowercase version
        "Personal Judgement",      # Current version
        "personal judgment",       # US spelling lowercase
        "Personal Judgment"        # US spelling
    ]
    
    new_column_name = "Personal Judgement Type"
    
    total_files_processed = 0
    total_files_updated = 0
    
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
                
                # Check if any of the old column names exist
                column_found = False
                for old_name in old_column_names:
                    if old_name in df.columns:
                        # Replace the column name
                        df = df.rename(columns={old_name: new_column_name})
                        column_found = True
                        print(f"  Updated '{old_name}' to '{new_column_name}' in {os.path.basename(csv_file)}")
                        break
                
                if column_found:
                    # Save the updated CSV file
                    df.to_csv(csv_file, index=False)
                    total_files_updated += 1
                else:
                    print(f"  No matching column found in {os.path.basename(csv_file)}")
                    # Print current column names for debugging
                    print(f"    Current columns: {list(df.columns)}")
                    
            except Exception as e:
                print(f"  Error processing {csv_file}: {str(e)}")
    
    print(f"\n--- Summary ---")
    print(f"Total files processed: {total_files_processed}")
    print(f"Total files updated: {total_files_updated}")

if __name__ == "__main__":
    print("Starting column header replacement...")
    replace_column_headers_in_csv_files()
    print("Column header replacement completed!")
