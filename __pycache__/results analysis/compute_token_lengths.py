import os
import tiktoken
from pathlib import Path

def count_tokens(text, encoding_name="cl100k_base"):
    """
    Count the number of tokens in a text string using tiktoken.
    cl100k_base is the encoding used by GPT-4 and GPT-3.5-turbo.
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Error encoding text: {e}")
        return 0

def process_txt_files(directory_path):
    """
    Process all .txt files in a directory and return token counts.
    """
    results = []
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Directory {directory_path} does not exist.")
        return results
    
    # Find all .txt files in the directory
    txt_files = list(directory.glob("*.txt"))
    
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as file:
                content = file.read()
                token_count = count_tokens(content)
                results.append({
                    'file_name': txt_file.name,
                    'file_path': str(txt_file),
                    'token_count': token_count,
                    'char_count': len(content)
                })
        except Exception as e:
            print(f"Error reading file {txt_file}: {e}")
            results.append({
                'file_name': txt_file.name,
                'file_path': str(txt_file),
                'token_count': 0,
                'char_count': 0,
                'error': str(e)
            })
    
    return results

def main():
    # Define the directories to process
    base_dir = r"c:\Users\dtian\GitHub_Issues_Prioritisation"
    directories = [
        os.path.join(base_dir, "random_issues")
    ]
    
    print("Computing token lengths for .txt files...\n")
    print("=" * 80)
    
    all_results = []
    
    for directory in directories:
        print(f"\nProcessing directory: {directory}")
        print("-" * 60)
        
        results = process_txt_files(directory)
        
        if not results:
            print("No .txt files found in this directory.")
            continue
        # Sort results by token count (ascending)
        results.sort(key=lambda x: x.get('token_count', 0), reverse=False)
        
        for result in results:
            if 'error' in result:
                print(f"ERROR - {result['file_name']}: {result['error']}")
            else:
                print(f"{result['file_name']}: {result['token_count']} tokens ({result['char_count']} characters)")
        
        all_results.extend(results)
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    valid_results = [r for r in all_results if 'error' not in r and r['token_count'] > 0]
    
    if valid_results:
        total_tokens = sum(r['token_count'] for r in valid_results)
        total_chars = sum(r['char_count'] for r in valid_results)
        avg_tokens = total_tokens / len(valid_results)
        max_tokens = max(r['token_count'] for r in valid_results)
        min_tokens = min(r['token_count'] for r in valid_results)
        
        print(f"Total files processed: {len(valid_results)}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Total characters: {total_chars:,}")
        print(f"Average tokens per file: {avg_tokens:.1f}")
        print(f"Maximum tokens in a single file: {max_tokens:,}")
        print(f"Minimum tokens in a single file: {min_tokens:,}")
        
        # Show top 5 shortest files
        print(f"\nTop 5 shortest files by token count:")
        top_5 = sorted(valid_results, key=lambda x: x['token_count'], reverse=False)[:5]
        for i, result in enumerate(top_5, 1):
            print(f"  {i}. {result['file_name']}: {result['token_count']:,} tokens")
    else:
        print("No valid files were processed.")

if __name__ == "__main__":
    main()
