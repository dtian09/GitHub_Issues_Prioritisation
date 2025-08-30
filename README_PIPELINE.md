# GitHub Issues Processing Pipeline

This pipeline processes GitHub issues by reading issue IDs from a text file, retrieving their content from a MySQL database, and using AI models (specifically grok-4) to generate summaries, classify types, and assign priorities.

## Updated Pipeline Features

The `pipeline.py` has been updated to:

1. **Read issue IDs from a text file** (one ID per line)
2. **Retrieve issue contents from MySQL database** using the provided IDs
3. **Process issues using grok-4 model** for:
   - Content cleaning and normalization
   - 40-word summarization
   - Type classification (24 different types)
   - Priority labeling (10 priority levels)
4. **Update the MySQL database** with the generated summaries, types, and priorities

## Usage

```bash
python pipeline.py --input sample_issue_ids.txt --model grok-4 --batch-size 20 --max-workers 8 --sleep 0.25
```

### Arguments

- `--input`: Text file containing issue IDs (one per line) - default: "issue_ids.txt"
- `--model`: AI model to use - default: "grok-4"
- `--batch-size`: Number of issues to process in each database batch - default: 20
- `--max-workers`: Maximum number of worker threads - default: 8
- `--sleep`: Sleep time between API calls (seconds) - default: 0.25

### Environment Variables Required

Make sure you have these environment variables set in your `.env` file:

```env
# Database configuration
DB_HOST=localhost
DB_USER=david
DB_PASS=david
DB_NAME=github_issues_db

# API Keys (depending on model used)
XAI_API_KEY=your_xai_api_key_here  # For grok-4
OPENAI_API_KEY=your_openai_key     # For GPT models
ANTHROPIC_API_KEY=your_anthropic_key  # For Claude models
GOOGLE_API_KEY=your_google_key     # For Gemini models
GROQ_API_KEY=your_groq_key         # For Llama models
DEEPSEEK_API_KEY=your_deepseek_key # For DeepSeek models

# Optional
TEMPERATURE=0.2
```

### Issue ID File Format

Create a text file with one issue ID per line:

```
144259136
237734712
315565490
339392528
354702553
```

### Database Schema

The pipeline expects a MySQL table with this structure:

```sql
CREATE TABLE issue (
    issue_id BIGINT PRIMARY KEY,
    content TEXT DEFAULT NULL,
    summary TEXT DEFAULT NULL,
    type VARCHAR(128) DEFAULT NULL,
    priority VARCHAR(128) DEFAULT NULL
);
```

### Supported Models

- **grok-4** (default, recommended)
- gpt-4o, gpt-5
- claude-3-5-sonnet-latest
- gemini-2.0-flash
- llama-3.3-70b-versatile
- deepseek-chat

### Output

The pipeline will:
1. Read issue IDs from the specified file
2. Fetch corresponding issue content from the database
3. Process each issue to generate:
   - A 40-word summary
   - Type classification (Bug, Feature, Task, etc.)
   - Priority level (Blocker, Critical, Major, etc.)
4. Update the database with the new information

### Performance

- Uses multithreading for parallel processing
- Batched database updates for efficiency
- Rate limiting to respect API limits
- Progress bars for monitoring

### Error Handling

- Continues processing if individual issues fail
- Logs errors for debugging
- Graceful handling of missing data
- Database transaction safety
