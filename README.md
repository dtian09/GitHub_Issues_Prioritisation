# GitHub Issues Summarization, Categorization and Prioritisation

An AI-powered system for automatically analyzing, summarizing, classifying, and prioritizing GitHub issues using large language models (LLMs). This project helps development teams efficiently triage and manage large volumes of GitHub issues by providing intelligent categorization and priority assignment.

## 🚀 Features

- **Automated Issue Processing**: Bulk processing of GitHub issues from various sources
- **AI-Powered Analysis**: Uses multiple LLM providers (Grok-4, GPT-4, Claude, Gemini, etc.)
- **Smart Summarization**: Generates concise 40-word summaries of complex issues
- **Type Classification**: Categorizes issues into 24 different types (Bug, Feature, Task, etc.)
- **Priority Assignment**: Assigns priority levels from Blocker to Trivial
- **Database Integration**: MySQL database for storing and managing issue data
- **Flexible Pipeline**: Configurable processing with rate limiting and batch operations
- **Multi-Model Support**: Compare results across different AI models
- **Batch Processing**: Process multiple issues simultaneously across all models
- **Performance Analysis**: Built-in accuracy and quality evaluation tools
- **Comprehensive Validation**: Ground truth comparison and accuracy measurement
- **Quality Metrics**: Cosine similarity analysis for summary relevance assessment

## 📁 Project Structure

```
├── pipeline.py                                    # Main processing pipeline
├── summarize_label_types_priorities_single_issue.py  # Single issue processor
├── multi_models_issue_summarizer_single_issue.py     # Multi-model comparison
├── database/
│   ├── create_db.sql                             # Database schema
│   ├── get_github_issues.py                     # Issue data collection
│   ├── insert_data_to_db.py                     # Data insertion utilities
│   └── sql_select_and_save_issues_into_files.py # Data export utilities
├── longest_issues/                               # Sample long issues for testing
├── random_issues/                                # Sample random issues
├── shortest_issues/                              # Sample short issues
├── results analysis/                             # Model accuracy and analysis tools
│   ├── compute_model_accuracy.py                # Calculate model prediction accuracy
│   ├── compute_priority_accuracy.py             # Priority classification accuracy
│   ├── analyse_cosine_similarity.py             # Summary quality analysis
│   ├── batch_process_issues.py                  # Batch processing utility
│   └── compute_token_lengths.py                 # Token count analysis
└── sample_issue_ids.txt                         # Example issue ID file
```

## 🔧 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dtian09/GitHub_Issues_Prioritisation.git
   cd GitHub_Issues_Prioritisation
   ```

2. **Install dependencies:**
   ```bash
   pip install mysql-connector-python pandas tqdm python-dotenv openai anthropic groq xai-sdk
   ```

3. **Set up MySQL database:**
   ```bash
   mysql -u root -p < database/create_db.sql
   ```

4. **Configure environment variables:**
   Create a `.env` file with your API keys and database configuration:
   ```env
   # Database Configuration
   DB_HOST=localhost
   DB_USER=your_username
   DB_PASS=your_password
   DB_NAME=github_issues_db
   
   # API Keys (add as needed)
   XAI_API_KEY=your_xai_api_key         # For Grok-4
   OPENAI_API_KEY=your_openai_key       # For GPT models
   ANTHROPIC_API_KEY=your_anthropic_key # For Claude models
   GOOGLE_API_KEY=your_google_key       # For Gemini models
   GROQ_API_KEY=your_groq_key          # For Llama models
   DEEPSEEK_API_KEY=your_deepseek_key   # For DeepSeek models
   
   # Optional Settings
   TEMPERATURE=0.2
   ```

## 🎯 Quick Start

### 1. Process Issues from ID File

```bash
# Basic usage with default settings (Grok-4, 8 workers, batch size 20)
python pipeline.py --input sample_issue_ids.txt

# Conservative approach for rate limiting
python pipeline.py --input sample_issue_ids.txt --max-workers 1 --sleep 1.0

# Custom configuration
python pipeline.py --input my_issue_ids.txt --model grok-4 --batch-size 10 --max-workers 4 --sleep 0.5
```

### 2. Process Single Issue

```bash
# Analyze a single issue file
python summarize_label_types_priorities_single_issue.py --input issue_file.txt --model grok-4 --output results.csv

# Compare multiple models
python multi_models_issue_summarizer_single_issue.py --input issue_file.txt
```

### 3. Issue ID File Format

Create a text file with one issue ID per line:
```
144259136
237734712
315565490
```

## 🤖 Supported AI Models

| Provider | Models | Best For |
|----------|--------|----------|
| **XAI** | grok-4 | General purpose, good balance |
| **OpenAI** | gpt-4o, gpt-5 | High quality analysis |
| **Anthropic** | claude-3-5-sonnet-latest | Complex reasoning |
| **Google** | gemini-2.0-flash | Fast processing |
| **Groq** | llama-3.3-70b-versatile | Cost-effective |
| **DeepSeek** | deepseek-chat | Alternative option |

## 📊 Classification Categories

### Issue Types (24 categories)
- **Bug**: Error, crash, unexpected behavior
- **New Feature**: Add new functionality  
- **Story**: User requirements and acceptance criteria
- **Improvement**: Optimize, enhance, performance
- **Technical Task**: Backend, API, implementation
- **Epic**: Large initiatives and milestones
- **Task**: General work items
- **Sub-task**: Breakdown of larger tasks
- **Documentation**: Docs, manuals, guides
- **Test**: Unit tests, integration tests, QA
- **Support Request**: Help and assistance
- **Question**: How-to, clarification requests
- **Suggestion**: Recommendations and proposals
- **Build Failure**: CI/CD, compilation errors
- **Investigation**: Root cause analysis
- **Incident**: Outages, alerts, production issues
- And more...

### Priority Levels (10 levels)
- **Blocker**: Production down, cannot proceed
- **Critical**: Security holes, severe issues  
- **Major**: Important bugs, high impact
- **High**: Needs attention soon
- **Medium**: Normal priority, moderate impact
- **Minor**: Low impact, cosmetic issues
- **Low**: Backlog items, non-urgent
- **Trivial**: Very low priority
- **Lowest**: Icebox items
- **None/To be reviewed**: Untriaged

## ⚙️ Configuration Options

### Pipeline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | issue_ids.txt | Text file with issue IDs |
| `--model` | grok-4 | AI model to use |
| `--batch-size` | 20 | Database batch size |
| `--max-workers` | 8 | Parallel processing threads |
| `--sleep` | 0.25 | Delay between API calls (seconds) |

### Rate Limiting Recommendations

| Model | Max Workers | Sleep | Notes |
|-------|-------------|-------|-------|
| **Grok-4** | 1-2 | 1.0 | Conservative for rate limits |
| **GPT-4o** | 4-8 | 0.25 | Higher rate limits |
| **Claude** | 2-4 | 0.5 | Moderate limits |
| **Gemini** | 4-6 | 0.3 | Good throughput |

## 🗄️ Database Schema

```sql
CREATE TABLE issue (
    issue_id BIGINT PRIMARY KEY,
    content TEXT DEFAULT NULL,      -- Original issue text
    summary TEXT DEFAULT NULL,      -- AI-generated 40-word summary
    type VARCHAR(128) DEFAULT NULL, -- Classified issue type
    priority VARCHAR(128) DEFAULT NULL -- Assigned priority level
);
```

## 📈 Performance Considerations

### Processing Flow
1. **Issue Reading**: Load issue IDs from text file
2. **Database Retrieval**: Fetch issue content in bulk
3. **Parallel Processing**: Process multiple issues concurrently
4. **Sequential Steps per Issue**: 
   - Content cleaning and normalization
   - AI summarization (40 words)
   - Type classification (depends on summary)
   - Priority assignment (depends on summary)
5. **Batch Database Updates**: Efficient bulk updates

### Optimization Tips
- Start with `--max-workers 1` to avoid rate limits
- Increase workers gradually based on API response
- Use larger `--batch-size` for better database performance
- Monitor API usage and adjust `--sleep` accordingly

## 📊 Analysis and Evaluation

The repository includes comprehensive tools for evaluating model performance and analyzing results:

### **Model Accuracy Assessment**
- **Type Classification Accuracy**: Compare AI predictions against human judgments
- **Priority Assignment Accuracy**: Evaluate priority classification performance  
- **Cross-Model Comparison**: Analyze performance differences between AI models
- **Cosine Similarity Analysis**: Measure summary quality and relevance

### **Analysis Tools**
```bash
# Compute overall model accuracy
python "results analysis/compute_model_accuracy.py"

# Analyze priority classification accuracy
python "results analysis/compute_priority_accuracy.py"

# Evaluate summary quality via cosine similarity
python "results analysis/analyse_cosine_similarity.py"

# Batch process multiple issues across all models
python "results analysis/batch_process_issues.py"
```

### **Performance Metrics**
- **Accuracy Scores**: Percentage of correct classifications
- **Cosine Similarity**: Summary relevance (0.0-1.0 scale)
- **Token Analysis**: Content length and processing efficiency
- **Cross-Model Consensus**: Agreement between different AI models

## 🧪 Testing

The repository includes comprehensive sample data for testing and evaluation:
- `longest_issues/`: Complex, lengthy GitHub issues with detailed analysis
- `random_issues/` & `random_issues2/`: Diverse issue samples across different categories
- `shortest_issues/`: Brief, simple issues for quick testing
- `sample_issue_ids.txt`: Ready-to-use issue ID list
- **Prediction Results**: Pre-computed analysis from multiple AI models
- **Personal Judgments**: Human-labeled ground truth for accuracy evaluation

```bash
# Test with sample data
python pipeline.py --input sample_issue_ids.txt --max-workers 1 --sleep 1.0
```

### **Evaluation and Validation**

The repository includes extensive validation capabilities:

```bash
# Run comprehensive model accuracy analysis
python "results analysis/compute_model_accuracy.py"

# Evaluate priority classification performance  
python "results analysis/compute_priority_accuracy.py"

# Analyze summary quality and relevance
python "results analysis/analyse_cosine_similarity.py"

# Process all models on issue sets for comparison
python "results analysis/batch_process_issues.py"
```

**Validation Features:**
- **Ground Truth Comparison**: Human-labeled data for accuracy measurement
- **Cross-Model Analysis**: Performance comparison across different AI models
- **Quality Metrics**: Cosine similarity scores for summary relevance
- **Comprehensive Reporting**: Detailed accuracy and performance statistics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes and test thoroughly
4. Commit your changes: `git commit -am 'Add new feature'`
5. Push to the branch: `git push origin feature/new-feature`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [BigCode Stack GitHub Issues](https://huggingface.co/datasets/bigcode/the-stack-github-issues) - Dataset source
- [OpenAI API](https://platform.openai.com/docs/api-reference) - GPT models
- [Anthropic Claude](https://docs.anthropic.com/claude/reference/getting-started) - Claude models
- [XAI Grok](https://console.x.ai/) - Grok models

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Review the [Pipeline Documentation](README_PIPELINE.md)
- Check the database setup in `database/create_db.sql`

---

**Built with ❤️ for efficient GitHub issue management**
