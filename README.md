# 🤖 Clary Health Pattern Reasoner

A conversational AI system that analyzes health conversation patterns using advanced language models to identify temporal relationships, recurring issues, and behavioral patterns in user-Clary interactions.

## 📋 Assignment Objective

Build an intelligent pattern recognition system for health conversations that can:
- Analyze user-Clary dialogue timelines
- Identify meaningful patterns in health-related behaviors
- Provide temporal reasoning about symptom progression and intervention effectiveness
- Generate structured, evidence-based insights from conversation data
- Maintain strict privacy and safety standards (no medical diagnosis)

## 🛠 Tech Stack

- **Frontend**: Streamlit (conversational chat interface)
- **Backend**: Python 3.10+
- **AI/ML**: Groq Llama-3.1-8b-instant (primary), Llama-3.1-70b-versatile, Mixtral-8x7b, Gemma-7b
- **Data Processing**: Pandas, Pydantic
- **Environment**: python-dotenv
- **Deployment**: Streamlit Cloud

## 🚀 How to Run Locally

### Prerequisites
- Python 3.10 or higher
- Groq API key

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd clary-health-pattern-reasoner
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
    ```bash
    cp .env.example .env
    # Edit .env and add your Groq API key
    ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Access the app**
   - Open your browser to `http://localhost:8501`
   - Upload a dataset and start analyzing patterns!

## 🔐 Environment Variables

Create a `.env` file in the project root with:

```env
GROQ_API_KEY=your_groq_api_key_here
```

**Security Note**: Never commit your `.env` file or expose API keys in your code.

## 📊 Dataset Handling

### Supported Format
The application processes JSON datasets with the following structure:

```json
{
  "dataset_info": {},
  "users": [
    {
      "user_id": "USR001",
      "name": "Alice Johnson",
      "occupation": "Software Engineer",
      "onboarding_notes": "New hire from engineering team",
      "conversations": [
        {
          "id": "conv_001",
          "timestamp": "2024-01-01T10:00:00Z",
          "content": "User message here",
          "metadata": {
            "user_message": "I feel tired today",
            "clary_response": "I understand your fatigue",
            "severity": "medium",
            "tags": ["fatigue", "sleep"],
            "session_id": "session_alice_001"
          }
        }
      ]
    }
  ]
}
```

### Processing Pipeline
1. **Validation**: Ensures required fields exist
2. **Security**: Removes `hidden_patterns_reference` immediately upon loading
3. **Normalization**: Converts nested conversations to flat timeline events
4. **Enrichment**: Calculates week numbers and temporal relationships

## ⚠️ Important Security Note

**The `hidden_patterns_reference` field is automatically removed during dataset loading and never used anywhere in the application.** This field is considered sensitive training data that should not influence pattern analysis results.

## 🏗 Architecture

### 1. Data Loader (`src/data_loader.py`)
- **Function**: Loads and validates JSON datasets
- **Security**: Removes sensitive fields immediately
- **Validation**: Ensures data structure integrity
- **Output**: Structured Dataset and summary objects

### 2. Timeline Builder (`src/timeline_builder.py`)
- **Function**: Converts conversations to temporal event sequences
- **Features**: Week calculation, event summarization, pattern detection setup
- **Output**: TimelineEvent objects with rich metadata

### 3. LLM Reasoning Layer (`src/pattern_reasoner.py`)
- **Function**: Orchestrates AI analysis of user timelines
- **Prompt Engineering**: Structured reasoning prompts for pattern detection
- **Context Management**: User-wise timeline compression and session summaries
- **Output**: Raw LLM responses for validation

### 4. JSON Validator (`src/schema.py`)
- **Function**: Validates and parses LLM output against strict schemas
- **Schemas**: PatternsAnalysis, Pattern, EvidenceTimeline, RejectedHypothesis
- **Repair**: Attempts to extract valid JSON from malformed responses
- **Output**: Structured analysis results or error messages

### 5. Streamlit Chat UI (`app.py`)
- **Function**: Provides conversational interface for pattern analysis
- **Features**: Example buttons, progress streaming, result visualization
- **Timeline Display**: User and pattern evidence timelines as interactive tables
- **Export**: JSON download functionality

## 📝 Chunking and Context Management Strategy

### Current Approach (Small Dataset Optimized)
- **User-wise Processing**: Each user analyzed independently to maintain context
- **Full History**: Complete conversation timeline per user (dataset is small)
- **Compact Summaries**: Events condensed into structured TimelineEvent objects
- **Session Grouping**: Conversations grouped by session_id for pattern analysis

### Scalability Strategy (For Larger Datasets)
- **Sliding Window**: Recent N weeks + key historical events
- **Pattern-based Sampling**: Prioritize events relevant to detected patterns
- **Hierarchical Summarization**: Week → Month → Quarter summaries
- **Adaptive Chunking**: Context length based on model capabilities

### Context Optimization
- **Metadata Preservation**: Severity, tags, session info maintained
- **Temporal Relationships**: Week numbers and timestamps preserved
- **Evidence Linking**: Direct mapping between timeline events and pattern evidence

## 🤖 LLM Choice and Rationale

**Primary Model**: Groq Llama-3.1-8b-instant

### Why Llama-3.1-8b-instant?
- **Fast Inference**: Extremely quick response times (milliseconds)
- **Cost-Effective**: Free tier available, low-cost API usage
- **Structured Output**: Strong performance on JSON generation tasks
- **Temporal Reasoning**: Excellent at pattern analysis and timeline understanding

### Alternative Models Available
- **Llama-3.1-70b-versatile**: Maximum reasoning capability
- **Mixtral-8x7b-32768**: Balanced performance for complex tasks
- **Gemma-7b-it**: Lightweight option for basic analysis

### Selection Criteria
- ✅ **Inference Speed**: Critical for interactive pattern analysis
- ✅ **Cost Efficiency**: Enables broader accessibility and testing
- ✅ **JSON Generation**: Reliable schema-compliant structured output
- ✅ **Pattern Recognition**: Strong temporal and relational reasoning

## ⚠️ Failure Modes & Error Handling

### Dataset Issues
- **Invalid JSON**: Clear error messages with validation details
- **Missing Fields**: Specific validation errors for required fields
- **Empty Dataset**: Graceful handling with user feedback

### API Failures
- **Rate Limiting**: Automatic retry with exponential backoff
- **Network Issues**: Clear error messages and retry options
- **Invalid API Key**: Immediate feedback with setup instructions

### LLM Response Issues
- **Malformed JSON**: Attempt repair or extract valid portions
- **Schema Violations**: Detailed validation error messages
- **Empty Responses**: Fallback to mock mode with sample data

### UI/UX Issues
- **Large Datasets**: Progress indicators and chunked processing
- **Slow Responses**: Streaming progress updates
- **Browser Timeouts**: Optimized chunking and session management

## 🚀 Future Improvements

### Enhanced Analysis Features
- **Multi-user Pattern Detection**: Cross-user pattern identification
- **Intervention Effectiveness**: Quantify treatment response patterns
- **Risk Stratification**: Automated severity progression analysis
- **Predictive Insights**: Early warning for pattern development

### Technical Enhancements
- **Batch Processing**: Parallel analysis for multiple users
- **Caching Layer**: Store analysis results for faster re-analysis
- **Model Fine-tuning**: Custom models for health pattern recognition
- **Real-time Analysis**: Streaming conversation analysis

### UI/UX Improvements
- **Advanced Visualizations**: Timeline charts and pattern graphs
- **Interactive Filtering**: Filter patterns by type, severity, time range
- **Export Options**: PDF reports, CSV data, visualization exports
- **Collaboration Features**: Share analysis results with team members

### Scalability Features
- **Database Integration**: Persistent storage for large datasets
- **API Endpoints**: REST API for programmatic access
- **Queue System**: Background processing for large analysis jobs
- **Multi-tenant Support**: Organization-level data isolation

## ☁️ Deployment on Streamlit Cloud

### Prerequisites
- GitHub repository with the project
- Groq API key
- Streamlit Cloud account

### Deployment Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Connect to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select the main branch

3. **Configure Secrets**
    - In Streamlit Cloud dashboard, go to your app settings
    - Add secret: `GROQ_API_KEY` with your API key value
    - The app will automatically use this instead of `.env`

4. **Deploy**
   - Click "Deploy" in Streamlit Cloud
   - Wait for build completion
   - Access your live app URL

### Streamlit Cloud Configuration
```toml
# packages.txt (if needed)
# requirements.txt (already configured)

# secrets.toml (configured via UI)
GROQ_API_KEY = "your-key-here"
```

### Production Considerations
- **Resource Limits**: Monitor usage within Streamlit Cloud free tier
- **API Costs**: Track Groq API usage and costs
- **Data Privacy**: Ensure compliance with health data regulations
- **Performance**: Consider upgrading to paid tier for heavy usage

## 📄 License

This project is developed for educational and research purposes. Please ensure compliance with applicable data privacy regulations when handling health-related conversation data.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper testing
4. Submit a pull request with detailed description

---

**Built with ❤️ for advancing health conversation analysis through AI-powered pattern recognition.**