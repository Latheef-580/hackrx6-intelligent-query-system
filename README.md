# HackRx 6.0 - Intelligent Query-Retrieval System

A complete LLM-powered document analysis system for insurance, legal, HR, and compliance documents. This system processes large documents and provides intelligent, explainable answers to natural language queries.

## 🚀 Features

- **Multi-format Document Processing**: PDF, DOCX, and Email support
- **Semantic Search**: FAISS-based vector search with embeddings
- **Intelligent Retrieval**: Context-aware clause matching and retrieval
- **Explainable AI**: Clear decision rationale and clause traceability
- **High Performance**: Optimized for low latency (<2s per query)
- **Token Efficient**: RAG-based approach to minimize LLM costs
- **Production Ready**: FastAPI backend with proper error handling

## 🏗️ System Architecture

```
Input Documents (PDF/DOCX/Email)
           ↓
    Document Processor
           ↓
    Text Extraction & Chunking
           ↓
    Embedding Generation (FAISS)
           ↓
    Vector Store Index
           ↓
    Query Processing
           ↓
    Clause Retrieval & Matching
           ↓
    LLM Response Generation
           ↓
    Structured JSON Output
```

## 📋 Requirements

- Python 3.8+
- OpenAI API Key (optional, falls back to local embeddings)
- 4GB+ RAM (for embedding models)
- Internet connection (for document downloads)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd hackrx6
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   echo "ENVIRONMENT=development" >> .env
   ```

## 🚀 Quick Start

1. **Start the server**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Test the API**
   ```bash
   curl -X GET http://localhost:8000/health
   ```

3. **Process documents**
   ```bash
   curl -X POST http://localhost:8000/hackrx/run \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_team_token" \
     -d @sample_payload.json
   ```

## 📚 API Documentation

### Base URL
```
http://localhost:8000/api/v1
```

### Authentication
```
Authorization: Bearer <team_token>
```

### Endpoints

#### 1. Health Check
```http
GET /health
```

#### 2. Main Processing Endpoint
```http
POST /hackrx/run
```

**Request Body:**
```json
{
    "documents": ["https://example.com/document.pdf"],
    "questions": [
        "What is the grace period for premium payment?",
        "Does this policy cover knee surgery?"
    ]
}
```

**Response:**
```json
{
    "answers": [
        "A grace period of thirty days is provided for premium payment...",
        "Yes, the policy covers knee surgery under the following conditions..."
    ]
}
```

#### 3. Test Endpoint
```http
POST /test/single-question
```

## 🔧 Configuration

Environment variables in `app/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key for embeddings/LLM |
| `USE_OPENAI_EMBEDDINGS` | true | Use OpenAI embeddings (fallback to local) |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Local embedding model |
| `LLM_MODEL` | gpt-3.5-turbo | OpenAI model for responses |
| `CHUNK_SIZE` | 300 | Words per document chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `MIN_RELEVANCE_SCORE` | 0.3 | Minimum similarity threshold |

## 🏭 Production Deployment

### Render Deployment
1. Connect your repository to Render
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables in Render dashboard

### Azure Deployment
1. Create Azure App Service
2. Deploy using Azure CLI or GitHub Actions
3. Set environment variables in App Service Configuration

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🧪 Testing

### Unit Tests
```bash
pytest tests/ -v
```

### Integration Tests
```bash
python -m pytest tests/test_integration.py -v
```

### Load Testing
```bash
# Using Apache Bench
ab -n 100 -c 10 -H "Authorization: Bearer your_token" \
   -T application/json -p sample_payload.json \
   http://localhost:8000/hackrx/run
```

## 📊 Performance Metrics

- **Document Processing**: ~2-5 seconds per document
- **Query Response**: <2 seconds per question
- **Memory Usage**: ~500MB-1GB (depending on document size)
- **Concurrent Requests**: Up to 10 simultaneous requests

## 🔍 Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce `CHUNK_SIZE` in config
   - Use smaller embedding model
   - Increase system RAM

2. **Slow Performance**
   - Enable GPU acceleration for embeddings
   - Use OpenAI embeddings instead of local
   - Optimize document chunking

3. **API Errors**
   - Check OpenAI API key validity
   - Verify document URLs are accessible
   - Check network connectivity

### Logs
```bash
# View application logs
tail -f logs/app.log

# Debug mode
ENVIRONMENT=development uvicorn app.main:app --log-level debug
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## 📄 License

This project is developed for HackRx 6.0 (Bajaj Allianz Hackathon).

## 🏆 HackRx 6.0 Evaluation Criteria

This solution addresses all evaluation parameters:

- ✅ **Accuracy**: Precise query understanding and clause matching
- ✅ **Token Efficiency**: Optimized LLM usage with RAG approach
- ✅ **Latency**: Fast response times with FAISS vector search
- ✅ **Reusability**: Modular, extensible code architecture
- ✅ **Explainability**: Clear decision reasoning and clause traceability

## 📞 Support

For HackRx 6.0 related queries, contact the hackathon organizers.

---

**Built with ❤️ for HackRx 6.0 - Bajaj Allianz Hackathon**
