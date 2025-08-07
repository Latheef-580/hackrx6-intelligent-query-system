# 🚀 HackRx 6.0 - Intelligent Query-Retrieval System

A complete LLM-powered document analysis system for insurance, legal, HR, and compliance documents. Built for the HackRx 6.0 (Bajaj Allianz) Hackathon.

## 🎯 Features

- **Document Processing**: PDF, DOCX, and Email support
- **Embedding Generation**: Sentence transformers for semantic search
- **Vector Search**: FAISS for efficient similarity search
- **RAG Implementation**: Retrieval-Augmented Generation for accurate answers
- **Structured Output**: JSON responses in HackRx 6.0 format
- **Authentication**: Bearer token support
- **Health Monitoring**: System status checks
- **API Documentation**: Interactive Swagger UI

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   Embedding     │    │   Vector        │
│   Processor     │───▶│   Manager       │───▶│   Store (FAISS) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Text          │    │   Semantic      │    │   Response      │
│   Extraction    │    │   Search        │    │   Builder       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/hackrx6-intelligent-query-system.git
   cd hackrx6-intelligent-query-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the server**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

5. **Access the API**
   - Health Check: http://localhost:8000/health
   - API Documentation: http://localhost:8000/docs
   - Main Endpoint: http://localhost:8000/hackrx/run

## 📡 API Endpoints

### Health Check
```bash
GET /health
```

### Main Endpoint (HackRx 6.0)
```bash
POST /hackrx/run
Authorization: Bearer YOUR_TEAM_TOKEN
Content-Type: application/json

{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the grace period for premium payment?",
    "What is the waiting period for pre-existing diseases?"
  ]
}
```

### Response Format
```json
{
  "answers": [
    "Based on the policy, there is a grace period of 15 days...",
    "The waiting period for pre-existing diseases is 48 months..."
  ]
}
```

## 🚀 Deployment

### Deploy to Render (Recommended)

1. **Fork this repository** to your GitHub account
2. **Go to [Render.com](https://render.com)** and sign up
3. **Click "New +" → "Web Service"**
4. **Connect your GitHub repository**
5. **Configure the deployment**:
   - **Name**: `hackrx6-intelligent-query-system`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path**: `/health`
6. **Click "Create Web Service"**

### Your Webhook URL
After deployment, your webhook URL will be:
```
https://hackrx6-intelligent-query-system.onrender.com/hackrx/run
```

## 🧪 Testing

### Test Locally
```bash
python demo_output.py
```

### Test Deployed System
```bash
python test_deployment.py
```

## 📁 Project Structure

```
hackrx6/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── document_processor.py # Document processing
│   ├── embeddings.py        # Embedding generation
│   ├── retriever.py         # Vector search
│   └── response_builder.py  # Response generation
├── tests/
│   └── test_main.py         # Unit tests
├── requirements.txt         # Python dependencies
├── render.yaml             # Render deployment config
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose setup
├── sample_payload.json     # Sample API request
├── test_system.py          # System test script
└── README.md               # This file
```

## 🔧 Configuration

Environment variables can be set in `.env` file:

```env
ENVIRONMENT=production
OPENAI_API_KEY=your_openai_key_here
USE_OPENAI_EMBEDDINGS=false
LOG_LEVEL=INFO
REQUIRE_AUTH=true
```

## 🏆 HackRx 6.0 Submission

### Webhook URL
```
https://your-app-name.onrender.com/hackrx/run
```

### Submission Notes
```
Complete LLM-powered Intelligent Query-Retrieval System with:
- RAG (Retrieval-Augmented Generation)
- Embedding-based semantic search
- Multi-format document processing (PDF, DOCX, Email)
- Explainable AI with source traceability
- Structured JSON responses
- Authentication and health monitoring
```

## 🤝 Contributing

This project was built for the HackRx 6.0 hackathon. Feel free to fork and improve!

## 📄 License

This project is created for educational and hackathon purposes.

---

**Built with ❤️ for HackRx 6.0 (Bajaj Allianz Hackathon)**
