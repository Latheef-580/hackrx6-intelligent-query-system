# HackRx 6.0 Deployment Guide

This guide provides step-by-step instructions for deploying the HackRx 6.0 Intelligent Query-Retrieval System.

## 🚀 Quick Start (Local Development)

### Prerequisites
- Python 3.8+
- pip
- 4GB+ RAM

### Steps
1. **Clone and setup**
   ```bash
   git clone <repository-url>
   cd hackrx6
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Start the server**
   ```bash
   python start.py
   # OR
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Test the system**
   ```bash
   python test_system.py
   ```

## 🐳 Docker Deployment

### Local Docker
```bash
# Build and run
docker build -t hackrx6 .
docker run -p 8000:8000 hackrx6

# Using docker-compose
docker-compose up --build
```

### Production Docker
```bash
# Build production image
docker build -t hackrx6:prod .

# Run with environment variables
docker run -d \
  -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e OPENAI_API_KEY=your_key \
  -e REQUIRE_AUTH=true \
  --name hackrx6-app \
  hackrx6:prod
```

## ☁️ Cloud Deployment

### Render Deployment

1. **Connect Repository**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New Web Service"
   - Connect your GitHub repository

2. **Configure Service**
   - **Name**: `hackrx6-app`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

3. **Environment Variables**
   ```
   ENVIRONMENT=production
   OPENAI_API_KEY=your_openai_api_key
   USE_OPENAI_EMBEDDINGS=true
   REQUIRE_AUTH=true
   ```

4. **Deploy**
   - Click "Create Web Service"
   - Wait for build to complete

### Azure App Service

1. **Create App Service**
   ```bash
   az group create --name hackrx6-rg --location eastus
   az appservice plan create --name hackrx6-plan --resource-group hackrx6-rg --sku B1
   az webapp create --name hackrx6-app --resource-group hackrx6-rg --plan hackrx6-plan --runtime "PYTHON|3.9"
   ```

2. **Configure Environment**
   ```bash
   az webapp config appsettings set --name hackrx6-app --resource-group hackrx6-rg --settings \
     ENVIRONMENT=production \
     OPENAI_API_KEY=your_key \
     USE_OPENAI_EMBEDDINGS=true
   ```

3. **Deploy**
   ```bash
   az webapp deployment source config-zip --resource-group hackrx6-rg --name hackrx6-app --src hackrx6.zip
   ```

### AWS Elastic Beanstalk

1. **Create Application**
   ```bash
   eb init hackrx6 --platform python-3.9 --region us-east-1
   eb create hackrx6-env
   ```

2. **Configure Environment**
   ```bash
   eb setenv ENVIRONMENT=production OPENAI_API_KEY=your_key
   ```

3. **Deploy**
   ```bash
   eb deploy
   ```

## 🔧 Production Configuration

### Environment Variables
```bash
# Required for production
ENVIRONMENT=production
OPENAI_API_KEY=sk-your-openai-key
USE_OPENAI_EMBEDDINGS=true
REQUIRE_AUTH=true

# Optional optimizations
WORKERS=2
MAX_CONCURRENT_REQUESTS=20
REQUEST_TIMEOUT=180
LOG_LEVEL=WARNING
```

### Performance Tuning
```bash
# For high traffic
WORKERS=4
MAX_CONCURRENT_REQUESTS=50
CHUNK_SIZE=500
CHUNK_OVERLAP=100

# For memory optimization
MAX_VECTOR_STORES=5
MAX_DOCUMENT_SIZE_MB=25
```

### Security
```bash
# Enable authentication
REQUIRE_AUTH=true

# Use HTTPS in production
# Configure reverse proxy (nginx/Apache)
# Set up rate limiting
# Enable CORS properly
```

## 📊 Monitoring & Logging

### Health Checks
```bash
# Check system health
curl http://your-domain/health

# Expected response:
{
  "status": "healthy",
  "components": {
    "document_processor": true,
    "embedding_manager": true,
    "document_retriever": true,
    "response_builder": true
  }
}
```

### Logging
```bash
# View logs
docker logs hackrx6-app

# Or if running directly
tail -f logs/app.log
```

### Performance Monitoring
```bash
# Test response time
time curl -X POST http://your-domain/hackrx/run \
  -H "Authorization: Bearer your_token" \
  -H "Content-Type: application/json" \
  -d @sample_payload.json
```

## 🔍 Troubleshooting

### Common Issues

1. **Memory Errors**
   ```bash
   # Reduce memory usage
   export CHUNK_SIZE=200
   export MAX_VECTOR_STORES=3
   ```

2. **Slow Performance**
   ```bash
   # Use OpenAI embeddings
   export USE_OPENAI_EMBEDDINGS=true
   export OPENAI_API_KEY=your_key
   ```

3. **Authentication Errors**
   ```bash
   # Check token format
   curl -H "Authorization: Bearer your_token" http://your-domain/health
   ```

4. **Document Processing Errors**
   ```bash
   # Check document URL accessibility
   curl -I "your-document-url"
   
   # Verify file format support
   # Supported: PDF, DOCX, Email
   ```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export ENVIRONMENT=development

# Start with debug
uvicorn app.main:app --log-level debug --reload
```

## 🧪 Testing Production

### Load Testing
```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Test with 100 requests, 10 concurrent
ab -n 100 -c 10 -H "Authorization: Bearer your_token" \
   -T application/json -p sample_payload.json \
   http://your-domain/hackrx/run
```

### Integration Testing
```bash
# Run the test script against production
python test_system.py
# Update BASE_URL in test_system.py to your production URL
```

## 📈 Scaling

### Horizontal Scaling
- Deploy multiple instances behind a load balancer
- Use Redis for session management (if needed)
- Configure auto-scaling based on CPU/memory usage

### Vertical Scaling
- Increase worker processes: `WORKERS=4`
- Increase memory allocation
- Use more powerful CPU instances

### Database Scaling (Future)
- Add PostgreSQL for persistent storage
- Use Redis for caching
- Implement connection pooling

## 🔐 Security Checklist

- [ ] HTTPS enabled
- [ ] Authentication configured
- [ ] CORS properly configured
- [ ] Rate limiting implemented
- [ ] Input validation enabled
- [ ] Error messages don't leak sensitive info
- [ ] API keys secured
- [ ] Logs don't contain sensitive data

## 📞 Support

For deployment issues:
1. Check the logs: `docker logs container-name`
2. Verify environment variables
3. Test with the provided test script
4. Check the health endpoint
5. Review the troubleshooting section above

---

**Happy Deploying! 🚀** 