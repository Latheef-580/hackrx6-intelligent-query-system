#!/usr/bin/env python3
"""
Startup script for HackRx 6.0 Intelligent Query-Retrieval System
This script starts the FastAPI application with proper configuration.
"""

import os
import sys
import uvicorn
from pathlib import Path

def setup_environment():
    """Setup environment variables if not already set"""
    if not os.getenv("ENVIRONMENT"):
        os.environ["ENVIRONMENT"] = "development"
    
    if not os.getenv("LOG_LEVEL"):
        os.environ["LOG_LEVEL"] = "INFO"
    
    # Set default values for development
    if os.getenv("ENVIRONMENT") == "development":
        if not os.getenv("USE_OPENAI_EMBEDDINGS"):
            os.environ["USE_OPENAI_EMBEDDINGS"] = "false"
        if not os.getenv("REQUIRE_AUTH"):
            os.environ["REQUIRE_AUTH"] = "false"

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import fastapi
        import uvicorn
        import sentence_transformers
        import faiss
        import PyPDF2
        import docx
        print("✅ All required dependencies are available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def main():
    """Main startup function"""
    print("🚀 Starting HackRx 6.0 Intelligent Query-Retrieval System")
    print("=" * 60)
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Get configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("ENVIRONMENT") == "development"
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    print(f"📍 Server will run on: http://{host}:{port}")
    print(f"🔧 Environment: {os.getenv('ENVIRONMENT', 'development')}")
    print(f"📝 Log level: {log_level}")
    print(f"🔄 Auto-reload: {reload}")
    
    if os.getenv("OPENAI_API_KEY"):
        print("🔑 OpenAI API key: Configured")
    else:
        print("🔑 OpenAI API key: Not configured (using local embeddings)")
    
    print("\n" + "=" * 60)
    print("🎯 Starting server...")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 