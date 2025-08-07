from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import time
from contextlib import asynccontextmanager

from app.document_processor import DocumentProcessor
from app.embeddings import EmbeddingManager
from app.retriever import DocumentRetriever
from app.response_builder import ResponseBuilder
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for managers
document_processor = None
embedding_manager = None
document_retriever = None
response_builder = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global document_processor, embedding_manager, document_retriever, response_builder
    
    logger.info("Initializing HackRx 6.0 Intelligent Query-Retrieval System...")
    
    # Initialize components
    document_processor = DocumentProcessor()
    embedding_manager = EmbeddingManager()
    document_retriever = DocumentRetriever(embedding_manager)
    response_builder = ResponseBuilder()
    
    logger.info("System initialized successfully!")
    yield
    
    # Shutdown
    logger.info("Shutting down system...")

app = FastAPI(
    title="HackRx 6.0 - Intelligent Query-Retrieval System",
    description="LLM-powered document analysis system for insurance/legal/HR/compliance documents",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HackRxRequest(BaseModel):
    documents: str  # Single document URL
    questions: List[str]  # List of questions to answer

class ClauseInfo(BaseModel):
    text: str
    relevance_score: float
    document_section: str

class HackRxResponse(BaseModel):
    question: str
    answer: str
    clauses_used: List[str]
    decision_rationale: str

def verify_auth_token(request: Request):
    """Verify authorization token (simulated for hackathon)"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization token")
    
    token = auth_header.split(" ")[1]
    # In real implementation, verify token against team database
    if len(token) < 10:  # Basic validation
        raise HTTPException(status_code=401, detail="Invalid team token")
    
    return token

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "HackRx 6.0 - Intelligent Query-Retrieval System",
        "status": "active",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        return {
            "status": "healthy",
            "components": {
                "document_processor": document_processor is not None,
                "embedding_manager": embedding_manager is not None,
                "document_retriever": document_retriever is not None,
                "response_builder": response_builder is not None
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="System health check failed")

@app.post("/hackrx/run")
async def process_documents(
    request: HackRxRequest,
    auth_token: str = Depends(verify_auth_token)
):
    """
    Main HackRx 6.0 endpoint for document processing and question answering
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing request with 1 document and {len(request.questions)} questions")
        
        if not request.documents or not request.documents.strip():
            raise HTTPException(status_code=400, detail="No document URL provided")
        
        if not request.questions:
            raise HTTPException(status_code=400, detail="No questions provided")
        
        # Step 1: Process documents
        logger.info("Step 1: Processing documents...")
        processed_docs = []
        
        try:
            doc_content = await document_processor.process_document(request.documents)
            processed_docs.append(doc_content)
            logger.info(f"Successfully processed document: {request.documents[:50]}...")
        except Exception as e:
            logger.error(f"Failed to process document {request.documents}: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to process document: {str(e)}")
        
        # Step 2: Create embeddings and build vector store
        logger.info("Step 2: Creating embeddings and building vector store...")
        vector_store_id = await document_retriever.build_vector_store(processed_docs)
        
        # Step 3: Process questions
        logger.info("Step 3: Processing questions...")
        answers = []
        
        for question in request.questions:
            try:
                # Retrieve relevant clauses
                relevant_clauses = await document_retriever.retrieve_relevant_clauses(
                    question, vector_store_id, top_k=5
                )
                
                # Generate response
                response = await response_builder.generate_response(
                    question, relevant_clauses
                )
                
                answers.append(response.answer)
                logger.info(f"Successfully processed question: {question[:50]}...")
                
            except Exception as e:
                logger.error(f"Failed to process question '{question}': {str(e)}")
                # Return error response for this question
                answers.append("Unable to process this question due to an error.")
        
        processing_time = time.time() - start_time
        logger.info(f"Request processed successfully in {processing_time:.2f} seconds")
        
        # Return in HackRx 6.0 format
        return {"answers": answers}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /hackrx/run: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/test/single-question")
async def test_single_question(
    document_url: str,
    question: str,
    auth_token: str = Depends(verify_auth_token)
):
    """Test endpoint for single question processing"""
    try:
        request = HackRxRequest(documents=[document_url], questions=[question])
        response = await process_documents(request, auth_token)
        return response[0] if response else None
    except Exception as e:
        logger.error(f"Test endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)