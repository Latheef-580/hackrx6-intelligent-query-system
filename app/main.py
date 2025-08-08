import logging
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn

from app.config import settings
from app.document_processor import DocumentProcessor
from app.retriever import DocumentRetriever
from app.response_builder import ResponseBuilder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HackRx 6.0 Intelligent Query-Retrieval System",
    description="LLM-Powered system for processing documents and answering queries",
    version="1.0.0"
)

# Security
security = HTTPBearer()

# Initialize components
document_processor = DocumentProcessor()
retriever = DocumentRetriever()
response_builder = ResponseBuilder()

# Pydantic models
class HackRxRequest(BaseModel):
    documents: str  # Single document URL
    questions: List[str]

class HackRxResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[str]
    reasoning: str

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != settings.TEAM_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return credentials.credentials

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Initializing HackRx 6.0 Intelligent Query-Retrieval System...")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "HackRx 6.0 Intelligent Query-Retrieval System",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "HackRx 6.0 Intelligent Query-Retrieval System",
        "version": "1.0.0"
    }

@app.post("/hackrx/run", response_model=Dict[str, List[str]])
async def process_documents(
    request: HackRxRequest,
    token: str = Depends(verify_token)
):
    """
    Process documents and answer questions.
    This is the main endpoint for the HackRx 6.0 challenge.
    """
    try:
        # Validate input
        if not request.documents or not request.documents.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document URL is required"
            )
        
        if not request.questions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one question is required"
            )
        
        logger.info(f"Processing 1 document with {len(request.questions)} questions")
        
        # Process document
        logger.info(f"Downloading and processing document: {request.documents}")
        chunks = await document_processor.process_document(request.documents)
        
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to extract text from document"
            )
        
        logger.info(f"Extracted {len(chunks)} text chunks from document")
        
        # Index documents for retrieval
        logger.info("Indexing documents for semantic search")
        retriever.index_documents(chunks)
        
        # Process each question
        answers = []
        for i, question in enumerate(request.questions, 1):
            logger.info(f"Processing question {i}/{len(request.questions)}: {question[:100]}...")
            
            try:
                # Get relevant context
                context = retriever.get_context_for_query(question, top_k=3)
                
                # Generate answer
                answer = response_builder.generate_answer(question, context)
                answers.append(answer)
                
                logger.info(f"Generated answer for question {i}")
                
            except Exception as e:
                logger.error(f"Error processing question {i}: {e}")
                answers.append(f"Error processing question: {str(e)}")
        
        logger.info(f"Successfully processed all {len(request.questions)} questions")
        return {"answers": answers}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/test/single-question")
async def test_single_question(
    request: HackRxRequest,
    token: str = Depends(verify_token)
):
    """
    Test endpoint for processing a single question.
    """
    try:
        if not request.questions or len(request.questions) != 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Exactly one question is required for this endpoint"
            )
        
        question = request.questions[0]
        logger.info(f"Testing single question: {question[:100]}...")
        
        # Process document
        chunks = await document_processor.process_document(request.documents)
        retriever.index_documents(chunks)
        
        # Get context and generate answer
        context = retriever.get_context_for_query(question, top_k=3)
        answer = response_builder.generate_answer(question, context)
        
        return {
            "question": question,
            "answer": answer,
            "context_length": len(context),
            "chunks_processed": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"Error in test_single_question: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )