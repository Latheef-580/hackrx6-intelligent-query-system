import logging
from typing import List, Dict, Any
from app.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)

class DocumentRetriever:
    def __init__(self):
        """Initialize the document retriever with embedding manager."""
        self.embedding_manager = EmbeddingManager()
        self.is_indexed = False
    
    def index_documents(self, chunks: List[str]) -> None:
        """Index document chunks for retrieval."""
        try:
            logger.info(f"Indexing {len(chunks)} document chunks")
            self.embedding_manager.build_index(chunks)
            self.is_indexed = True
            logger.info("Document indexing completed")
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            raise
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a given query."""
        try:
            if not self.is_indexed:
                logger.warning("Documents not indexed yet")
                return []
            
            logger.info(f"Retrieving relevant chunks for query: {query[:100]}...")
            results = self.embedding_manager.search(query, top_k)
            
            logger.info(f"Retrieved {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
    
    def get_context_for_query(self, query: str, top_k: int = 3) -> str:
        """Get context string from relevant chunks for a query."""
        try:
            chunks = self.retrieve_relevant_chunks(query, top_k)
            if not chunks:
                return ""
            
            # Combine chunks with their relevance scores
            context_parts = []
            for i, chunk_data in enumerate(chunks, 1):
                chunk = chunk_data['chunk']
                score = chunk_data['score']
                context_parts.append(f"Relevant Section {i} (Relevance: {score:.3f}):\n{chunk}\n")
            
            context = "\n".join(context_parts)
            logger.info(f"Generated context with {len(chunks)} chunks")
            return context
            
        except Exception as e:
            logger.error(f"Error generating context: {e}")
            return ""