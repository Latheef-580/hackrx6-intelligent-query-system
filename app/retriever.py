import logging
import re
from typing import List, Dict, Any
from app.embeddings import EmbeddingManager
from app.config import settings

logger = logging.getLogger(__name__)

class SimpleTextRetriever:
    """Simple text-based retriever for memory-constrained environments."""
    
    def __init__(self):
        self.chunks = []
        self.is_indexed = False
    
    def index_documents(self, chunks: List[str]) -> None:
        """Index document chunks using simple text storage."""
        try:
            logger.info(f"Indexing {len(chunks)} document chunks using simple text matching")
            # Limit chunks for memory
            max_chunks = settings.MAX_CHUNKS
            if len(chunks) > max_chunks:
                logger.info(f"Limiting chunks from {len(chunks)} to {max_chunks}")
                chunks = chunks[:max_chunks]
            
            self.chunks = chunks
            self.is_indexed = True
            logger.info("Simple text indexing completed")
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using simple keyword matching."""
        try:
            if not self.is_indexed or not self.chunks:
                return []
            
            # Simple keyword-based scoring
            query_words = set(re.findall(r'\b\w+\b', query.lower()))
            results = []
            
            for i, chunk in enumerate(self.chunks):
                chunk_words = set(re.findall(r'\b\w+\b', chunk.lower()))
                
                # Calculate simple overlap score
                if chunk_words:
                    overlap = len(query_words.intersection(chunk_words))
                    total_unique = len(query_words.union(chunk_words))
                    score = overlap / total_unique if total_unique > 0 else 0
                else:
                    score = 0
                
                if score > 0:  # Only include chunks with some relevance
                    results.append({
                        'chunk': chunk,
                        'score': score,
                        'index': i
                    })
            
            # Sort by score and return top_k
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in simple search: {e}")
            return []
    
    def get_context_for_query(self, query: str, top_k: int = 3) -> str:
        """Get context string from relevant chunks for a query."""
        try:
            chunks = self.search(query, top_k)
            if not chunks:
                return ""
            
            # Combine chunks with their relevance scores
            context_parts = []
            for i, chunk_data in enumerate(chunks, 1):
                chunk = chunk_data['chunk']
                score = chunk_data['score']
                context_parts.append(f"Relevant Section {i} (Relevance: {score:.3f}):\n{chunk}\n")
            
            context = "\n".join(context_parts)
            logger.info(f"Generated context with {len(chunks)} chunks using simple matching")
            return context
            
        except Exception as e:
            logger.error(f"Error generating context: {e}")
            return ""

class DocumentRetriever:
    def __init__(self):
        """Initialize the document retriever with fallback support."""
        self.use_fallback = settings.USE_FALLBACK_MODE
        self.embedding_manager = None
        self.simple_retriever = None
        self.is_indexed = False
        
        if not self.use_fallback:
            try:
                self.embedding_manager = EmbeddingManager()
                logger.info("Using embedding-based retrieval")
            except Exception as e:
                logger.warning(f"Failed to initialize embedding manager: {e}")
                self.use_fallback = True
        
        if self.use_fallback:
            self.simple_retriever = SimpleTextRetriever()
            logger.info("Using simple text-based retrieval")
    
    def index_documents(self, chunks: List[str]) -> None:
        """Index document chunks for retrieval."""
        try:
            logger.info(f"Indexing {len(chunks)} document chunks")
            
            if self.use_fallback or self.embedding_manager is None:
                self.simple_retriever.index_documents(chunks)
            else:
                self.embedding_manager.build_index(chunks)
            
            self.is_indexed = True
            logger.info("Document indexing completed")
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            # Fallback to simple retriever if embedding fails
            logger.info("Falling back to simple text retrieval")
            self.use_fallback = True
            self.simple_retriever = SimpleTextRetriever()
            self.simple_retriever.index_documents(chunks)
            self.is_indexed = True
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a given query."""
        try:
            if not self.is_indexed:
                logger.warning("Documents not indexed yet")
                return []
            
            logger.info(f"Retrieving relevant chunks for query: {query[:100]}...")
            
            if self.use_fallback or self.embedding_manager is None:
                results = self.simple_retriever.search(query, top_k)
            else:
                results = self.embedding_manager.search(query, top_k)
            
            logger.info(f"Retrieved {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
    
    def get_context_for_query(self, query: str, top_k: int = 3) -> str:
        """Get context string from relevant chunks for a query."""
        try:
            if self.use_fallback or self.embedding_manager is None:
                return self.simple_retriever.get_context_for_query(query, top_k)
            else:
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