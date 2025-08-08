import os
import logging
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
import asyncio
import aiohttp
import json
from sentence_transformers import SentenceTransformer
import pickle
import tempfile
from datetime import datetime
import torch
import gc

from app.config import settings
from app.document_processor import DocumentChunk

logger = logging.getLogger(__name__)

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L3-v2"):
        """
        Initialize the embedding manager with the smallest available model.
        """
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunks = []
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model with aggressive memory optimizations."""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            
            # Aggressive memory optimization settings
            torch.set_num_threads(1)  # Limit CPU threads
            torch.set_grad_enabled(False)  # Disable gradients globally
            
            # Force garbage collection
            gc.collect()
            
            # Use CPU only to avoid GPU memory issues
            device = torch.device("cpu")
            
            # Load model with minimal memory footprint
            self.model = SentenceTransformer(
                self.model_name,
                device=device,
                cache_folder=None,  # Disable caching
                trust_remote_code=False  # Don't load custom code
            )
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False
                param.data = param.data.to(device)
            
            # Clear any cached computations
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = 128  # Reduce max sequence length
                
            logger.info(f"Model loaded successfully on {device}")
            
            # Force garbage collection again
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Try with an even smaller fallback model
            try:
                logger.info("Trying fallback model: distilbert-base-nli-mean-tokens")
                self.model = SentenceTransformer('distilbert-base-nli-mean-tokens', device='cpu')
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False
                logger.info("Fallback model loaded successfully")
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {fallback_error}")
                raise Exception("Could not load any embedding model")
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts with minimal memory usage."""
        try:
            if not texts:
                return np.array([])
            
            # Process in very small batches to minimize memory usage
            batch_size = 4  # Very small batch size
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Truncate texts to reduce memory usage
                truncated_batch = [text[:500] for text in batch]  # Limit text length
                
                batch_embeddings = self.model.encode(
                    truncated_batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    batch_size=1  # Process one at a time
                )
                embeddings.append(batch_embeddings)
                
                # Force garbage collection after each batch
                gc.collect()
            
            return np.vstack(embeddings)
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def build_index(self, chunks: List[str]) -> None:
        """Build FAISS index from text chunks with memory optimization."""
        try:
            self.chunks = chunks
            if not chunks:
                logger.warning("No chunks provided for indexing")
                return
            
            # Limit number of chunks to reduce memory usage
            max_chunks = 100  # Limit to 100 chunks maximum
            if len(chunks) > max_chunks:
                logger.info(f"Limiting chunks from {len(chunks)} to {max_chunks}")
                chunks = chunks[:max_chunks]
                self.chunks = chunks
            
            # Create embeddings
            embeddings = self.create_embeddings(chunks)
            
            # Build FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            self.index.add(embeddings.astype('float32'))
            
            logger.info(f"FAISS index built with {len(chunks)} chunks")
            
            # Clear embeddings from memory
            del embeddings
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error building index: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks using the query."""
        try:
            if not self.index or not self.chunks:
                return []
            
            # Truncate query to reduce memory usage
            query = query[:200]  # Limit query length
            
            # Create query embedding
            query_embedding = self.create_embeddings([query])
            
            # Search
            scores, indices = self.index.search(
                query_embedding.astype('float32'), 
                min(top_k, len(self.chunks))
            )
            
            # Return results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks):
                    results.append({
                        'chunk': self.chunks[idx],
                        'score': float(score),
                        'index': int(idx)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    def get_chunk_by_index(self, index: int) -> str:
        """Get chunk by index."""
        if 0 <= index < len(self.chunks):
            return self.chunks[index]
        return ""

class VectorStore:
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.index = None
        self.chunks = []
        self.metadata = []
        self.store_id = None
        self.created_at = None
    
    async def build_index(self, chunks: List[DocumentChunk]) -> str:
        """Build FAISS index from document chunks"""
        try:
            logger.info(f"Building FAISS index from {len(chunks)} chunks")
            
            if not chunks:
                raise Exception("No chunks provided for indexing")
            
            self.chunks = chunks
            self.metadata = [chunk.metadata for chunk in chunks]
            self.created_at = datetime.now()
            self.store_id = f"store_{int(self.created_at.timestamp())}"
            
            # Extract texts for embedding
            texts = [chunk.text for chunk in chunks]
            
            # Generate embeddings
            embeddings = await self.embedding_manager.generate_embeddings(texts)
            
            if len(embeddings) == 0:
                raise Exception("No embeddings generated")
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add embeddings to index
            self.index.add(embeddings)
            
            logger.info(f"FAISS index built successfully. Index size: {self.index.ntotal}")
            return self.store_id
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {str(e)}")
            raise
    
    async def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Search for most relevant chunks"""
        try:
            if self.index is None or len(self.chunks) == 0:
                logger.warning("No index available for search")
                return []
            
            # Generate query embedding
            query_embedding = await self.embedding_manager.generate_query_embedding(query)
            
            if query_embedding is None:
                logger.warning("Could not generate query embedding")
                return []
            
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_embedding)
            
            # Search
            k = min(top_k, self.index.ntotal)
            scores, indices = self.index.search(query_embedding, k)
            
            # Prepare results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= 0 and idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    results.append((chunk, float(score)))
            
            # Sort by score (descending)
            results.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Search completed. Found {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            'store_id': self.store_id,
            'total_chunks': len(self.chunks),
            'index_size': self.index.ntotal if self.index else 0,
            'embedding_dim': self.embedding_manager.embedding_dim,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class VectorStoreManager:
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.stores = {}  # store_id -> VectorStore
        self.max_stores = 10  # Limit memory usage
    
    async def create_store(self, chunks: List[DocumentChunk]) -> str:
        """Create a new vector store"""
        try:
            # Clean up old stores if we have too many
            if len(self.stores) >= self.max_stores:
                oldest_store_id = min(self.stores.keys())
                del self.stores[oldest_store_id]
                logger.info(f"Removed old vector store: {oldest_store_id}")
            
            # Create new store
            store = VectorStore(self.embedding_manager)
            store_id = await store.build_index(chunks)
            
            self.stores[store_id] = store
            logger.info(f"Created vector store: {store_id}")
            
            return store_id
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def get_store(self, store_id: str) -> VectorStore:
        """Get vector store by ID"""
        return self.stores.get(store_id)
    
    def list_stores(self) -> List[Dict[str, Any]]:
        """List all available stores"""
        return [store.get_stats() for store in self.stores.values()]
    
    async def close(self):
        """Clean up resources"""
        if self.embedding_manager.session and not self.embedding_manager.session.closed:
            await self.embedding_manager.session.close()