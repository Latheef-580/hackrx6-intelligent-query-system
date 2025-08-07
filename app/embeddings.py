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

from app.config import settings
from app.document_processor import DocumentChunk

logger = logging.getLogger(__name__)

class EmbeddingManager:
    def __init__(self):
        self.model = None
        self.embedding_dim = settings.EMBEDDING_DIM
        self.model_name = settings.EMBEDDING_MODEL
        self.openai_api_key = settings.OPENAI_API_KEY
        self.use_openai = settings.USE_OPENAI_EMBEDDINGS
        self.session = None
        
        # Initialize embedding model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model"""
        try:
            if self.use_openai and self.openai_api_key:
                logger.info("Using OpenAI embeddings")
                self.embedding_dim = 1536  # OpenAI ada-002 dimension
            else:
                logger.info(f"Loading sentence transformer model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                # Get actual embedding dimension from the model
                test_embedding = self.model.encode(["test"])
                self.embedding_dim = test_embedding.shape[1]
                logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            # Fallback to a lightweight model
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                test_embedding = self.model.encode(["test"])
                self.embedding_dim = test_embedding.shape[1]
                logger.info(f"Fallback model loaded. Embedding dimension: {self.embedding_dim}")
            except Exception as fallback_error:
                logger.error(f"Fallback model failed: {str(fallback_error)}")
                raise Exception("Could not initialize any embedding model")
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=60)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def _get_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from OpenAI API"""
        try:
            session = await self._get_session()
            
            headers = {
                'Authorization': f'Bearer {self.openai_api_key}',
                'Content-Type': 'application/json'
            }
            
            # Process in batches to avoid token limits
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                payload = {
                    'input': batch_texts,
                    'model': 'text-embedding-ada-002'
                }
                
                async with session.post(
                    'https://api.openai.com/v1/embeddings',
                    headers=headers,
                    json=payload
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"OpenAI API error {response.status}: {error_text}")
                    
                    result = await response.json()
                    batch_embeddings = [item['embedding'] for item in result['data']]
                    all_embeddings.extend(batch_embeddings)
                
                # Small delay to respect rate limits
                await asyncio.sleep(0.1)
            
            return np.array(all_embeddings, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error getting OpenAI embeddings: {str(e)}")
            raise
    
    def _get_local_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using local model"""
        try:
            if self.model is None:
                raise Exception("Local embedding model not initialized")
            
            # Encode texts in batches for memory efficiency
            batch_size = 32
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch_texts,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                embeddings.append(batch_embeddings)
            
            return np.vstack(embeddings).astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error getting local embeddings: {str(e)}")
            raise
    
    async def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            
            if not texts:
                return np.array([]).reshape(0, self.embedding_dim)
            
            # Filter out empty texts
            non_empty_texts = [text.strip() for text in texts if text.strip()]
            if not non_empty_texts:
                logger.warning("All texts are empty after filtering")
                return np.array([]).reshape(0, self.embedding_dim)
            
            if self.use_openai and self.openai_api_key:
                embeddings = await self._get_openai_embeddings(non_empty_texts)
            else:
                embeddings = self._get_local_embeddings(non_empty_texts)
            
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    async def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        try:
            embeddings = await self.generate_embeddings([query])
            return embeddings[0] if len(embeddings) > 0 else np.zeros(self.embedding_dim)
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise

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