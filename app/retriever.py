import logging
from typing import List, Dict, Any, Tuple
import re
from dataclasses import dataclass

from app.embeddings import EmbeddingManager, VectorStoreManager
from app.document_processor import ProcessedDocument, DocumentChunk

logger = logging.getLogger(__name__)

@dataclass
class RelevantClause:
    text: str
    relevance_score: float
    document_section: str
    chunk_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'relevance_score': self.relevance_score,
            'document_section': self.document_section,
            'chunk_metadata': self.chunk_metadata
        }

class DocumentRetriever:
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.vector_store_manager = VectorStoreManager(embedding_manager)
        self.min_relevance_score = 0.3  # Minimum cosine similarity threshold
        
    def _extract_document_section(self, chunk: DocumentChunk) -> str:
        """Extract or infer document section from chunk text"""
        text = chunk.text[:200]  # First 200 chars for section identification
        
        # Common section patterns in insurance/legal documents
        section_patterns = [
            r'^(SECTION\s+\d+[:\.\s].*?)$',
            r'^(ARTICLE\s+\d+[:\.\s].*?)$',
            r'^(CLAUSE\s+\d+[:\.\s].*?)$',
            r'^(\d+\.\s+[A-Z][^.]*?)[\.\:]',
            r'^([A-Z][A-Z\s]{5,}?)[\:\.]',  # ALL CAPS headers
            r'^(COVERAGE\s+[A-Z].*?)$',
            r'^(DEFINITIONS?[:\.].*?)$',
            r'^(EXCLUSIONS?[:\.].*?)$',
            r'^(CONDITIONS?[:\.].*?)$',
            r'^(BENEFITS?[:\.].*?)$',
            r'^(POLICY\s+[A-Z].*?)$'
        ]
        
        for pattern in section_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                section = match.group(1).strip()
                return section[:100]  # Limit section name length
        
        # Fallback: use chunk metadata or generic section
        chunk_index = chunk.metadata.get('chunk_index', 0)
        doc_url = chunk.metadata.get('doc_url', 'document')
        
        # Extract filename from URL for section naming
        doc_name = doc_url.split('/')[-1] if '/' in doc_url else doc_url
        return f"{doc_name} - Section {chunk_index + 1}"
    
    def _enhance_clause_relevance(self, query: str, clause: RelevantClause) -> float:
        """Enhance relevance score based on keyword matching and context"""
        enhanced_score = clause.relevance_score
        
        # Extract important keywords from query
        query_words = set(query.lower().split())
        clause_words = set(clause.text.lower().split())
        
        # Boost score for exact keyword matches
        exact_matches = query_words.intersection(clause_words)
        if exact_matches:
            keyword_boost = min(len(exact_matches) * 0.05, 0.2)
            enhanced_score += keyword_boost
        
        # Insurance/legal domain-specific keyword boosting
        important_keywords = {
            'coverage', 'policy', 'claim', 'benefit', 'deductible', 'premium',
            'exclusion', 'condition', 'liability', 'contract', 'agreement',
            'terms', 'provisions', 'clause', 'section', 'article'
        }
        
        domain_matches = query_words.intersection(important_keywords)
        if domain_matches and any(word in clause.text.lower() for word in domain_matches):
            enhanced_score += 0.1
        
        # Penalize very short clauses (likely incomplete)
        if len(clause.text.split()) < 10:
            enhanced_score *= 0.8
        
        # Cap the maximum score
        return min(enhanced_score, 1.0)
    
    def _filter_and_rank_clauses(self, 
                                query: str, 
                                raw_results: List[Tuple[DocumentChunk, float]]) -> List[RelevantClause]:
        """Filter and rank clauses based on relevance and quality"""
        relevant_clauses = []
        
        for chunk, score in raw_results:
            if score < self.min_relevance_score:
                continue
                
            # Create relevant clause object
            clause = RelevantClause(
                text=chunk.text,
                relevance_score=score,
                document_section=self._extract_document_section(chunk),
                chunk_metadata=chunk.metadata
            )
            
            # Enhance relevance score
            clause.relevance_score = self._enhance_clause_relevance(query, clause)
            
            relevant_clauses.append(clause)
        
        # Sort by enhanced relevance score
        relevant_clauses.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Remove near-duplicate clauses
        filtered_clauses = self._remove_duplicate_clauses(relevant_clauses)
        
        return filtered_clauses
    
    def _remove_duplicate_clauses(self, clauses: List[RelevantClause]) -> List[RelevantClause]:
        """Remove clauses that are too similar to each other"""
        if len(clauses) <= 1:
            return clauses
            
        filtered = [clauses[0]]  # Always keep the highest-scoring clause
        
        for clause in clauses[1:]:
            is_duplicate = False
            
            for existing in filtered:
                # Check text similarity
                similarity = self._calculate_text_similarity(clause.text, existing.text)
                
                if similarity > 0.8:  # 80% similarity threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(clause)
                
            # Limit total number of clauses
            if len(filtered) >= 10:
                break
        
        return filtered
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity based on word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def build_vector_store(self, processed_docs: List[ProcessedDocument]) -> str:
        """Build vector store from processed documents"""
        try:
            logger.info(f"Building vector store from {len(processed_docs)} documents")
            
            # Collect all chunks from all documents
            all_chunks = []
            for doc in processed_docs:
                all_chunks.extend(doc.chunks)
                logger.info(f"Added {len(doc.chunks)} chunks from {doc.doc_url}")
            
            if not all_chunks:
                raise Exception("No chunks available for vector store creation")
            
            # Create vector store
            store_id = await self.vector_store_manager.create_store(all_chunks)
            
            logger.info(f"Vector store created successfully: {store_id}")
            return store_id
            
        except Exception as e:
            logger.error(f"Error building vector store: {str(e)}")
            raise
    
    async def retrieve_relevant_clauses(self, 
                                      query: str, 
                                      store_id: str, 
                                      top_k: int = 5) -> List[RelevantClause]:
        """Retrieve relevant clauses for a given query"""
        try:
            logger.info(f"Retrieving relevant clauses for query: {query[:100]}...")
            
            # Get vector store
            vector_store = self.vector_store_manager.get_store(store_id)
            if not vector_store:
                raise Exception(f"Vector store not found: {store_id}")
            
            # Search for relevant chunks
            raw_results = await vector_store.search(query, top_k=top_k * 2)  # Get more results for filtering
            
            if not raw_results:
                logger.warning("No relevant chunks found")
                return []
            
            # Filter and rank clauses
            relevant_clauses = self._filter_and_rank_clauses(query, raw_results)
            
            # Limit to requested number
            final_clauses = relevant_clauses[:top_k]
            
            logger.info(f"Retrieved {len(final_clauses)} relevant clauses")
            
            # Log relevance scores for debugging
            for i, clause in enumerate(final_clauses):
                logger.debug(f"Clause {i+1}: score={clause.relevance_score:.3f}, "
                           f"section='{clause.document_section}', "
                           f"text_preview='{clause.text[:50]}...'")
            
            return final_clauses
            
        except Exception as e:
            logger.error(f"Error retrieving relevant clauses: {str(e)}")
            raise
    
    def get_vector_store_stats(self) -> List[Dict[str, Any]]:
        """Get statistics about all vector stores"""
        return self.vector_store_manager.list_stores()
    
    async def close(self):
        """Clean up resources"""
        await self.vector_store_manager.close()