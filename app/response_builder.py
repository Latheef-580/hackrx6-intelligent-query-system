import logging
import asyncio
import aiohttp
import json
from typing import List, Dict, Any
import re

from app.config import settings
from app.retriever import RelevantClause
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class HackRxResponse(BaseModel):
    question: str
    answer: str
    clauses_used: List[str]
    decision_rationale: str

class ResponseBuilder:
    def __init__(self):
        self.openai_api_key = settings.OPENAI_API_KEY
        self.model_name = settings.LLM_MODEL
        self.max_tokens = settings.MAX_RESPONSE_TOKENS
        self.session = None
        
        # Prompt templates
        self.system_prompt = """You are an expert AI assistant specializing in insurance, legal, HR, and compliance document analysis. Your task is to provide accurate, helpful answers to questions based on the provided document clauses.

Guidelines:
1. Answer questions directly and accurately based ONLY on the provided clauses
2. If the clauses don't contain enough information, clearly state this limitation
3. Use professional, clear language appropriate for business contexts
4. Focus on practical implications and actionable information
5. Cite specific clauses when making claims
6. If multiple clauses are relevant, synthesize them coherently
7. Be concise but thorough - aim for 2-4 sentences unless more detail is needed"""

        self.user_prompt_template = """Based on the following document clauses, please answer this question:

QUESTION: {question}

RELEVANT CLAUSES:
{clauses}

Please provide a clear, accurate answer based on these clauses. If the clauses don't fully address the question, mention what information is missing."""

    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=60)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    def _format_clauses_for_llm(self, clauses: List[RelevantClause]) -> str:
        """Format clauses for LLM input"""
        formatted_clauses = []
        
        for i, clause in enumerate(clauses, 1):
            clause_text = f"CLAUSE {i} (from {clause.document_section}):\n{clause.text}\n"
            formatted_clauses.append(clause_text)
        
        return "\n".join(formatted_clauses)

    def _extract_clause_excerpts(self, clauses: List[RelevantClause], max_length: int = 150) -> List[str]:
        """Extract short excerpts from clauses for the response"""
        excerpts = []
        
        for clause in clauses:
            # Take first sentence or up to max_length characters
            text = clause.text.strip()
            
            # Try to get first complete sentence
            sentences = re.split(r'[.!?]+', text)
            if sentences and len(sentences[0]) <= max_length:
                excerpt = sentences[0].strip() + "."
            else:
                # Truncate at word boundary
                if len(text) <= max_length:
                    excerpt = text
                else:
                    excerpt = text[:max_length]
                    last_space = excerpt.rfind(' ')
                    if last_space > max_length * 0.8:  # Don't cut too short
                        excerpt = excerpt[:last_space] + "..."
                    else:
                        excerpt = excerpt + "..."
            
            excerpts.append(excerpt)
        
        return excerpts

    async def _call_openai_api(self, messages: List[Dict[str, str]]) -> str:
        """Call OpenAI API for answer generation"""
        try:
            session = await self._get_session()
            
            headers = {
                'Authorization': f'Bearer {self.openai_api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': self.model_name,
                'messages': messages,
                'max_tokens': self.max_tokens,
                'temperature': 0.1,  # Low temperature for factual responses
                'top_p': 0.9,
                'frequency_penalty': 0.0,
                'presence_penalty': 0.0
            }
            
            async with session.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=payload
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error {response.status}: {error_text}")
                
                result = await response.json()
                
                if 'choices' not in result or not result['choices']:
                    raise Exception("No response from OpenAI API")
                
                answer = result['choices'][0]['message']['content'].strip()
                return answer
                
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise

    def _generate_fallback_answer(self, question: str, clauses: List[RelevantClause]) -> str:
        """Generate a simple fallback answer without LLM"""
        if not clauses:
            return "I couldn't find relevant information in the provided documents to answer this question."
        
        # Simple rule-based response generation
        clause_texts = [clause.text for clause in clauses[:2]]  # Use top 2 clauses
        
        combined_text = " ".join(clause_texts)
        
        # Basic answer construction
        if len(combined_text) > 300:
            answer = f"Based on the relevant clauses found: {combined_text[:300]}..."
        else:
            answer = f"Based on the relevant clauses: {combined_text}"
        
        return answer

    def _generate_decision_rationale(self, 
                                   question: str, 
                                   clauses: List[RelevantClause], 
                                   answer: str) -> str:
        """Generate explanation of how the answer was derived"""
        if not clauses:
            return "No relevant clauses were found in the documents to answer this question."
        
        rationale_parts = []
        
        # Explain clause selection
        if len(clauses) == 1:
            rationale_parts.append(f"I identified 1 relevant clause from {clauses[0].document_section}")
        else:
            sections = list(set(clause.document_section for clause in clauses))
            if len(sections) == 1:
                rationale_parts.append(f"I identified {len(clauses)} relevant clauses from {sections[0]}")
            else:
                rationale_parts.append(f"I identified {len(clauses)} relevant clauses from {len(sections)} different sections")
        
        # Explain relevance
        avg_score = sum(clause.relevance_score for clause in clauses) / len(clauses)
        if avg_score > 0.8:
            confidence = "high"
        elif avg_score > 0.5:
            confidence = "moderate"
        else:
            confidence = "low"
        
        rationale_parts.append(f"with {confidence} relevance to your question")
        
        # Explain answer construction
        if "based on" in answer.lower() or "according to" in answer.lower():
            rationale_parts.append("The answer synthesizes information from these clauses to directly address your question")
        elif len(clauses) > 1:
            rationale_parts.append("The answer combines information from multiple clauses to provide a comprehensive response")
        else:
            rationale_parts.append("The answer is derived directly from the most relevant clause")
        
        return ". ".join(rationale_parts) + "."

    async def generate_response(self, 
                              question: str, 
                              relevant_clauses: List[RelevantClause]) -> HackRxResponse:
        """Generate complete response for a question"""
        try:
            logger.info(f"Generating response for question: {question[:100]}...")
            
            if not relevant_clauses:
                return HackRxResponse(
                    question=question,
                    answer="I couldn't find relevant information in the provided documents to answer this question.",
                    clauses_used=[],
                    decision_rationale="No relevant clauses were found in the documents that match this question."
                )
            
            # Format clauses for LLM
            formatted_clauses = self._format_clauses_for_llm(relevant_clauses)
            
            # Generate answer
            if self.openai_api_key and self.openai_api_key.startswith('sk-'):
                try:
                    # Use OpenAI API
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": self.user_prompt_template.format(
                            question=question,
                            clauses=formatted_clauses
                        )}
                    ]
                    
                    answer = await self._call_openai_api(messages)
                    
                except Exception as e:
                    logger.warning(f"OpenAI API failed, using fallback: {str(e)}")
                    answer = self._generate_fallback_answer(question, relevant_clauses)
            else:
                # Use fallback method
                answer = self._generate_fallback_answer(question, relevant_clauses)
            
            # Extract clause excerpts
            clause_excerpts = self._extract_clause_excerpts(relevant_clauses)
            
            # Generate decision rationale
            rationale = self._generate_decision_rationale(question, relevant_clauses, answer)
            
            response = HackRxResponse(
                question=question,
                answer=answer,
                clauses_used=clause_excerpts,
                decision_rationale=rationale
            )
            
            logger.info("Response generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            
            # Return error response
            return HackRxResponse(
                question=question,
                answer="I encountered an error while processing your question. Please try again.",
                clauses_used=[],
                decision_rationale=f"An error occurred during response generation: {str(e)}"
            )

    def _validate_response(self, response: HackRxResponse) -> HackRxResponse:
        """Validate and clean up the response"""
        # Ensure answer is not empty
        if not response.answer or response.answer.strip() == "":
            response.answer = "I was unable to generate a proper answer for this question based on the available information."
        
        # Limit answer length
        if len(response.answer) > 1000:
            response.answer = response.answer[:997] + "..."
        
        # Ensure clauses_used is a list of strings
        if not isinstance(response.clauses_used, list):
            response.clauses_used = []
        
        response.clauses_used = [str(clause) for clause in response.clauses_used]
        
        # Limit number of clause excerpts
        if len(response.clauses_used) > 5:
            response.clauses_used = response.clauses_used[:5]
        
        # Ensure rationale is not empty
        if not response.decision_rationale or response.decision_rationale.strip() == "":
            response.decision_rationale = "The response was generated based on document analysis and clause matching."
        
        return response

    async def close(self):
        """Clean up resources"""
        if self.session and not self.session.closed:
            await self.session.close()