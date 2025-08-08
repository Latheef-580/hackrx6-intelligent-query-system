import logging
import asyncio
import aiohttp
import json
from typing import List, Dict, Any
import re

from app.config import settings

logger = logging.getLogger(__name__)

class ResponseBuilder:
    def __init__(self):
        self.openai_api_key = settings.OPENAI_API_KEY
        self.model_name = settings.LLM_MODEL
        self.max_tokens = settings.MAX_RESPONSE_TOKENS
        self.session = None
        
        # Prompt templates
        self.system_prompt = """You are an expert AI assistant specializing in insurance, legal, HR, and compliance document analysis. Your task is to provide accurate, helpful answers to questions based on the provided document context.

Guidelines:
1. Answer questions directly and accurately based ONLY on the provided context
2. If the context doesn't contain enough information, clearly state this limitation
3. Use professional, clear language appropriate for business contexts
4. Focus on practical implications and actionable information
5. Be concise but thorough - aim for 2-4 sentences unless more detail is needed
6. If you cannot answer based on the provided context, say so clearly"""

        self.user_prompt_template = """Based on the following document context, please answer this question:

QUESTION: {question}

DOCUMENT CONTEXT:
{context}

Please provide a clear, accurate answer based on this context. If the context doesn't fully address the question, mention what information is missing."""

    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=60)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

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

    def _generate_fallback_answer(self, question: str, context: str) -> str:
        """Generate a simple fallback answer without LLM"""
        if not context or not context.strip():
            return "I couldn't find relevant information in the provided documents to answer this question."
        
        # Simple rule-based response generation
        # Take first 300 characters of context
        context_preview = context[:300]
        
        # Basic answer construction
        if len(context) > 300:
            answer = f"Based on the relevant document sections: {context_preview}..."
        else:
            answer = f"Based on the relevant document sections: {context}"
        
        return answer

    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer for a question using the provided context"""
        try:
            logger.info(f"Generating answer for question: {question[:100]}...")
            
            if not context or not context.strip():
                return "I couldn't find relevant information in the provided documents to answer this question."
            
            # Generate answer
            if self.openai_api_key and self.openai_api_key.startswith('sk-'):
                try:
                    # Use OpenAI API
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": self.user_prompt_template.format(
                            question=question,
                            context=context
                        )}
                    ]
                    
                    # Run async function in sync context
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        answer = loop.run_until_complete(self._call_openai_api(messages))
                    finally:
                        loop.close()
                    
                except Exception as e:
                    logger.warning(f"OpenAI API failed, using fallback: {str(e)}")
                    answer = self._generate_fallback_answer(question, context)
            else:
                # Use fallback method
                answer = self._generate_fallback_answer(question, context)
            
            # Clean up answer
            answer = answer.strip()
            if not answer:
                answer = "I was unable to generate a proper answer for this question based on the available information."
            
            # Limit answer length
            if len(answer) > 1000:
                answer = answer[:997] + "..."
            
            logger.info("Answer generated successfully")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "I encountered an error while processing your question. Please try again."

    async def close(self):
        """Clean up resources"""
        if self.session and not self.session.closed:
            await self.session.close()