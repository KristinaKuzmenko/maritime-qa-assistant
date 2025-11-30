"""
Embedding service for text vectorization via OpenAI API.
Handles embedding generation with caching and validation.
"""

import openai
from typing import List
import logging
import asyncio
import hashlib

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings with validation."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize embedding service.
        
        :param api_key: OpenAI API key
        :param model: Embedding model name
        """
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self.dimension = 1536  # text-embedding-3-small dimension
        
        # Simple in-memory cache
        self._cache = {}
        self._cache_max_size = 1000
        
        # Stats for monitoring
        self._cache_hits = 0
        self._cache_misses = 0
        
    async def create_embedding(
        self, 
        text: str, 
        retry_count: int = 3
    ) -> List[float]:
        """
        Create embedding for text with caching and retry logic.
        NOW WITH EXPLICIT VALIDATION - does not silently truncate.
        
        :param text: Text to embed
        :param retry_count: Number of retries on failure
        :return: Embedding vector (1536 dimensions)
        :raises ValueError: If text exceeds model token limit
        """
        # Check cache
        cache_key = self._get_cache_key(text)
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        
        self._cache_misses += 1
        
        # CRITICAL: Validate length BEFORE calling API
        token_estimate = len(text) // 4
        if token_estimate > 8191:
            logger.error(
                f"Text too long ({token_estimate} tokens est., limit 8191). "
                f"Text should be chunked before calling create_embedding!"
            )
            raise ValueError(
                f"Text exceeds model limit ({token_estimate} > 8191 tokens). "
                f"Please chunk the text before embedding. "
                f"First 100 chars: {text[:100]}..."
            )
        
        # Prepare text (cleaning only, no truncation)
        cleaned_text = self._prepare_text(text)
        
        # Retry loop
        for attempt in range(retry_count):
            try:
                response = await self.client.embeddings.create(
                    input=cleaned_text,
                    model=self.model
                )
                
                embedding = response.data[0].embedding
                
                # Cache result
                self._add_to_cache(cache_key, embedding)
                
                return embedding
                
            except openai.RateLimitError as e:
                wait_time = 2 ** attempt
                logger.warning(
                    f"Rate limit (attempt {attempt + 1}/{retry_count}), "
                    f"waiting {wait_time}s"
                )
                if attempt < retry_count - 1:
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Rate limit exceeded, returning zero embedding")
                    return [0.0] * self.dimension
            
            except openai.APIError as e:
                logger.error(f"API error (attempt {attempt + 1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return [0.0] * self.dimension
            
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return [0.0] * self.dimension
    
    def _prepare_text(self, text: str) -> str:
        """
        Clean text for API (NO TRUNCATION - validation happens before this).
        
        :param text: Raw text
        :return: Cleaned text
        """
        text = text.strip()
        text = ' '.join(text.split())  # Normalize whitespace
        
        return text
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _add_to_cache(self, key: str, embedding: List[float]):
        """Add to cache with FIFO eviction."""
        if len(self._cache) >= self._cache_max_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        
        self._cache[key] = embedding
    
    def get_stats(self) -> dict:
        """Get service statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / max(total_requests, 1)
        
        return {
            "cache_size": len(self._cache),
            "cache_max_size": self._cache_max_size,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
        }