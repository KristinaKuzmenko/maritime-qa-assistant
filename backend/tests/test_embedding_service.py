"""
Embedding Service Tests

Comprehensive tests for text embedding generation including:
- Caching with FIFO eviction
- Token limit validation
- Retry logic for API errors
- Text preparation
- Statistics tracking

Run with: pytest test_embedding_service.py -v
"""

import pytest
import hashlib
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import openai

from services.embedding_service import EmbeddingService


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_openai_client():
    """Create mock OpenAI client."""
    with patch('services.embedding_service.openai.AsyncOpenAI') as mock_class:
        mock_client = Mock()
        mock_client.embeddings = Mock()
        mock_client.embeddings.create = AsyncMock(return_value=Mock(
            data=[Mock(embedding=[0.1] * 1536)]
        ))
        mock_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def service(mock_openai_client):
    """Create EmbeddingService with mocked client."""
    with patch('services.embedding_service.openai.AsyncOpenAI') as mock_class:
        mock_class.return_value = mock_openai_client
        svc = EmbeddingService(api_key="test-key")
        svc.client = mock_openai_client
        return svc


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestInitialization:
    """Test service initialization."""
    
    def test_init_with_api_key(self):
        """Test initialization with API key."""
        with patch('services.embedding_service.openai.AsyncOpenAI') as mock_class:
            service = EmbeddingService(api_key="test-api-key")
            
            mock_class.assert_called_once_with(api_key="test-api-key")
    
    def test_init_default_model(self):
        """Test default model is text-embedding-3-small."""
        with patch('services.embedding_service.openai.AsyncOpenAI'):
            service = EmbeddingService(api_key="test")
            
            assert service.model == "text-embedding-3-small"
    
    def test_init_custom_model(self):
        """Test custom model initialization."""
        with patch('services.embedding_service.openai.AsyncOpenAI'):
            service = EmbeddingService(
                api_key="test",
                model="text-embedding-3-large"
            )
            
            assert service.model == "text-embedding-3-large"
    
    def test_init_dimension(self):
        """Test dimension is set correctly."""
        with patch('services.embedding_service.openai.AsyncOpenAI'):
            service = EmbeddingService(api_key="test")
            
            assert service.dimension == 1536
    
    def test_init_empty_cache(self):
        """Test cache is initialized empty."""
        with patch('services.embedding_service.openai.AsyncOpenAI'):
            service = EmbeddingService(api_key="test")
            
            assert service._cache == {}
            assert service._cache_hits == 0
            assert service._cache_misses == 0


# =============================================================================
# CREATE EMBEDDING TESTS
# =============================================================================

class TestCreateEmbedding:
    """Test create_embedding method."""
    
    @pytest.mark.asyncio
    async def test_create_embedding_success(self, service, mock_openai_client):
        """Test successful embedding creation."""
        result = await service.create_embedding("test text")
        
        assert len(result) == 1536
        mock_openai_client.embeddings.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_embedding_returns_vector(self, service, mock_openai_client):
        """Test embedding returns correct vector."""
        mock_openai_client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=[0.5] * 1536)]
        )
        
        result = await service.create_embedding("test")
        
        assert result == [0.5] * 1536
    
    @pytest.mark.asyncio
    async def test_create_embedding_cleans_text(self, service, mock_openai_client):
        """Test text is cleaned before embedding."""
        await service.create_embedding("  text  with   spaces  ")
        
        call_args = mock_openai_client.embeddings.create.call_args
        # Text should be normalized
        assert call_args[1]['input'] == "text with spaces"


# =============================================================================
# CACHING TESTS
# =============================================================================

class TestCaching:
    """Test caching functionality."""
    
    @pytest.mark.asyncio
    async def test_cache_hit_skips_api(self, service, mock_openai_client):
        """Test cache hit skips API call."""
        # First call
        await service.create_embedding("test text")
        assert mock_openai_client.embeddings.create.call_count == 1
        
        # Second call - should use cache
        await service.create_embedding("test text")
        assert mock_openai_client.embeddings.create.call_count == 1
    
    @pytest.mark.asyncio
    async def test_cache_miss_calls_api(self, service, mock_openai_client):
        """Test cache miss calls API."""
        await service.create_embedding("text one")
        await service.create_embedding("text two")
        
        assert mock_openai_client.embeddings.create.call_count == 2
    
    @pytest.mark.asyncio
    async def test_cache_stats_updated(self, service, mock_openai_client):
        """Test cache statistics are updated."""
        await service.create_embedding("text")
        assert service._cache_misses == 1
        assert service._cache_hits == 0
        
        await service.create_embedding("text")
        assert service._cache_misses == 1
        assert service._cache_hits == 1
    
    @pytest.mark.asyncio
    async def test_cache_fifo_eviction(self, service, mock_openai_client):
        """Test FIFO eviction when cache is full."""
        service._cache_max_size = 3
        
        # Fill cache
        await service.create_embedding("text1")
        await service.create_embedding("text2")
        await service.create_embedding("text3")
        
        assert len(service._cache) == 3
        first_key = service._get_cache_key("text1")
        assert first_key in service._cache
        
        # Add one more - should evict first
        await service.create_embedding("text4")
        
        assert len(service._cache) == 3
        assert first_key not in service._cache
    
    def test_cache_key_deterministic(self, service):
        """Test cache key is deterministic."""
        key1 = service._get_cache_key("test text")
        key2 = service._get_cache_key("test text")
        
        assert key1 == key2
    
    def test_cache_key_different_for_different_text(self, service):
        """Test different texts have different keys."""
        key1 = service._get_cache_key("text one")
        key2 = service._get_cache_key("text two")
        
        assert key1 != key2
    
    def test_cache_key_is_md5(self, service):
        """Test cache key is MD5 hash."""
        text = "test text"
        expected = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        result = service._get_cache_key(text)
        
        assert result == expected


# =============================================================================
# TOKEN VALIDATION TESTS
# =============================================================================

class TestTokenValidation:
    """Test token limit validation."""
    
    @pytest.mark.asyncio
    async def test_long_text_raises_error(self, service):
        """Test text exceeding token limit raises ValueError."""
        # Create text > 8191 tokens (~32k+ chars)
        long_text = "word " * 10000  # ~50k chars, ~12k tokens
        
        with pytest.raises(ValueError) as exc_info:
            await service.create_embedding(long_text)
        
        assert "exceeds model limit" in str(exc_info.value)
        assert "8191" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_text_at_limit_accepted(self, service, mock_openai_client):
        """Test text at token limit is accepted."""
        # Text at approximately 8000 tokens (~32k chars)
        text = "word " * 6400  # ~32k chars, ~8k tokens
        
        # Should not raise
        result = await service.create_embedding(text)
        
        assert len(result) == 1536
    
    @pytest.mark.asyncio
    async def test_short_text_accepted(self, service, mock_openai_client):
        """Test short text is accepted."""
        result = await service.create_embedding("short text")
        
        assert len(result) == 1536


# =============================================================================
# RETRY LOGIC TESTS
# =============================================================================

class TestRetryLogic:
    """Test retry logic for API errors."""
    
    @pytest.mark.asyncio
    async def test_rate_limit_retry(self, service, mock_openai_client):
        """Test retry on RateLimitError."""
        # First two calls fail, third succeeds
        mock_openai_client.embeddings.create.side_effect = [
            openai.RateLimitError("Rate limit", response=Mock(), body=None),
            openai.RateLimitError("Rate limit", response=Mock(), body=None),
            Mock(data=[Mock(embedding=[0.1] * 1536)]),
        ]
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await service.create_embedding("test")
        
        assert len(result) == 1536
        assert mock_openai_client.embeddings.create.call_count == 3
    
    @pytest.mark.asyncio
    async def test_rate_limit_exhausted_returns_zero(self, service, mock_openai_client):
        """Test zero embedding returned when retries exhausted."""
        mock_openai_client.embeddings.create.side_effect = openai.RateLimitError(
            "Rate limit", response=Mock(), body=None
        )
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await service.create_embedding("test", retry_count=3)
        
        assert result == [0.0] * 1536
    
    @pytest.mark.asyncio
    async def test_api_error_retry(self, service, mock_openai_client):
        """Test retry on APIError."""
        mock_openai_client.embeddings.create.side_effect = [
            openai.APIError("Server error", request=Mock(), body=None),
            Mock(data=[Mock(embedding=[0.2] * 1536)]),
        ]
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await service.create_embedding("test")
        
        assert len(result) == 1536
    
    @pytest.mark.asyncio
    async def test_unexpected_error_retry(self, service, mock_openai_client):
        """Test retry on unexpected errors."""
        mock_openai_client.embeddings.create.side_effect = [
            Exception("Unexpected error"),
            Mock(data=[Mock(embedding=[0.3] * 1536)]),
        ]
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await service.create_embedding("test")
        
        assert len(result) == 1536
    
    @pytest.mark.asyncio
    async def test_custom_retry_count(self, service, mock_openai_client):
        """Test custom retry count."""
        mock_openai_client.embeddings.create.side_effect = Exception("Error")
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await service.create_embedding("test", retry_count=5)
        
        assert mock_openai_client.embeddings.create.call_count == 5


# =============================================================================
# TEXT PREPARATION TESTS
# =============================================================================

class TestTextPreparation:
    """Test _prepare_text method."""
    
    def test_strip_whitespace(self, service):
        """Test leading/trailing whitespace is stripped."""
        result = service._prepare_text("  text  ")
        
        assert result == "text"
    
    def test_normalize_internal_whitespace(self, service):
        """Test internal whitespace is normalized."""
        result = service._prepare_text("word1   word2\t\tword3")
        
        assert result == "word1 word2 word3"
    
    def test_preserve_content(self, service):
        """Test content is preserved."""
        result = service._prepare_text("Hello World")
        
        assert result == "Hello World"
    
    def test_empty_string(self, service):
        """Test empty string handling."""
        result = service._prepare_text("")
        
        assert result == ""
    
    def test_whitespace_only(self, service):
        """Test whitespace-only string."""
        result = service._prepare_text("   \t\n   ")
        
        assert result == ""


# =============================================================================
# STATISTICS TESTS
# =============================================================================

class TestStatistics:
    """Test get_stats method."""
    
    def test_stats_initial(self, service):
        """Test initial statistics."""
        stats = service.get_stats()
        
        assert stats["cache_size"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert stats["hit_rate"] == 0.0
    
    @pytest.mark.asyncio
    async def test_stats_after_requests(self, service, mock_openai_client):
        """Test statistics after requests."""
        await service.create_embedding("text1")
        await service.create_embedding("text2")
        await service.create_embedding("text1")  # Cache hit
        
        stats = service.get_stats()
        
        assert stats["cache_size"] == 2
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 2
        assert stats["hit_rate"] == 1/3
    
    def test_stats_cache_max_size(self, service):
        """Test cache max size in stats."""
        stats = service.get_stats()
        
        assert stats["cache_max_size"] == service._cache_max_size


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases."""
    
    @pytest.mark.asyncio
    async def test_empty_text(self, service, mock_openai_client):
        """Test empty text handling."""
        result = await service.create_embedding("")
        
        assert len(result) == 1536
    
    @pytest.mark.asyncio
    async def test_whitespace_text(self, service, mock_openai_client):
        """Test whitespace-only text."""
        service._cache.clear()
        result = await service.create_embedding("   ")
        
        assert len(result) == 1536
    
    @pytest.mark.asyncio
    async def test_unicode_text(self, service, mock_openai_client):
        """Test Unicode text handling."""
        result = await service.create_embedding("Проверка 中文 العربية 🚀")
        
        assert len(result) == 1536
    
    @pytest.mark.asyncio
    async def test_special_characters(self, service, mock_openai_client):
        """Test special characters."""
        result = await service.create_embedding("Text with © ™ ® symbols")
        
        assert len(result) == 1536
    
    @pytest.mark.asyncio
    async def test_newlines_in_text(self, service, mock_openai_client):
        """Test text with newlines."""
        result = await service.create_embedding("Line1\nLine2\nLine3")
        
        assert len(result) == 1536


# =============================================================================
# ADD TO CACHE TESTS
# =============================================================================

class TestAddToCache:
    """Test _add_to_cache method."""
    
    def test_add_to_empty_cache(self, service):
        """Test adding to empty cache."""
        service._add_to_cache("key1", [0.1] * 1536)
        
        assert "key1" in service._cache
        assert len(service._cache) == 1
    
    def test_fifo_eviction(self, service):
        """Test FIFO eviction when full."""
        service._cache_max_size = 2
        
        service._add_to_cache("key1", [0.1] * 1536)
        service._add_to_cache("key2", [0.2] * 1536)
        service._add_to_cache("key3", [0.3] * 1536)
        
        assert "key1" not in service._cache
        assert "key2" in service._cache
        assert "key3" in service._cache
    
    def test_cache_stores_embedding(self, service):
        """Test cache stores correct embedding."""
        embedding = [0.5] * 1536
        service._add_to_cache("key", embedding)
        
        assert service._cache["key"] == embedding


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
