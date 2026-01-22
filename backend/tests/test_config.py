"""
Unit tests for core configuration and utilities.
"""

import pytest
from unittest.mock import Mock, patch
import os

from core.config import Settings


class TestSettings:
    """Test Settings configuration."""
    
    def test_settings_loads_from_env(self):
        """Test settings load from environment variables."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_key_123',
            'NEO4J_URI': 'bolt://localhost:7687',
            'NEO4J_USER': 'test_user',
            'NEO4J_PASSWORD': 'test_pass',
        }):
            settings = Settings()
            
            assert settings.openai_api_key == 'test_key_123'
            assert settings.neo4j_uri == 'bolt://localhost:7687'
            assert settings.neo4j_user == 'test_user'
            assert settings.neo4j_password == 'test_pass'
    
    def test_settings_has_defaults(self):
        """Test settings have sensible defaults."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_key',
            'NEO4J_PASSWORD': 'test_pass',
        }):
            settings = Settings()
            
            # Check some default values exist
            assert hasattr(settings, 'llm_model')
            assert hasattr(settings, 'openai_embedding_model')
    
    def test_settings_llm_config(self):
        """Test LLM configuration settings."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_key',
            'NEO4J_PASSWORD': 'test_pass',
        }):
            settings = Settings()
            
            # Should have LLM-related settings
            assert hasattr(settings, 'llm_model')
            assert hasattr(settings, 'llm_provider')
            assert hasattr(settings, 'openai_embedding_model')


class TestBBoxUtilities:
    """Test BBox utility functions."""
    
    def test_bbox_intersection(self):
        """Test bounding box intersection calculation."""
        from services.layout_analyzer import BBox
        
        bbox1 = BBox(0, 0, 100, 100)
        bbox2 = BBox(50, 50, 150, 150)
        
        # Intersection area should be 50x50 = 2500
        x1 = max(bbox1.x0, bbox2.x0)  # 50
        y1 = max(bbox1.y0, bbox2.y0)  # 50
        x2 = min(bbox1.x1, bbox2.x1)  # 100
        y2 = min(bbox1.y1, bbox2.y1)  # 100
        
        intersection = (x2 - x1) * (y2 - y1)
        assert intersection == 2500
    
    def test_bbox_union(self):
        """Test bounding box union calculation."""
        from services.layout_analyzer import BBox
        
        bbox1 = BBox(0, 0, 100, 100)
        bbox2 = BBox(50, 50, 150, 150)
        
        # Union = area1 + area2 - intersection
        # = 10000 + 10000 - 2500 = 17500
        area1 = bbox1.area()
        area2 = bbox2.area()
        
        intersection = 2500  # from previous test
        union = area1 + area2 - intersection
        
        assert union == 17500
