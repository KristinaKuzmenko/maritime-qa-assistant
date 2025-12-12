"""
Configuration settings for Streamlit app.
"""

# API configuration
API_BASE_URL = "http://localhost:8000"

# Pagination settings
ITEMS_PER_PAGE = 10

# File upload settings
MAX_FILE_SIZE_MB = 150
ALLOWED_FILE_TYPES = ["pdf"]

# Cache settings
CACHE_TTL = 300  # 5 minutes