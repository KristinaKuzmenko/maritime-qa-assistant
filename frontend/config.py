"""
Configuration settings for Streamlit app.
"""

import os

# API configuration
API_BASE_URL = "http://localhost:8000"

# Authentication configuration
AUTH_COOKIE_NAME = os.getenv("AUTH_COOKIE_NAME", "maritime_qa_auth")
AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "maritime_qa_secret_key_change_in_production")
AUTH_COOKIE_EXPIRY_DAYS = int(os.getenv("AUTH_COOKIE_EXPIRY_DAYS", "30"))

# Pagination settings
ITEMS_PER_PAGE = 10

# File upload settings
MAX_FILE_SIZE_MB = 150
ALLOWED_FILE_TYPES = ["pdf"]

# Cache settings
CACHE_TTL = 300  # 5 minutes