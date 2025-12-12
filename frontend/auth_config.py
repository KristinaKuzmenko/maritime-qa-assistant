"""
Simple authentication for MVP.
Uses streamlit-authenticator with JSON credentials file.

First run: set environment variables to create admin user:
  ADMIN_USERNAME=admin
  ADMIN_PASSWORD=your_secure_password

User management:
  from auth_config import add_user, remove_user, list_users
  add_user("john", "password123", "John Doe", "user")
  remove_user("john")
  list_users()
"""

import streamlit_authenticator as stauth
import json
import sys
from pathlib import Path
from typing import Literal

# Add backend to path to import settings
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from core.config import settings

# Path to credentials file
CREDENTIALS_FILE = Path(__file__).parent / "credentials.json"


def _hash_password(password: str) -> str:
    """Hash password using streamlit-authenticator 0.4.x API."""
    try:
        # Version 0.4.x: Hasher.hash() is a static method
        return stauth.Hasher.hash(password)
    except (AttributeError, TypeError):
        # Fallback to bcrypt directly if streamlit-authenticator fails
        import bcrypt
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _load_creds() -> dict:
    """Load credentials from file."""
    if not CREDENTIALS_FILE.exists():
        return {"usernames": {}}
    with open(CREDENTIALS_FILE, "r") as f:
        return json.load(f)


def _save_creds(creds: dict) -> None:
    """Save credentials to file."""
    with open(CREDENTIALS_FILE, "w") as f:
        json.dump(creds, f, indent=2)


def add_user(
    username: str,
    password: str,
    name: str = "",
    role: Literal["admin", "user"] = "user",
    email: str = ""
) -> str:
    """
    Add a new user.
    
    Example:
        add_user("john", "securepass123", "John Doe", "user")
    """
    creds = _load_creds()
    
    if username in creds.get("usernames", {}):
        return f"❌ User '{username}' already exists"
    
    creds["usernames"][username] = {
        "name": name or username,
        "password": _hash_password(password),
        "role": role,
        "email": email or f"{username}@example.com"
    }
    
    _save_creds(creds)
    return f"✅ User '{username}' created (role: {role})"


def remove_user(username: str) -> str:
    """Remove a user by username."""
    creds = _load_creds()
    
    if username not in creds.get("usernames", {}):
        return f"❌ User '{username}' not found"
    
    del creds["usernames"][username]
    _save_creds(creds)
    return f"✅ User '{username}' removed"


def change_password(username: str, new_password: str) -> str:
    """Change user's password."""
    creds = _load_creds()
    
    if username not in creds.get("usernames", {}):
        return f"❌ User '{username}' not found"
    
    creds["usernames"][username]["password"] = _hash_password(new_password)
    _save_creds(creds)
    return f"✅ Password changed for '{username}'"


def list_users() -> list[dict]:
    """List all users (without passwords)."""
    creds = _load_creds()
    return [
        {"username": u, "name": d["name"], "role": d["role"], "email": d["email"]}
        for u, d in creds.get("usernames", {}).items()
    ]


def _init_credentials():
    """Initialize credentials file. Admin created from settings on first run."""
    if CREDENTIALS_FILE.exists():
        with open(CREDENTIALS_FILE, "r") as f:
            return json.load(f)
    
    # First run - create admin from settings
    admin_user = settings.admin_username
    admin_pass = settings.admin_password
    
    if not admin_pass:
        raise ValueError(
            "❌ First run requires ADMIN_PASSWORD in .env file!\n"
            "Add to your .env:\n"
            "  ADMIN_PASSWORD=your_secure_password\n"
            "  ADMIN_USERNAME=admin  # optional, default: admin"
        )
    
    creds = {
        "usernames": {
            admin_user: {
                "name": "Administrator",
                "password": _hash_password(admin_pass),
                "role": "admin",
                "email": f"{admin_user}@example.com"
            }
        }
    }
    
    _save_creds(creds)
    print(f"✅ Created credentials file with admin user: {admin_user}")
    
    return creds


# Load credentials
credentials = _init_credentials()

# Cookie config
cookie = {
    'name': 'maritime_auth',
    'key': settings.auth_secret_key,
    'expiry_days': 1
}