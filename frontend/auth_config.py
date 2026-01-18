"""
User authentication using AWS DynamoDB.
Secure, scalable storage for user credentials.

Environment variables:
  DYNAMODB_USERS_TABLE: DynamoDB table name (default: dev-maritime-qa-users)
  AWS_REGION: AWS region (default: us-east-1)
  
First run: Admin user created from ADMIN_USERNAME/ADMIN_PASSWORD env vars
"""

import streamlit_authenticator as stauth
import boto3
import sys
import os
from pathlib import Path
from typing import Literal, Optional, Dict, List
from botocore.exceptions import ClientError
import logging

logger = logging.getLogger(__name__)

# Add backend to path to import settings
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from core.config import settings

# DynamoDB configuration
DYNAMODB_TABLE = os.getenv("DYNAMODB_USERS_TABLE", "dev-maritime-qa-users")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
users_table = dynamodb.Table(DYNAMODB_TABLE)


def _hash_password(password: str) -> str:
    """Hash password using streamlit-authenticator 0.4.x API."""
    try:
        return stauth.Hasher.hash(password)
    except (AttributeError, TypeError):
        import bcrypt
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _get_user(username: str) -> Optional[Dict]:
    """Get user from DynamoDB."""
    try:
        response = users_table.get_item(Key={'username': username})
        return response.get('Item')
    except ClientError as e:
        logger.error(f"Error getting user {username}: {e}")
        return None


def _load_creds() -> dict:
    """Load all credentials from DynamoDB."""
    try:
        response = users_table.scan()
        items = response.get('Items', [])
        
        # Convert DynamoDB format to streamlit-authenticator format
        credentials = {"usernames": {}}
        for item in items:
            credentials["usernames"][item['username']] = {
                "name": item.get('name', item['username']),
                "password": item['password_hash'],
                "role": item.get('role', 'user'),
                "email": item.get('email', f"{item['username']}@example.com")
            }
        
        return credentials
    except ClientError as e:
        logger.error(f"Error loading credentials: {e}")
        return {"usernames": {}}


def add_user(
    username: str,
    password: str,
    name: str = "",
    role: Literal["admin", "user"] = "user",
    email: str = ""
) -> str:
    """
    Add a new user to DynamoDB.
    
    Example:
        add_user("john", "securepass123", "John Doe", "user")
    """
    # Check if user exists
    if _get_user(username):
        return f"ERROR: user '{username}' already exists"
    
    try:
        users_table.put_item(
            Item={
                'username': username,
                'password_hash': _hash_password(password),
                'name': name or username,
                'role': role,
                'email': email or f"{username}@example.com"
            }
        )
        logger.info(f"Created user: {username} (role: {role})")
        return f"OK: user '{username}' created (role: {role})"
    except ClientError as e:
        logger.error(f"Error creating user {username}: {e}")
        return f"ERROR: {str(e)}"


def remove_user(username: str) -> str:
    """Remove a user from DynamoDB."""
    if not _get_user(username):
        return f"ERROR: user '{username}' not found"
    
    try:
        users_table.delete_item(Key={'username': username})
        logger.info(f"Deleted user: {username}")
        return f"OK: user '{username}' removed"
    except ClientError as e:
        logger.error(f"Error deleting user {username}: {e}")
        return f"ERROR: {str(e)}"


def change_password(username: str, new_password: str) -> str:
    """Change user password in DynamoDB."""
    if not _get_user(username):
        return f"ERROR: user '{username}' not found"
    
    try:
        users_table.update_item(
            Key={'username': username},
            UpdateExpression='SET password_hash = :pwd',
            ExpressionAttributeValues={':pwd': _hash_password(new_password)}
        )
        logger.info(f"Changed password for user: {username}")
        return f"OK: password changed for '{username}'"
    except ClientError as e:
        logger.error(f"Error changing password for {username}: {e}")
        return f"ERROR: {str(e)}"


def list_users() -> list[dict]:
    """List all users (without passwords) from DynamoDB."""
    try:
        response = users_table.scan(
            ProjectionExpression='username, #n, #r, email',
            ExpressionAttributeNames={'#n': 'name', '#r': 'role'}
        )
        return response.get('Items', [])
    except ClientError as e:
        logger.error(f"Error listing users: {e}")
        return []


def _init_credentials():
    """Initialize DynamoDB with default admin user if not exists."""
    admin_username = os.getenv("ADMIN_USERNAME", "admin")
    admin_password = os.getenv("ADMIN_PASSWORD", "admin")
    admin_email = os.getenv("ADMIN_EMAIL", "admin@example.com")
    
    # Check if admin exists
    if _get_user(admin_username):
        logger.info(f"Admin user '{admin_username}' already exists in DynamoDB")
        return
    
    # Create default admin
    result = add_user(
        username=admin_username,
        password=admin_password,
        name="Administrator",
        role="admin",
        email=admin_email
    )
    logger.info(f"Initialized DynamoDB: {result}")


# Initialize credentials on first import
_init_credentials()


def get_credentials():
    """Get fresh credentials from DynamoDB."""
    return _load_creds()