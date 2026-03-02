from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
import os

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Verify API key from header"""
    # For development, use a hardcoded key. In production, use environment variables
    VALID_API_KEY = os.getenv("API_KEY", "development_key")

    if api_key != VALID_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )
    return api_key