"""Environment and utility functions."""

import logging
import os
import sys
from pathlib import Path

from ..config import ENV_FILE, HF_TOKEN_KEY, LOG_FORMAT, LOG_DATE_FORMAT

# Keyring support
try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False

SERVICE_NAME = "manhua_cleaner"


def _read_token_from_file() -> str | None:
    """Internal helper to read token from .env file (fallback method)."""
    env_path = Path(ENV_FILE)
    if env_path.exists():
        try:
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f"{HF_TOKEN_KEY}="):
                        token = line.split('=', 1)[1].strip()
                        if token and token != 'your_token_here':
                            os.environ[HF_TOKEN_KEY] = token
                            return token
        except Exception as e:
            logging.getLogger(__name__).debug(f"Failed to read {ENV_FILE}: {e}")
    return None


def store_token_secure(token: str) -> bool:
    """Store token in system keyring if available, otherwise file.
    
    Returns:
        True if stored in keyring, False if fallback to file
    """
    if KEYRING_AVAILABLE:
        try:
            keyring.set_password(SERVICE_NAME, "hf_token", token)
            return True
        except keyring.Error:
            pass
    # Fallback to file with warning
    logger = logging.getLogger(__name__)
    logger.warning("keyring not available, storing token in plaintext")
    return False


def retrieve_token_secure() -> str | None:
    """Retrieve token from keyring or file (fallback)."""
    if KEYRING_AVAILABLE:
        try:
            return keyring.get_password(SERVICE_NAME, "hf_token")
        except keyring.Error:
            pass
    # Fallback to file
    return _read_token_from_file()

# Default log file location
DEFAULT_LOG_FILE = "manhua_cleaner.log"


def setup_logging(level: int = logging.INFO, log_file: str | None = DEFAULT_LOG_FILE) -> None:
    """Set up logging configuration.
    
    Logs are written to both console (stderr) and a file.
    
    Args:
        level: Logging level
        log_file: Path to log file (None to disable file logging)
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
            handlers.append(file_handler)
        except (OSError, PermissionError) as e:
            # Fall back to console-only if file logging fails
            print(f"Warning: Could not open log file '{log_file}': {e}", file=sys.stderr)
    
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=handlers,
        force=True  # Replace any existing handlers
    )


def load_hf_token() -> str | None:
    """Load Hugging Face token from environment, keyring, or .env file.
    
    Returns:
        Token string or None if not found
    """
    # First check environment
    token = os.getenv(HF_TOKEN_KEY)
    if token:
        return token
    
    # Try secure storage
    token = retrieve_token_secure()
    if token:
        os.environ[HF_TOKEN_KEY] = token
        return token
    
    return None


def save_hf_token(token: str) -> None:
    """Save Hugging Face token to system keyring if available, otherwise .env file.
    
    Args:
        token: Token to save
        
    Raises:
        ValueError: If token is empty
    """
    if not token or not token.strip():
        raise ValueError("Token cannot be empty")
    
    token = token.strip()
    
    # Set in environment
    os.environ[HF_TOKEN_KEY] = token
    
    # Try secure storage first
    if not store_token_secure(token):
        # Fallback to file storage
        env_path = Path(ENV_FILE)
        
        # Read existing content
        lines = []
        token_found = False
        
        if env_path.exists():
            try:
                with open(env_path) as f:
                    lines = f.readlines()
            except Exception:
                pass
        
        # Update or add token
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{HF_TOKEN_KEY}="):
                lines[i] = f"{HF_TOKEN_KEY}={token}\n"
                token_found = True
                break
        
        if not token_found:
            # Add comment if file is new
            if not lines:
                lines.append(f"# Hugging Face API Token\n")
                lines.append(f"# Get your token from: https://huggingface.co/settings/tokens\n")
                lines.append(f"# SECURITY: This file contains sensitive data.\n")
            lines.append(f"{HF_TOKEN_KEY}={token}\n")
        
        # Write back
        with open(env_path, 'w') as f:
            f.writelines(lines)
        
        # Set restrictive permissions on Unix systems (0o600 = rw-------)
        try:
            os.chmod(env_path, 0o600)
        except Exception:
            # Windows or permission error - ignore
            pass
        
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Token saved to {env_path.absolute()}. "
            "Note: Token is stored in plaintext. Ensure this file is not committed to version control."
        )


def validate_hf_token(token: str) -> dict:
    """Validate a Hugging Face token via API.
    
    Args:
        token: Token to validate
        
    Returns:
        Dictionary with validation results
    """
    import requests
    
    try:
        response = requests.get(
            "https://huggingface.co/api/whoami",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            return {
                "valid": True,
                "username": data.get("name", "Unknown"),
                "orgs": data.get("orgs", []),
            }
        else:
            return {
                "valid": False,
                "error": f"Invalid token (HTTP {response.status_code})"
            }
            
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }
