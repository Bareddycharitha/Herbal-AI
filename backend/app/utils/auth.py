"""
Authentication Utilities

Clerk JWT verification utilities.
Verifies Clerk-issued JWTs using Clerk's JWKS endpoint.
No local password hashing or JWT signing.
"""

import json
import time
from typing import Optional

from jose import jwt, jwk
from jose.exceptions import ExpiredSignatureError, JWTError

from backend.app.config import settings
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)

# Cache for JWKS keys to avoid fetching on every request
_jwks_cache: dict[str, object] = {}
_jwks_fetched_at: float = 0
_JWKS_CACHE_TTL = 300  # 5 minutes


def _fetch_jwks() -> dict:
    """Fetch Clerk's JWKS from the JWKS URL.

    Results are cached for JWKS_CACHE_TTL seconds to avoid
    repeated network calls on every request.

    Returns:
        The JWKS keys dictionary.

    Raises:
        RuntimeError: If the JWKS URL is not configured or fetching fails.
    """
    global _jwks_cache, _jwks_fetched_at

    now = time.time()
    if _jwks_cache and (now - _jwks_fetched_at) < _JWKS_CACHE_TTL:
        return _jwks_cache

    if not settings.clerk_jwks_url:
        raise RuntimeError(
            "CLERK_JWKS_URL must be set via the CLERK_JWKS_URL environment variable."
        )

    try:
        import httpx

        with httpx.Client(timeout=10.0) as client:
            response = client.get(settings.clerk_jwks_url)
            response.raise_for_status()
            jwks = response.json()
    except Exception as e:
        logger.error("Failed to fetch Clerk JWKS", error=str(e))
        raise RuntimeError(f"Failed to fetch Clerk JWKS: {e}") from e

    _jwks_cache = jwks
    _jwks_fetched_at = now

    return jwks


def verify_clerk_token(token: str) -> Optional[dict]:
    """Verify a Clerk JWT and return the user payload.

    Uses Clerk's JWKS to validate the token signature and
    checks standard claims (exp, nbf, iss).

    Args:
        token: The Clerk JWT access token.

    Returns:
        User payload dict if the token is valid, None otherwise.
    """
    try:
        jwks = _fetch_jwks()

        # Decode the token header to get the key ID (kid)
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")

        # Find the matching key in JWKS
        signing_key = None
        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                signing_key = key
                break

        if signing_key is None:
            logger.warning("No matching JWKS key found for kid", kid=kid)
            return None

        # Convert JWK to PEM format for python-jose
        jwk_obj = jwk.construct(signing_key)

        # Decode and verify the token
        payload = jwt.decode(
            token,
            jwk_obj,
            algorithms=["RS256"],
            options={
                "verify_exp": True,
                "verify_nbf": True,
                "verify_iss": False,
                "verify_aud": False,
            },
        )

        # Extract user information from payload
        user_id = payload.get("sub", "")
        email = payload.get("email", "")
        full_name = payload.get("name")

        # Warn if email is missing from token (should be configured in Clerk JWT template)
        if not email:
            logger.warning(
                "Clerk JWT missing email claim. "
                "Configure Clerk JWT template to include email claim for proper user identification."
            )

        # Extract role from public_claims if available, otherwise default to 'user'
        role = "user"
        public_claims = payload.get("public_claims")
        if isinstance(public_claims, dict):
            role = public_claims.get("role", "user")

        # Determine if user is active based on email verification
        is_active = True

        return {
            "id": user_id,
            "email": email,
            "full_name": full_name,
            "role": role,
            "is_active": is_active,
        }

    except ExpiredSignatureError:
        logger.warning("Clerk token expired")
        return None
    except JWTError as e:
        logger.warning("Clerk token verification failed", error=str(e))
        return None
    except Exception as e:
        logger.error("Unexpected error verifying Clerk token", error=str(e))
        return None


def get_user_id_from_token(token: str) -> Optional[str]:
    """Extract the Clerk user ID from a Clerk JWT access token.

    Args:
        token: The Clerk JWT access token.

    Returns:
        The Clerk user ID string, or None if the token is invalid.
    """
    payload = verify_clerk_token(token)
    if not payload:
        return None
    return payload.get("id")
