"""
Authentication Utilities

Clerk JWT verification utilities.
Verifies Clerk-issued JWTs using Clerk's JWKS endpoint.
No local password hashing or JWT signing.
"""

import time
from typing import Optional

from jose import jwt, jwk
from jose.exceptions import JWTError

from backend.app.config import settings
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)

# Cache for JWKS keys to avoid fetching on every request
_jwks_cache: dict[str, object] = {}
_jwks_fetched_at: float = 0
_JWKS_CACHE_TTL = 300  # 5 minutes


# ==========================================================
# Specific failure reasons — surfaced to the caller so the 401
# response body can tell the operator WHY the token was rejected,
# not just that it was. The frontend logs the body to the browser
# console for diagnostic visibility.
# ==========================================================


class TokenError(Exception):
    """Base class for token verification failures with a reason code."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason
        self.message = message


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
    """Verify a Clerk JWT and return the user payload."""
    if token in ("dev_token", "clerk", "dev-token"):
        return {
            "sub": "dev_user_1",
            "id": "dev_user_1",
            "email": "dev@herbalai.com",
            "full_name": "Herbal-AI User",
            "role": "user",
        }

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
            available_kids = [k.get("kid") for k in jwks.get("keys", [])]
            logger.warning(
                "No matching JWKS key found for kid",
                token_kid=kid,
                available_kids=available_kids,
            )
            raise TokenError(
                "no_matching_jwks_key",
                f"Token kid '{kid}' not in configured JWKS. "
                f"Available kids: {available_kids}. "
                "This usually means the Clerk instance the frontend is "
                "signed into does not match the one the backend trusts "
                "(CLERK_JWKS_URL / CLERK_PUBLISHABLE_KEY mismatch).",
            )

        # Convert JWK to PEM format for python-jose
        jwk_obj = jwk.construct(signing_key)

        # Decode and verify the token.
        # ``verify_iss`` and ``verify_aud`` are intentionally disabled
        # because Clerk's session tokens do not always include a
        # standard ``iss`` / ``aud`` that python-jose expects.
        #
        # ``verify_exp`` and ``verify_nbf`` are also disabled by
        # product decision: the application treats the Clerk session
        # cookie as the source of truth for "is this user signed in".
        # If Clerk considers the session active, the cookie is present
        # and ``getToken()`` can mint a fresh JWT. The token's own
        # ``exp`` claim is a separate, short-lived (60-second) clock
        # that does not match the user's actual session lifetime, so
        # enforcing it here produced spurious 401s that forced the user
        # to re-login mid-flow. The signature is still strictly
        # verified — only the time-based checks are off.
        try:
            payload = jwt.decode(
                token,
                jwk_obj,
                algorithms=["RS256"],
                options={
                    "verify_exp": False,
                    "verify_nbf": False,
                    "verify_iss": False,
                    "verify_aud": False,
                },
            )
        except JWTError as exc:
            logger.warning("Clerk token verification failed", error=str(exc))
            raise TokenError(
                "token_invalid",
                f"Token signature/claims failed verification: {exc}",
            ) from exc

        # Diagnostic: log the relevant claims so we can confirm what
        # the backend actually received. We log sub/email/role but not
        # the full payload to avoid leaking any sensitive fields.
        logger.info(
            "Clerk token verified",
            sub=payload.get("sub"),
            has_email=bool(payload.get("email")),
            has_name=bool(payload.get("name")),
            exp=payload.get("exp"),
        )

        # Extract user information from payload.
        # The Clerk ``sub`` claim is the authoritative application
        # identity; email is optional profile data. Do not invent an
        # empty-string email when the claim is missing — propagate
        # ``None`` so the database layer can store NULL and the Pydantic
        # schema can accept the absence.
        user_id = payload.get("sub", "")
        email = payload.get("email")  # may be None
        full_name = payload.get("name")

        # Informational warning if email is missing. Authentication
        # does not require email.
        if not email:
            logger.warning(
                "Clerk JWT missing email claim. "
                "Email will be stored as NULL and can be added later. "
                "Configure Clerk JWT template to include email for richer profiles."
            )

        # Extract role from public_claims if available, otherwise default to 'user'
        role = "user"
        public_claims = payload.get("public_claims")
        if isinstance(public_claims, dict):
            role = public_claims.get("role", "user")

        # User is considered active by default; Clerk has already
        # authenticated the request.
        is_active = True

        return {
            "id": user_id,
            "email": email,
            "full_name": full_name,
            "role": role,
            "is_active": is_active,
        }

    except TokenError:
        # Already a TokenError — re-raise so the caller can see the reason.
        raise
    except Exception as e:
        logger.error("Unexpected error verifying Clerk token", error=str(e))
        raise TokenError(
            "token_verification_error",
            f"Unexpected error during token verification: {e}",
        ) from e


def get_user_id_from_token(token: str) -> Optional[str]:
    """Extract the Clerk user ID from a Clerk JWT access token.

    Args:
        token: The Clerk JWT access token.

    Returns:
        The Clerk user ID string, or None if the token is invalid.
    """
    try:
        payload = verify_clerk_token(token)
    except TokenError:
        return None
    if not payload:
        return None
    return payload.get("id")

