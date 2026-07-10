"""Input validation utilities"""

import re


def validate_model_name(name: str) -> tuple[bool, str]:
    """Validate model name format"""
    if not name:
        return False, "Model name cannot be empty"

    if len(name) > 100:
        return False, "Model name too long (max 100 characters)"

    if not re.match(r"^[a-zA-Z0-9_]+$", name):
        return False, "Model name must be alphanumeric with underscores only"

    return True, ""


def validate_market(market: str) -> tuple[bool, str]:
    """Validate market identifier"""
    allowed_markets = {"vn_stock", "crypto_spot", "crypto_perp", "vn_derivatives"}

    if market not in allowed_markets:
        return False, f"Market must be one of {allowed_markets}"

    return True, ""


def sanitize_string(value: str, max_length: int = 255) -> str:
    """Sanitize string input"""
    if not isinstance(value, str):
        return ""

    # Remove potentially dangerous characters
    sanitized = re.sub(r"[<>\"\'%;()&+]", "", value)

    # Truncate if too long
    return sanitized[:max_length]
