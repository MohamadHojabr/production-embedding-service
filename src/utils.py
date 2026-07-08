import hashlib
import re


def normalize_text(text: str) -> str:
    """
    Normalize text before embedding/cache key generation
    """

    text = text.strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text



def generate_text_hash(text: str) -> str:
    """
    Generate deterministic cache key
    """

    normalized = normalize_text(text)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()