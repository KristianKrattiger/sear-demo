from .schema import Claim


def gate(claims: list[Claim], phi: float) -> bool:
    """True if the report passes; False means emit the failure state."""
    if not claims:
        return False
    verified = sum(c.verified for c in claims)
    return verified / len(claims) >= phi
