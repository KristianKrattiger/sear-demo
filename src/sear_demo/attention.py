"""Attention baseline (ENG 01 §2b, item 4).

TODO: map each retrieved chunk to its token span in the prompt, then for each
generated claim sum attention mass (mean over layers/heads) per chunk span and
store it on Claim.attention. Compare against verifier pass/fail for the
attention-weight grounding correlation metric. A hypothesis, not a result.
"""


def attach_attention(claims, trace, chunk_spans):
    return claims
