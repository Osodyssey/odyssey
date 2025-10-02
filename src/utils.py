"""Utility functions: math helpers, code validators and formatters."""
from typing import List, Tuple
import ast, subprocess, sys
import math

def compute_cross_entropy(log_probs: List[float]) -> float:
    """Compute average cross-entropy given token log-probs (natural log). Complexity: O(n)"""
    if not log_probs:
        raise ValueError('log_probs empty')
    return -sum(log_probs) / len(log_probs)

def logsumexp(xs: List[float]) -> float:
    """Stable log-sum-exp implementation."""
    m = max(xs)
    s = sum(math.exp(x - m) for x in xs)
    return m + math.log(s)

def is_syntax_valid(code: str) -> Tuple[bool, str]:
    """Quick syntax check using ast.parse"""
    try:
        ast.parse(code)
        return True, ''
    except SyntaxError as e:
        return False, str(e)

def try_format_with_black(code: str) -> str:
    """Attempt to format with black if available; otherwise return original."""
    try:
        import black
        mode = black.Mode()
        return black.format_str(code, mode=mode)
    except Exception:
        return code

if __name__ == '__main__':
    sample = 'print( 1+1)'
    print('Formatted:', try_format_with_black(sample))
