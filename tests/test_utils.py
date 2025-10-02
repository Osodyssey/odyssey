from src.utils import compute_cross_entropy, logsumexp

def test_logsumexp():
    xs = [0.0, -float('inf')]
    assert abs(logsumexp(xs) - 0.0) < 1e-6

def test_cross_entropy():
    import math
    logs = [math.log(0.5), math.log(0.5)]
    ce = compute_cross_entropy(logs)
    assert abs(ce - 0.693147) < 1e-4
