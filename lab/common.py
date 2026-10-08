"""Shared helpers. Pure stdlib -- no numpy, no install, runs anywhere python3 runs."""
import math

def softmax(scores, temperature=1.0):
    """Turn raw scores into a budget of belief that sums to 1."""
    s = [x / temperature for x in scores]
    biggest = max(s)                      # subtract the max: pure numerical hygiene,
    e = [math.exp(x - biggest) for x in s] # changes nothing (see 02), stops overflow
    total = sum(e)
    return [x / total for x in e]

def entropy_bits(p):
    """How spread out the budget is, in yes/no questions."""
    h = -sum(x * math.log2(x) for x in p if x > 0)
    return max(0.0, h)          # entropy is never negative; this kills "-0.000"

def cross_entropy_bits(truth, pred):
    """How surprised `pred` was by `truth`."""
    return -sum(t * math.log2(q) for t, q in zip(truth, pred) if t > 0)

def kl_bits(p, q):
    """How far q sits from p. = cross_entropy(p,q) - entropy(p)."""
    return sum(x * math.log2(x / y) for x, y in zip(p, q) if x > 0)

def table(rows, headers):
    w = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    line = "  ".join(str(h).ljust(w[i]) for i, h in enumerate(headers))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))

def notice(*lines):
    print()
    for l in lines:
        print("NOTICE: " + l)
