"""07 -- Log-probs: why every paper is drowning in logarithms.

TLDR: probabilities multiply and vanish. Logs add and survive.
"""
import math
from common import notice

print(__doc__)

print("A 500-token answer. Say the model gave each token 5% belief.\n")
p, n = 0.05, 500

prob = 1.0
for _ in range(n):
    prob *= p
print(f"  multiplying 500 probabilities : {prob}")
print(f"  ... the true value is about 1e-{abs(round(n*math.log10(p)))}, far below the")
print(f"      smallest number a float can hold (~1e-308). It silently became zero.")
print(f"      Every gradient computed from it is now zero. Training is dead.\n")

logp = sum(math.log(p) for _ in range(n))
print(f"  adding 500 log-probabilities  : {logp:.2f}")
print(f"  perfectly ordinary number. Nothing overflowed. This is the whole reason.\n")

print("\nReading the units:\n")
for q in [1.0, 0.9, 0.5, 0.1, 0.01]:
    print(f"  prob {q:<6} -> log-prob {math.log(q):8.3f}   (0 is perfect, "
          f"more negative is worse)")

print("\n\nPerplexity, from the same numbers:\n")
avg_bits = -math.log2(p)
print(f"  average surprise per token = {avg_bits:.3f} bits")
print(f"  perplexity = 2^{avg_bits:.3f} = {2**avg_bits:.1f}")
print(f"  and 1/{p} = {1/p:.0f}. Same number. 'The model was as lost as someone")
print(f"  guessing uniformly between {1/p:.0f} options at every word.'")

notice(
  "log(a*b) = log(a) + log(b). That identity is doing all the work.",
  "'log-prob of a sequence' = sum of the log-probs of its tokens.",
  "-log(p) = log(1/p) = surprise. Same quantity, three names, pick one and keep it.",
  "Natural log (ln) vs log2 only changes the unit: nats vs bits. Ratios are unaffected.",
  "This is also why you see 'log-sum-exp' everywhere -- it is how you add",
  "  probabilities while staying in log space.",
)
