"""02 -- Softmax: the exchange rate from raw scores to a budget.

TLDR: exponentiate each score, then divide by the total. Two lines of arithmetic.
"""
import math
from common import softmax, table, notice

print(__doc__)

scores = [2.0, 1.0, 0.0]          # rain, cloud, sun -- raw "enthusiasm", any number
labels = ["rain", "cloud", "sun"]

print("Step by step, by hand:\n")
e = [math.exp(s) for s in scores]
tot = sum(e)
table([[labels[i], scores[i], round(e[i], 3), f"{e[i]:.3f}/{tot:.3f}", f"{e[i]/tot:.3f}"]
       for i in range(3)],
      ["option", "score", "e^score", "divide by total", "probability"])
print(f"\n(total of the e^score column = {tot:.3f}; the probabilities sum to {sum(e)/tot:.3f})")

print("\n\nProperty 1 -- softmax only cares about GAPS between scores.\n")
table([[str(s), str([round(x, 3) for x in softmax(s)])] for s in
       [[2, 1, 0], [102, 101, 100], [-8, -9, -10]]],
      ["scores", "probabilities"])

print("\n\nProperty 2 -- the gap IS the confidence.\n")
rows = []
for gap in [0, 0.5, 1, 2, 4, 8]:
    p = softmax([gap, 0, 0])
    rows.append([gap, f"{p[0]:.1%}", f"{p[1]:.1%}", f"{p[2]:.1%}"])
table(rows, ["winner's lead", "rain", "cloud", "sun"])

print("\n\nProperty 3 -- it never outputs exactly 0 or exactly 1.\n")
p = softmax([50, 0, 0])
print(f"  scores [50, 0, 0]  ->  {p[0]:.12f}, {p[1]:.3e}, {p[2]:.3e}")

notice(
  "'logits' / 'scores' / 'unnormalised' all mean: the numbers BEFORE softmax.",
  "Why exponentiate? (a) e^x is always positive, so no negative probabilities.",
  "  (b) it makes only the GAPS matter -- adding 100 to everything changes nothing.",
  "Why 'soft' max? Hard max would answer [1, 0, 0] -- winner takes all, a cliff.",
  "  Softmax is the smooth version: the winner gets most, not all. Smooth means",
  "  you can differentiate it, which means you can train through it.",
  "Property 3 matters more than it looks. Keep reading to 05.",
)
