"""03 -- Temperature: one knob, and all it does is stretch or squash the gaps.

TLDR: divide the scores by T, then softmax. T<1 = bolder. T>1 = more hedged.
"""
from common import softmax, entropy_bits, table, notice

print(__doc__)
scores = [2.0, 1.0, 0.0]
print(f"Same scores every row: {scores}\n")

rows = []
for T in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 100.0]:
    p = softmax(scores, T)
    rows.append([T, f"{p[0]:.1%}", f"{p[1]:.1%}", f"{p[2]:.1%}",
                 f"{entropy_bits(p):.3f}"])
table(rows, ["T", "rain", "cloud", "sun", "entropy (bits)"])

notice(
  "T -> 0   : always pick the top score. This is 'greedy' / 'temperature 0'.",
  "T = 1    : the model's own honest opinion, untouched.",
  "T -> inf : every option equally likely. A blind coin flip.",
  "Dividing by T=0.5 doubles every gap -> peakier. T=2 halves them -> flatter.",
  "That is the entire mechanism. 'Temperature' is a gap multiplier, nothing else.",
)
