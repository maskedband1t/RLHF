"""01 -- What a "probability distribution" actually is.

TLDR: 100 points of belief. You must spend all of them. That's it.
"""
from common import table, notice

print(__doc__)

print("Tomorrow's weather. Three things can happen. Spend your 100 points.\n")
forecasts = {
    "confident (rain)": [0.90, 0.07, 0.03],
    "leaning rain":     [0.70, 0.20, 0.10],
    "no clue":          [0.3334, 0.3333, 0.3333],
    "certain (rain)":   [1.00, 0.00, 0.00],
}
rows = []
for name, p in forecasts.items():
    rows.append([name, f"{p[0]:.0%}", f"{p[1]:.0%}", f"{p[2]:.0%}", f"{sum(p):.2f}"])
table(rows, ["forecast", "rain", "cloud", "sun", "SUM"])

print("\nNot distributions -- and why:\n")
bad = [
    ([0.9, 0.2, 0.1], "sums to 1.2 -- you spent points you don't have"),
    ([0.5, 0.2, 0.1], "sums to 0.8 -- you left belief on the table; where did it go?"),
    ([0.9, -0.1, 0.2], "negative -- what is a -10% chance of cloud?"),
]
table([[str(p), f"{sum(p):.1f}", why] for p, why in bad], ["list", "sum", "problem"])

notice(
  "A 'distribution' is nothing but a list of non-negative numbers that add to 1.",
  "A model's output IS this list. One number per option. Not a mystery object.",
  "Because the total is fixed, belief is ZERO-SUM: to raise one option you must",
  "  lower the others. That single fact explains all of training (see 05).",
)
