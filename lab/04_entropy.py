"""04 -- Entropy: how wide is your hedge?

TLDR: entropy in bits = the average number of yes/no questions needed to pin
down the answer. Pile all belief on one option -> 0 questions -> 0 entropy.
"""
from common import entropy_bits, table, notice

print(__doc__)

print("Check the 'yes/no questions' claim on cases you can verify by hand:\n")
rows = []
for n in [1, 2, 4, 8, 16, 1024]:
    p = [1.0 / n] * n
    rows.append([f"{n} equally likely options", f"{entropy_bits(p):.3f}",
                 "guess it in that many halvings"])
table(rows, ["situation", "entropy (bits)", "sanity check"])

print("\n\nNow unequal belief. Same 3 options, different spending:\n")
rows = []
for name, p in [("all in on rain", [1.0, 0.0, 0.0]),
                ("very sure",      [0.90, 0.07, 0.03]),
                ("leaning",        [0.70, 0.20, 0.10]),
                ("mild lean",      [0.40, 0.35, 0.25]),
                ("no clue",        [1/3, 1/3, 1/3])]:
    rows.append([name, str([round(x, 2) for x in p]), f"{entropy_bits(p):.3f}"])
table(rows, ["forecast", "budget", "entropy (bits)"])

print("\n\nWhere the formula comes from: entropy = YOUR AVERAGE SURPRISE.\n")
import math
p = [0.70, 0.20, 0.10]
rows = []
for lab, q in zip(["rain", "cloud", "sun"], p):
    rows.append([lab, f"{q:.2f}", f"{math.log2(1/q):.3f}", f"{q * math.log2(1/q):.3f}"])
table(rows, ["if this happens", "its prob", "surprise log2(1/p)", "prob x surprise"])
print(f"\n  sum of last column = {entropy_bits(p):.3f} bits  <- that's the entropy")

notice(
  "'surprise' of an event = log2(1/p). Certain event -> 0 surprise.",
  "  1-in-1000 event -> ~10 bits of surprise. Rare = surprising = large number.",
  "Entropy = your surprise, averaged over what you yourself think will happen.",
  "Why RLHF cares: entropy is EXPLORATION. 'Entropy collapse' = the model got",
  "  so sure of one answer it stopped trying others, and learning stalls.",
  "  'Entropy bonus' = a nudge in the loss saying 'stay a bit unsure'.",
)
