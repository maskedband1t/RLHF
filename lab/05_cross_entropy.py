"""05 -- Cross-entropy: the grade on your forecast. This is the training loss.

TLDR: how surprised were you by what actually happened? Lower is better.
It looks at ONE number: the belief you put on the thing that occurred.
"""
import math
from common import entropy_bits, cross_entropy_bits, kl_bits, table, notice

print(__doc__)

print("It rained. Here is everyone's bill.\n")
rows = []
for name, p in [("nailed it",   [0.90, 0.07, 0.03]),
                ("leaning",     [0.70, 0.20, 0.10]),
                ("coin flip",   [0.50, 0.30, 0.20]),
                ("wrong",       [0.10, 0.50, 0.40]),
                ("badly wrong", [0.01, 0.50, 0.49]),
                ("ruled it out",[0.00, 0.50, 0.50])]:
    truth = [1.0, 0.0, 0.0]                      # one-hot: rain happened
    try:
        loss = f"{cross_entropy_bits(truth, p):.3f}"
    except ValueError:
        loss = "INFINITE"
    if p[0] == 0.0:
        loss = "INFINITE"
    rows.append([name, f"{p[0]:.2f}", loss])
table(rows, ["forecast", "belief on RAIN", "loss (bits)"])

print("\n  Notice the other two columns never appeared. The loss used p[rain] only.")

print("\n\nWhy the last row is fatal, and what saves you from it:\n")
for q in [0.1, 0.01, 1e-4, 1e-8]:
    print(f"  belief {q:<8} -> loss {math.log2(1/q):6.2f} bits")
print("  belief 0        -> loss    inf bits  (log2(1/0))")
print("\n  Softmax can never output exactly 0 (see 02, property 3).")
print("  That is not a detail -- it is what keeps the loss finite. The pieces fit.")

print("\n\nThe decomposition. Truth is itself uncertain: [0.70, 0.20, 0.10].\n")
truth = [0.70, 0.20, 0.10]
rows = []
for name, p in [("perfect",  [0.70, 0.20, 0.10]),
                ("close",    [0.60, 0.25, 0.15]),
                ("off",      [0.30, 0.40, 0.30]),
                ("backwards",[0.10, 0.20, 0.70])]:
    rows.append([name, f"{cross_entropy_bits(truth, p):.3f}",
                 f"{entropy_bits(truth):.3f}", f"{kl_bits(truth, p):.3f}"])
table(rows, ["your forecast", "cross-entropy", "entropy of truth", "KL (your error)"])

notice(
  "cross-entropy = entropy(truth) + KL(truth || you).",
  "  Your grade = irreducible uncertainty nobody could beat, PLUS how wrong you were.",
  "  Only the second part is yours to fix. The floor is never 0, and that's fine.",
  "THIS IS HOW LANGUAGE MODELS ARE TRAINED. 'Next-token prediction' = for each",
  "  position, raise the belief on the word that actually came next. Because the",
  "  budget must sum to 1 (see 01), raising the right word lowers every wrong word",
  "  automatically. You never have to push the wrong answers down by hand.",
  "perplexity = 2 ^ (average loss in bits). PPL 20 = 'as confused as a 20-sided die'.",
)
