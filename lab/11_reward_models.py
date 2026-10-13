"""11 -- Reward models: turning "A is better than B" into a number.

TLDR: Bradley-Terry. The chance A beats B is softmax over their two scores,
which is the same thing as a sigmoid of the GAP between them. You already
own softmax, so this chapter is mostly free.

Chapter 5 of the RLHF Book.
"""
import math, random
from common import softmax, table, notice

print(__doc__)

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

# ---------------------------------------------------------------- demo 1
print("DEMO 1. Bradley-Terry IS softmax on two options. Same numbers, two spellings.\n")
rows = []
for gap in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
    two_way = softmax([gap, 0.0])[0]      # softmax over scores [gap, 0]
    sig     = sigmoid(gap)                # sigmoid of the gap
    rows.append([f"{gap:.2f}", f"{two_way:.6f}", f"{sig:.6f}",
                 "identical" if abs(two_way - sig) < 1e-12 else "DIFFERENT"])
table(rows, ["A's lead over B", "softmax([gap,0])", "sigmoid(gap)", ""])
print("\n  Every row identical. 'Bradley-Terry', 'the logistic model', 'a two-option")
print("  softmax' and 'sigmoid of the difference' are four names for one formula.")
print("\n  And read the left column as what it means: a lead of 1 means A wins ~73%")
print("  of comparisons. A lead of 4 means ~98%. The GAP is the confidence.")

# ---------------------------------------------------------------- demo 2
print("\n\nDEMO 2. Learn scores from nothing but comparisons.")
print("        Four answers. Humans only ever say which of two they prefer.\n")

NAMES = ["A", "B", "C", "D"]
TRUE  = [2.0, 1.2, 0.3, -0.5]        # the real quality. The model never sees this.

def make_comparisons(n, seed=0, noise=0.9):
    """Humans compare two random answers and pick one. They are noisy."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        i, j = rng.sample(range(4), 2)
        # the better answer usually wins, but not always
        p_i_wins = sigmoid((TRUE[i] - TRUE[j]) / noise)
        out.append((i, j) if rng.random() < p_i_wins else (j, i))   # (winner, loser)
    return out

def fit(comparisons, steps=4000, lr=0.05):
    """Train a reward model: raise the winner's score, lower the loser's.
    The loss for one comparison is  -log sigmoid(r_winner - r_loser)
    -- which is cross-entropy, on a two-option softmax. Nothing new."""
    r = [0.0, 0.0, 0.0, 0.0]
    for _ in range(steps):
        grad = [0.0] * 4
        for w, l in comparisons:
            err = 1.0 - sigmoid(r[w] - r[l])     # how wrong we were on this pair
            grad[w] += err
            grad[l] -= err
        for k in range(4):
            r[k] += lr * grad[k] / len(comparisons)
    return r

for n in (20, 100, 1000):
    learned = fit(make_comparisons(n, seed=7))
    order = sorted(range(4), key=lambda k: -learned[k])
    rows = [[NAMES[k], f"{TRUE[k]:+.2f}", f"{learned[k]:+.2f}"] for k in range(4)]
    print(f"  after {n} human comparisons:")
    table(rows, ["answer", "true quality (hidden)", "learned score"])
    print(f"  ranking it recovered: {' > '.join(NAMES[k] for k in order)}"
          f"   {'CORRECT' if order == [0,1,2,3] else 'wrong'}\n")

# ---------------------------------------------------------------- demo 3
print("\nDEMO 3. The scores have NO absolute meaning. Only gaps are real.\n")
learned = fit(make_comparisons(1000, seed=7))
rows = []
for shift, lab in ((0.0, "as learned"), (10.0, "+10 to every score"), (-7.5, "-7.5 to every score")):
    s = [x + shift for x in learned]
    rows.append([lab, f"{s[0]:+.2f}", f"{s[1]:+.2f}",
                 f"{sigmoid(s[0]-s[1]):.4f}", f"{sigmoid(s[1]-s[2]):.4f}"])
table(rows, ["", "score A", "score B", "P(A beats B)", "P(B beats C)"])
print("\n  Every prediction identical. Shifting all scores changes NOTHING --")
print("  the same shift-invariance as softmax (script 02), for the same reason.")

notice(
  "A reward model is a language model with the final layer swapped: instead of",
  "  100,000 numbers (one per next word) it emits ONE number, a score for the",
  "  whole answer. Everything else about it is unchanged.",
  "Its training loss is cross-entropy on a two-option softmax. You already own",
  "  both halves -- there is genuinely no new maths in this chapter.",
  "DEMO 3 is the one with consequences. A reward of '7.2' means nothing on its",
  "  own. There is no scale, no zero, no units. Only differences carry signal.",
  "  Which is exactly WHY policy gradient subtracts a baseline (script 08):",
  "  'better than usual' is a real signal and 'good' is not -- and now you can",
  "  see that it is not a trick, it is forced. The reward model never had an",
  "  absolute scale to give you.",
  "Lambert's caution, worth keeping: best practices for training reward models",
  "  are not strongly established and depend on the area of application.",
)
