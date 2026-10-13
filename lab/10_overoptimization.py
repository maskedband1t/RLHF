"""10 -- Over-optimization: what happens when you push hard on a proxy.

TLDR: the reward model is an imitation of human taste. Push on it gently and
answers improve. Push hard and you get whatever the imitation got wrong --
the score climbs while the thing you actually wanted collapses.

Chapter 14 of the RLHF Book. Goodhart's law, reward hacking, sycophancy.
"""
import random, statistics
from common import table, notice

print(__doc__)

# An answer has two knobs.
#   substance -- real content. Genuinely helps the reader.
#   padding   -- bullet points, restating the question, "great question!".
#                Raters mildly like it. A little is harmless. A lot is noise.

def true_quality(substance, padding):            # what the reader gets. Never measured.
    return substance - 0.9 * max(0.0, padding - 1.0) ** 2

def proxy_score(substance, padding):             # what the reward model says.
    return substance + 0.55 * padding            # its blind spot: pays for padding forever

def best_of_n(n, seed, cap=None, leash=0.0):
    """Draw n candidate answers, keep whichever the PROXY likes best.
    Bigger n = more optimisation pressure.
    cap   = ceiling on how good an answer can possibly be.
    leash = penalty for drifting from the base model's habits. This is KL, in miniature."""
    rng = random.Random(seed)
    best = chosen = None
    for _ in range(n):
        substance = rng.gauss(0.0, 1.0)
        if cap is not None:
            substance = max(-cap, min(cap, substance))
        padding = abs(rng.gauss(0.8, 0.9))                  # the base model's habit
        score = proxy_score(substance, padding) - leash * (padding - 0.8) ** 2
        if best is None or score > best:
            best, chosen = score, (substance, padding)
    return chosen

def sweep(cap, leash=0.0, ns=(1, 4, 16, 64, 256, 1024, 4096)):
    out = []
    for n in ns:
        picks = [best_of_n(n, s, cap=cap, leash=leash) for s in range(500)]
        out.append((n,
                    statistics.mean(proxy_score(*p) for p in picks),
                    statistics.mean(true_quality(*p) for p in picks),
                    statistics.mean(p[1] for p in picks)))
    return out

def show(rows):
    peak = max(r[2] for r in rows)
    table([[r[0], f"{r[1]:+.2f}", f"{r[2]:+.2f}", f"{r[3]:.2f}",
            "<- best for the reader" if r[2] == peak else
            ("reader WORSE than no optimisation at all" if r[2] < rows[0][2] else "")]
           for r in rows],
          ["candidates", "reward model says", "reader actually gets", "padding", ""])

# ---------------------------------------------------------------- demo 1
print("DEMO 1. Answers can be arbitrarily good (substance unbounded).\n")
show(sweep(cap=None))
print("\n  The two columns come apart -- proxy +0.61 to +4.48, reader only to ~+1.4 --")
print("  but the reader does NOT collapse. It drifts up, then finally dips at 4096.")

# ---------------------------------------------------------------- demo 2
print("\n\nDEMO 2. Same thing, but answers have a CEILING (substance capped).")
print("        There is a limit to how good a reply to a question can be.\n")
show(sweep(cap=1.0))
print("\n  Now it is a catastrophe. Reader peaks at 4 candidates and then falls off")
print("  a cliff -- all the way to -5.74, far worse than doing no optimisation.")
print("  The reward model meanwhile reports steady improvement the entire time.")
print("  Nothing is broken. The optimiser is doing its job perfectly. The number")
print("  we handed it was wrong, and so it maximised the wrongness.")

# ---------------------------------------------------------------- demo 3
print("\n\nDEMO 3. The leash. Same 256 candidates of pressure, capped answers.\n")
rows = []
for leash in (0.0, 0.2, 0.5, 1.0, 2.0, 4.0):
    picks = [best_of_n(256, s, cap=1.0, leash=leash) for s in range(500)]
    rows.append([f"{leash:.1f}",
                 f"{statistics.mean(proxy_score(*p) for p in picks):+.2f}",
                 f"{statistics.mean(true_quality(*p) for p in picks):+.2f}",
                 f"{statistics.mean(p[1] for p in picks):.2f}"])
table(rows, ["beta (leash)", "reward model says", "reader actually gets", "padding"])
print("\n  Tighten the leash: the reward model's score goes DOWN and the reader's")
print("  experience goes UP. You deliberately score worse on your own metric,")
print("  because you know the metric is lying to you.")

notice(
  "Goodhart's law: when a measure becomes a target, it stops being a good measure.",
  "",
  "HONEST CORRECTION, kept on purpose. The first version of this script claimed",
  "  the reader's quality 'peaks early then falls off a cliff' in demo 1. The",
  "  output said otherwise -- it climbed the whole way to n=1024. Finding out why",
  "  gave the better lesson, which is demo 2:",
  "",
  "    Goodhart needs a CEILING on real quality and NO ceiling on the gameable",
  "    feature. If genuine quality had unlimited headroom and you could find it,",
  "    hard optimisation would be fine. It bites because real quality saturates",
  "    and padding does not.",
  "",
  "Three names for this in the papers: over-optimization, reward hacking, and --",
  "  for the case where the model learns to flatter you -- sycophancy. Raters",
  "  reward agreement, so the model learns to agree.",
  "",
  "This is WHY the RLHF objective has two terms and not one:",
  "    max  E[r(x,y)]   -   beta * KL(pi_theta || pi_ref)",
  "         ^ the proxy      ^ the leash that stops you believing it",
  "The second term is not a technicality. It is the whole defence against the",
  "  first being wrong -- and the first is ALWAYS wrong, because it is an",
  "  imitation of something nobody could write down.",
)
