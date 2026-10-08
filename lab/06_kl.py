"""06 -- KL divergence: the leash.

TLDR: how far has my new distribution drifted from the old one? In RLHF it is
the penalty that stops your tuned model turning into a different animal.
"""
from common import kl_bits, table, notice

print(__doc__)

base = [0.70, 0.20, 0.10]
print(f"Base model's belief:  {base}\n")
rows = []
for name, tuned in [("barely moved",  [0.72, 0.19, 0.09]),
                    ("moved a bit",   [0.80, 0.14, 0.06]),
                    ("moved a lot",   [0.95, 0.04, 0.01]),
                    ("went all-in",   [0.999, 0.0005, 0.0005]),
                    ("flipped",       [0.10, 0.20, 0.70])]:
    rows.append([name, str([round(x, 3) for x in tuned]), f"{kl_bits(base, tuned):.3f}"])
table(rows, ["after tuning", "new belief", "KL from base (bits)"])

print("\n\nIt is NOT symmetric. Order matters.\n")
pairs = [("p=[.7,.2,.1]  q=[.3,.3,.4]", [0.70, 0.20, 0.10], [0.30, 0.30, 0.40]),
         ("p=[.5,.3,.2]  q=[.98,.01,.01]", [0.50, 0.30, 0.20], [0.98, 0.01, 0.01]),
         ("p=[.7,.2,.1]  q=[.1,.2,.7]", [0.70, 0.20, 0.10], [0.10, 0.20, 0.70])]
table([[lab, f"{kl_bits(a, b):.3f}", f"{kl_bits(b, a):.3f}",
        "equal -- COINCIDENCE, see below" if abs(kl_bits(a,b)-kl_bits(b,a)) < 1e-9 else ""]
       for lab, a, b in pairs],
      ["pair", "KL(p||q)", "KL(q||p)", ""])

print("\n  Row 3 comes out equal because q is p reversed -- a mirror pair is a special")
print("  case, not the rule. (My first draft of this script used only that pair and")
print("  'proved' asymmetry with two identical numbers. Check your examples.)")
print("\n  Row 2 is the one to stare at: q is confident where p is spread out.")
print(f"  KL(p||q) = {kl_bits(pairs[1][1], pairs[1][2]):.3f} -- p is badly surprised: q put almost nothing")
print("    on outcomes p thinks are quite likely. Penalised hard.")
print(f"  KL(q||p) = {kl_bits(pairs[1][2], pairs[1][1]):.3f} -- q is barely bothered: p still covers q\'s")
print("    one favourite outcome, just less keenly.")
print("  So the direction you minimise decides the behaviour you get: one direction")
print("  makes you COVER everything, the other lets you pick one mode and commit.")

print("\n\nWhat the leash is for:\n")
print("  Reward model says: answers with bullet points score high.")
print("  Without a KL penalty, your model discovers this and emits bullet points")
print("  forever, for every question. Reward: excellent. Model: useless.")
print("  That is reward hacking. The KL term prices it: every bit of drift costs you.")

notice(
  "In RLHF the objective is roughly:  maximise  reward - beta * KL(tuned || base).",
  "  beta is the leash length. Small beta = roam free, risk hacking the reward.",
  "  Large beta = stay near the base model, learn almost nothing.",
  "KL is the same arithmetic as cross-entropy minus entropy. Not a new idea --",
  "  the SAME idea pointed at two distributions you both chose, instead of at reality.",
)
