"""06 -- KL divergence: the leash.

TLDR: how far has my new distribution drifted from the old one? In RLHF it is
the penalty that stops your tuned model turning into a different animal.
"""
import math
from common import kl_bits, entropy_bits, table, notice

print(__doc__)


# ---------------------------------------------------------------- demo 0
# The framing that actually landed in conversation: your bill splits into a
# floor you cannot beat plus waste that is your own fault. KL is the waste.
print("DEMO 0. KL = the WASTE on your bill. Count the days.\n")

def bill_per_day(belief, world):
    """Average shock, in bits, for someone believing `belief` living in `world`."""
    return sum(pr * math.log2(1 / ps) for pr, ps in zip(world, belief))

ALICE = [0.50, 0.50]      # hedges: 50% rain, 50% sun
BOB   = [0.01, 0.99]      # confident: sun is basically certain

for wname, world in [("coin-flip weather (50 rain / 50 sun)", [0.50, 0.50]),
                     ("desert (1 rain / 99 sun)",             [0.01, 0.99])]:
    print(f"  World: {wname}")
    floor = entropy_bits(world)
    rows = []
    for pname, belief in [("Alice  [.50,.50]", ALICE), ("Bob    [.01,.99]", BOB)]:
        paid = bill_per_day(belief, world)
        rows.append([pname, f"{paid:.3f}", f"{floor:.3f}",
                     f"{paid - floor:.3f}", f"{kl_bits(world, belief):.3f}"])
    table(rows, ["who", "pays/day", "floor", "paid - floor", "KL(world||them)"])
    print()

print("  'paid - floor' and 'KL(world||them)' are the SAME COLUMN. That is the definition:")
print("    your bill  =  the floor nobody can beat  +  the waste that is your fault")
print("    and KL is the waste. Which is why KL can never go below 0.\n")
print("  Swap the worlds and the same two belief sets cost very differently:")
print(f"    Bob wrong in a random world:      {kl_bits([.5,.5],   BOB):.3f} bits wasted")
print(f"    Alice wrong in a predictable one:  {kl_bits([.01,.99], ALICE):.3f} bits wasted")
print("  Overconfident in a random world is far worse than over-hedged in a sure one.")
print("  Bob gets blindsided by rain 50 times. Alice just carries a pointless umbrella.")
print("  THAT is why KL is not symmetric -- the FIRST slot says which world is real.\n")

print("  Same thing, in a language model. Two models, one next-token slot after")
print('  "The capital of France is ___":\n')
REAL  = [0.90, 0.07, 0.03]          # how often each word truly comes next
names = ["Paris", "Lyon", "Nice"]
for mname, m in [("good model  [.88,.08,.04]", [0.88, 0.08, 0.04]),
                 ("cocky model [.999,.0005,.0005]", [0.999, 0.0005, 0.0005])]:
    print(f"    {mname:<34} wastes {kl_bits(REAL, m):.3f} bits per token")
print(f"    {'(floor set by language itself)':<34}        {entropy_bits(REAL):.3f} bits per token")
print("  The cocky model is nearly right and still wastes more, because the rare")
print("  words it wrote off DO occur -- and each time, it pays through the nose.")
print("  That is hallucination's cost function, and it is the same arithmetic.\n")

# ---------------------------------------------------------------- demo 1
print("DEMO 1. The leash: how far has the tuned model drifted from the base model?\n")
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
