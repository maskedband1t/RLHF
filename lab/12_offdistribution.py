"""12 -- Why the leash exists, properly: the reward model goes blind as you move.

TLDR: the reward model learned from answers the BASE model wrote. Optimising
means walking away from the base model. So optimisation pressure drives the
policy into exactly the region where its own judge has no information -- and
the judge does not report uncertainty. It keeps returning confident numbers.

The deeper version of script 10. There the proxy had a FIXED blind spot.
Here the blind spot GROWS the harder you push.
"""
import math, random, statistics
from common import table, notice

print(__doc__)

# An answer is described by how far its style sits from the base model (d >= 0)
# and how much real substance it has (s).

def true_quality(d, s):
    """What the reader gets. Moving a little away from the base model genuinely
    helps -- that is the whole point of tuning. Moving a lot does not."""
    return s + 0.8 * d - 0.5 * d * d          # real peak at d = 0.8

def reward_model(d, s, trained_at=0.0, rng=None):
    """The judge. Accurate near where it was trained, and increasingly
    over-optimistic and noisy the further it has to extrapolate."""
    gap = abs(d - trained_at)
    bias = 0.95 * gap                          # systematically flatters the unfamiliar
    noise = rng.gauss(0.0, 0.45 * gap) if rng else 0.0
    return true_quality(d, s) + bias + noise

# ---------------------------------------------------------------- demo 1
print("DEMO 1. The judge's accuracy, by distance from where it was trained.\n")
rng = random.Random(0)
rows = []
for d in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0):
    pairs = [(true_quality(d, rng.gauss(0, 1)), None) for _ in range(2000)]
    ts, ps = [], []
    for _ in range(2000):
        s = rng.gauss(0, 1)
        ts.append(true_quality(d, s))
        ps.append(reward_model(d, s, 0.0, rng))
    err = statistics.mean(p - t for p, t in zip(ps, ts))
    # how often does the judge rank two random answers correctly?
    ok = 0
    for _ in range(4000):
        s1, s2 = rng.gauss(0, 1), rng.gauss(0, 1)
        t1, t2 = true_quality(d, s1), true_quality(d, s2)
        r1, r2 = reward_model(d, s1, 0.0, rng), reward_model(d, s2, 0.0, rng)
        if (t1 > t2) == (r1 > r2):
            ok += 1
    rows.append([f"{d:.1f}", f"{statistics.mean(ts):+.2f}", f"{statistics.mean(ps):+.2f}",
                 f"{err:+.2f}", f"{ok/4000:.1%}"])
table(rows, ["distance from base", "true quality", "judge says", "judge's error",
             "ranks pairs correctly"])
print("\n  At the base model the judge is perfect. Four steps out it overstates")
print("  quality by ~3.8 and its pairwise accuracy has fallen toward a coin flip.")
print("  Nothing in its output says so. There is no uncertainty flag.")

# ---------------------------------------------------------------- demo 2
def best_of_n(n, seed, leash=0.0, trained_at=0.0, reach=4.0):
    rng = random.Random(seed)
    best = chosen = None
    for _ in range(n):
        d = abs(rng.gauss(0.0, reach))
        s = rng.gauss(0.0, 1.0)
        score = reward_model(d, s, trained_at, rng) - leash * (d - trained_at) ** 2
        if best is None or score > best:
            best, chosen = score, (d, s)
    return chosen

print("\n\nDEMO 2. No leash. Turn up the pressure.\n")
rows = []
base_true = None
for n in (1, 4, 16, 64, 256, 1024):
    picks = [best_of_n(n, s) for s in range(600)]
    tq = statistics.mean(true_quality(*p) for p in picks)
    if base_true is None:
        base_true = tq
    rows.append([n,
                 f"{statistics.mean(reward_model(p[0], p[1], 0.0, random.Random(i)) for i, p in enumerate(picks)):+.2f}",
                 f"{tq:+.2f}",
                 f"{statistics.mean(p[0] for p in picks):.2f}",
                 ""])
table(rows, ["candidates", "judge says", "reader gets", "distance travelled"])
print("\n  CAUTION about this table, stated rather than hidden: the n=1 row is NOT")
print("  'no optimisation' -- it is one answer drawn from a wide spread, which is")
print("  simply bad. So quality rising from -5.44 to +0.29 here proves little.")
print("  What it DOES show is the judge and the reader coming apart: the judge")
print("  climbs to +2.90 while the reader stalls near zero, parked at distance")
print("  ~2.7 where demo 1 says the judge overstates by ~2.5 and ranks pairs")
print("  barely better than chance. The clean comparison is demo 3.")

# ---------------------------------------------------------------- demo 3
print("\n\nDEMO 3. The leash. 1024 candidates of pressure throughout.\n")
rows = []
for leash in (0.0, 0.1, 0.3, 0.6, 1.0, 2.0, 5.0):
    picks = [best_of_n(1024, s, leash=leash) for s in range(600)]
    rows.append([f"{leash:.1f}",
                 f"{statistics.mean(true_quality(*p) for p in picks):+.2f}",
                 f"{statistics.mean(p[0] for p in picks):.2f}"])
best = max(rows, key=lambda r: float(r[1]))
for r in rows:
    r.append("<- best" if r is best else "")
table(rows, ["beta (leash)", "reader gets", "distance travelled", ""])
print("\n  The real peak sits at distance 0.8. A tight leash lands you near it.")
print("  No leash sends you far past it, on a judge that is cheering you on.")

# ---------------------------------------------------------------- demo 4
print("\n\nDEMO 4. What practitioners actually do: move a little, then RETRAIN")
print("        the judge on the new region. Repeat.\n")
rows = []
# one big unleashed jump
picks = [best_of_n(1024, s, leash=0.0) for s in range(600)]
rows.append(["one big jump, no leash", f"{statistics.mean(true_quality(*p) for p in picks):+.2f}",
             f"{statistics.mean(p[0] for p in picks):.2f}"])
# iterative: short leashed hops, re-centring the judge each round
at = 0.0
for rnd in (1, 2, 3):
    picks = [best_of_n(1024, s, leash=1.0, trained_at=at, reach=1.5) for s in range(600)]
    at = statistics.mean(p[0] for p in picks)
    rows.append([f"round {rnd}, leashed + retrained",
                 f"{statistics.mean(true_quality(*p) for p in picks):+.2f}", f"{at:.2f}"])
table(rows, ["strategy", "reader gets", "distance travelled"])

notice(
  "This is the honest justification for the KL term, and it is stronger than",
  "  'do not become a different animal'. It is:",
  "",
  "    STAY WHERE YOUR JUDGE STILL KNOWS WHAT IT IS TALKING ABOUT.",
  "",
  "The reward model is only valid near the data it was fitted on. The policy's",
  "  job is to leave there. KL prices the distance, so the optimiser cannot buy",
  "  unlimited confidence from a judge that has stopped being informed.",
  "A reward model reports no uncertainty. Its number looks identical whether it",
  "  is interpolating on familiar ground or extrapolating into nothing.",
  "Which is why real RLHF is a LOOP, not a single optimisation: tune a little,",
  "  collect fresh human comparisons on the NEW model's outputs, retrain the",
  "  reward model, tune again. Demo 4 shows why -- short leashed hops with a",
  "  re-fitted judge beat one long unleashed run.",
  "And this is calibration again (script 09), wearing a different hat: the",
  "  danger is never the judge being unsure. It is the judge being confident",
  "  and wrong, in a place nobody checked.",
)
