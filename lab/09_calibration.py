"""09 -- Calibration: the one that connects all of this to your own RLCD repo.

TLDR: of all the times you said 70%, did it happen 70% of the time?
Accuracy asks "were you right". Calibration asks "did you know how right you were".
"""
import random
from common import table, notice

print(__doc__)

def simulate(name, bias, n=4000, seed=3):
    """bias > 0 means the forecaster claims more confidence than it earns."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        true_p = rng.random()                   # reality's actual chance
        stated = min(0.999, max(0.001, true_p + bias))   # what it claims
        happened = rng.random() < true_p
        rows.append((stated, happened))
    return name, rows

def report(name, rows):
    print(f"\n{name}\n")
    buckets, ece, n = {}, 0.0, len(rows)
    for stated, happened in rows:
        b = min(9, int(stated * 10))
        buckets.setdefault(b, []).append((stated, happened))
    out = []
    for b in sorted(buckets):
        grp = buckets[b]
        said = sum(s for s, _ in grp) / len(grp)
        got  = sum(1 for _, h in grp if h) / len(grp)
        ece += (len(grp) / n) * abs(said - got)
        flag = "" if abs(said - got) < 0.05 else ("OVERCONFIDENT" if said > got else "under")
        out.append([f"{b*10}-{b*10+10}%", len(grp), f"{said:.3f}", f"{got:.3f}",
                    f"{said-got:+.3f}", flag])
    table(out, ["said", "n", "avg claimed", "actual hit rate", "over (claimed-actual)", ""])
    acc = sum(1 for s, h in rows if (s > 0.5) == h) / n
    print(f"\n  accuracy = {acc:.1%}   |   EXPECTED CALIBRATION ERROR = {ece:.3f}")
    return ece

e1 = report(*simulate("HONEST forecaster", 0.0))
e2 = report(*simulate("OVERCONFIDENT forecaster (+0.15 on every claim)", 0.15))

print("\n" + "=" * 72)
print(f"  honest ECE {e1:.3f}   vs   overconfident ECE {e2:.3f}")
print("=" * 72)

notice(
  "ECE = average gap between what you claimed and what you got, weighted by how",
  "  often you claimed it. One number. Lower is better. Nothing more to it.",
  "A model can be ACCURATE and badly CALIBRATED: right often, but saying 99%",
  "  when it should say 70%. That model is useless for a hand-off decision,",
  "  because its threshold means nothing.",
  "This is exactly the axis your RLCD repo measures: README cites the 421M copy at",
  "  ECE .054 and -.007 over, against a dense 27B at ECE .153 and +.135 over.",
  "  '+.135 over' = says 13.5 points more confidence than it earns. Now that reads.",
)
