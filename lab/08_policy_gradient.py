"""08 -- Policy gradient, and what "advantage" is actually for.

TLDR: do the thing, see what you got, make what paid more likely.
Then: subtract the average first. Not for speed -- for not being fooled by
where someone happened to put "zero".
"""
import math, random, statistics
from common import softmax, table, notice

print(__doc__)

ARMS = ["A", "B", "C"]
print("Three buttons. A pays best, but only just. The agent is told nothing.\n")

def sample(p, rng):
    r, c = rng.random(), 0.0
    for i, pi in enumerate(p):
        c += pi
        if r <= c:
            return i
    return len(p) - 1

def train(pay, use_baseline, steps=400, lr=0.02, group=8, seed=7, noise=0.0):
    rng = random.Random(seed)
    theta = [0.0, 0.0, 0.0]          # the raw scores from script 02. These ARE the model.
    for _ in range(steps):
        p = softmax(theta)
        acts = [sample(p, rng) for _ in range(group)]          # try it `group` times
        rews = [pay[a] + (rng.gauss(0, noise) if noise else 0.0) for a in acts]
        # the ONLY line that differs between the two runs:
        baseline = (sum(rews) / group) if use_baseline else 0.0
        grad = [0.0, 0.0, 0.0]
        for a, r in zip(acts, rews):
            advantage = r - baseline
            for i in range(3):
                # d/dtheta_i of log p(a)  ==  (1 if i==a else 0) - p[i]
                grad[i] += advantage * ((1.0 if i == a else 0.0) - p[i])
        for i in range(3):
            theta[i] += lr * grad[i] / group                    # step uphill
    return softmax(theta)

# ---------------------------------------------------------------- demo 1
print("DEMO 1. Three payout tables. The GAPS are identical in all three --")
print("        A beats B by 0.5, B beats C by 0.5. Only 'zero' moved.\n")
rows = []
for pay in ([10.0, 9.5, 9.0], [110.0, 109.5, 109.0], [0.5, 0.0, -0.5]):
    for bl in (False, True):
        p = train(pay, bl)
        rows.append([str(pay), "yes" if bl else "NO",
                     f"{p[0]:.1%}", f"{p[1]:.1%}", f"{p[2]:.1%}"])
table(rows, ["payouts", "baseline?", "P(A) best", "P(B)", "P(C) worst"])
print("\n  The three 'baseline=yes' rows are IDENTICAL. Same problem, same answer.")
print("  The three 'baseline=NO' rows are all over the place. Same problem.")

# ---------------------------------------------------------------- demo 2
print("\n\nDEMO 2. Add 100 to every payout -- which changes nothing about which")
print("        button is best -- and ask which button it settles on. 12 seeds.\n")
rows = []
for bl in (False, True):
    wins = []
    for s in range(12):
        p = train([110.0, 109.5, 109.0], bl, seed=s)
        wins.append(ARMS[p.index(max(p))])
    rows.append(["yes" if bl else "NO", "".join(wins),
                 f"{wins.count('A')}/12"])
table(rows, ["baseline?", "winner, one char per seed", "got it right"])

# ---------------------------------------------------------------- demo 3
print("\n\nDEMO 3. Noisy payouts, small groups. Spread of outcomes over 20 seeds.\n")
rows = []
for bl in (False, True):
    res = [train([10.0, 9.5, 9.0], bl, seed=s, group=2, noise=2.0)[0] for s in range(20)]
    rows.append(["yes" if bl else "NO", f"{statistics.mean(res):.1%}",
                 f"{statistics.stdev(res):.1%}", f"{min(res):.1%}", f"{max(res):.1%}"])
table(rows, ["baseline?", "mean P(A)", "spread (stdev)", "worst seed", "best seed"])

# ---------------------------------------------------------------- the signal
print("\n\nWhy. What the update signal literally says, for payouts 10 / 9.5 / 9:\n")
table([["A", 10.0, "+10.0   push UP hard", "+0.5   push up"],
       ["B",  9.5, " +9.5   push UP hard", " 0.0   leave alone"],
       ["C",  9.0, " +9.0   push UP hard", "-0.5   push DOWN"]],
      ["button", "payout", "no baseline", "with baseline (avg 9.5)"])

notice(
  "Left column: every button gets shoved up. But the budget sums to 1 (script 01),",
  "  so 'everything up' cannot all happen -- what survives is whichever button got",
  "  sampled most this round, and early on that is luck. It is learning from noise.",
  "Right column: the signal is SIGNED, and it is the same signal no matter whether",
  "  payouts are 10/9.5/9 or 110/109.5/109. It only ever sees DIFFERENCES.",
  "",
  "HONEST CORRECTION, kept on purpose. My first version of this script claimed the",
  "  baseline makes learning faster, and the output said the opposite: raw rewards",
  "  of ~10 act like a 20x bigger learning rate, so they move faster -- sometimes",
  "  confidently to the WRONG button (demo 2). The baseline is not a speed trick.",
  "  It buys invariance (demo 1) and lower variance (demo 3). Run it, don't trust me.",
  "",
  "Names for that subtraction: 'baseline', 'advantage', A = r - V(s). One idea.",
  "GRPO is the code above: sample a GROUP for one prompt, use the group's own mean",
  "  as the baseline. That is why it needs no value network. That is the whole trick.",
  "REINFORCE is one line:  theta += lr * advantage * grad log p(action).",
  "  Read it as: 'raise the log-prob of what I did, scaled by how well it went.'",
)
