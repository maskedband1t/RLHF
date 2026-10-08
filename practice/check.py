"""Self-checking harness for stubs.py.  Run:  python3 check.py

Reports each function separately, so you can do one at a time.
"""
import math
import stubs

TOL = 1e-6

def close(a, b):
    return abs(a - b) <= TOL

def vclose(a, b):
    return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))

# (group, description, callable, expected, comparator, hint)
T = [
 ("softmax", "softmax([2,1,0]) sums to 1",
  lambda: sum(stubs.softmax([2,1,0])), 1.0, close,
  "Divide each exponentiated score by the total of all of them."),
 ("softmax", "softmax([2,1,0]) values",
  lambda: stubs.softmax([2,1,0]), [0.66524096,0.24472847,0.09003057], vclose,
  "e^2=7.389, e^1=2.718, e^0=1.0, total=11.107. Divide each by the total."),
 ("softmax", "softmax([0,0,0,0]) is uniform",
  lambda: stubs.softmax([0,0,0,0]), [0.25]*4, vclose,
  "Equal scores must give equal shares."),
 ("softmax", "no negatives from negative scores",
  lambda: min(stubs.softmax([-8,-9,-10])), 0.0, lambda a,b: a > 0,
  "Every output must be > 0. Exponentiate before dividing -- e^x is always positive."),
 ("softmax", "shift invariance: [102,101,100] == [2,1,0]",
  lambda: stubs.softmax([102,101,100]), [0.66524096,0.24472847,0.09003057], vclose,
  "Adding a constant to every score must change nothing. If this overflows, "
  "subtract max(scores) from each score first -- it cancels out exactly."),
 ("softmax", "survives huge scores (overflow trap)",
  lambda: stubs.softmax([1002,1001,1000]), [0.66524096,0.24472847,0.09003057], vclose,
  "e^1002 overflows a float. Subtract max(scores) from every score before "
  "exponentiating -- softmax only cares about gaps, so this changes the answer "
  "by exactly nothing and is what every real implementation does."),
 ("softmax", "temperature=2 flattens",
  lambda: stubs.softmax([2,1,0], 2.0), [0.50648039,0.30719589,0.18632372], vclose,
  "Divide the scores by temperature FIRST, then do the usual work."),
 ("softmax", "temperature=0.5 sharpens",
  lambda: stubs.softmax([2,1,0], 0.5), [0.86681333,0.11731043,0.01587624], vclose,
  "T=0.5 doubles every gap, so the winner takes a bigger share."),

 ("entropy", "entropy([1,0,0]) == 0",
  lambda: stubs.entropy([1,0,0]), 0.0, close,
  "Certainty means zero doubt. Make sure p=0 terms are skipped, not computed."),
 ("entropy", "entropy([.5,.5]) == 1 bit",
  lambda: stubs.entropy([0.5,0.5]), 1.0, close,
  "A fair coin is exactly one yes/no question. Are you using log base 2?"),
 ("entropy", "entropy([.25]*4) == 2 bits",
  lambda: stubs.entropy([0.25]*4), 2.0, close,
  "4 equally likely options = 2 halvings = 2 bits."),
 ("entropy", "entropy([.7,.2,.1])",
  lambda: stubs.entropy([0.7,0.2,0.1]), 1.15677965, close,
  "Sum of p * log2(1/p). If you got a negative number, your sign is flipped."),
 ("entropy", "entropy([.9,.07,.03])",
  lambda: stubs.entropy([0.9,0.07,0.03]), 0.55712468, close,
  "More confident than [.7,.2,.1], so this must come out LOWER."),

 ("cross_entropy", "70% on the right answer",
  lambda: stubs.cross_entropy([0.7,0.2,0.1], 0), 0.51457317, close,
  "-log2(0.7). Only the probability of the outcome that happened matters."),
 ("cross_entropy", "10% on the right answer",
  lambda: stubs.cross_entropy([0.7,0.2,0.1], 2), 3.32192809, close,
  "-log2(0.1) = 3.32. Being wrong is expensive, and the cost is not linear."),
 ("cross_entropy", "90% on the right answer",
  lambda: stubs.cross_entropy([0.9,0.07,0.03], 0), 0.15200309, close,
  "-log2(0.9). Nearly no penalty."),
 ("cross_entropy", "uniform over 3",
  lambda: stubs.cross_entropy([1/3,1/3,1/3], 1), 1.5849625, close,
  "log2(3) = 1.585, whichever outcome you pick. Makes sense: no opinion, flat grade."),

 ("kl", "kl(p,p) == 0",
  lambda: stubs.kl([0.7,0.2,0.1],[0.7,0.2,0.1]), 0.0, close,
  "No drift means no penalty."),
 ("kl", "kl([.7,.2,.1],[.3,.3,.4])",
  lambda: stubs.kl([0.7,0.2,0.1],[0.3,0.3,0.4]), 0.53868219, close,
  "sum of p * log2(p/q). Weight by p, the FIRST argument."),
 ("kl", "kl is asymmetric (reverse the args)",
  lambda: stubs.kl([0.3,0.3,0.4],[0.7,0.2,0.1]), 0.60877102, close,
  "If you got 0.539 again, you built something symmetric. Order must matter."),
 ("kl", "kl([.5,.3,.2],[.98,.01,.01])",
  lambda: stubs.kl([0.5,0.3,0.2],[0.98,0.01,0.01]), 1.85102597, close,
  "q put almost nothing where p has real mass. Penalised hard."),

 ("advantage", "advantage([10,9.5,9])",
  lambda: stubs.advantage([10,9.5,9]), [0.5,0.0,-0.5], vclose,
  "Subtract the mean (9.5) from each."),
 ("advantage", "shift invariance: +100 changes nothing",
  lambda: stubs.advantage([110,109.5,109]), [0.5,0.0,-0.5], vclose,
  "This is the whole reason advantage exists. Must match the row above exactly."),
 ("advantage", "all-equal rewards carry no signal",
  lambda: stubs.advantage([5,5,5]), [0.0,0.0,0.0], vclose,
  "If everything scores the same, there is nothing to learn. All zeros."),
]

groups, order = {}, []
for g, desc, fn, exp, cmp_, hint in T:
    if g not in groups:
        groups[g] = []
        order.append(g)
    groups[g].append((desc, fn, exp, cmp_, hint))

passed = total = 0
todo = []
for g in order:
    print(f"\n{g}")
    for desc, fn, exp, cmp_, hint in groups[g]:
        total += 1
        try:
            got = fn()
        except NotImplementedError:
            print(f"  [ ]   {desc}  -- not implemented yet")
            todo.append(g)
            continue
        except Exception as e:
            print(f"  ERR   {desc}")
            print(f"        {type(e).__name__}: {e}")
            print(f"        hint: {hint}")
            continue
        try:
            ok = cmp_(got, exp)
        except Exception:
            ok = False
        if ok:
            passed += 1
            print(f"  PASS  {desc}")
        else:
            shown = [round(x, 6) for x in got] if isinstance(got, (list, tuple)) else got
            want = [round(x, 6) for x in exp] if isinstance(exp, (list, tuple)) else exp
            print(f"  FAIL  {desc}")
            print(f"        got      {shown}")
            print(f"        expected {want}")
            print(f"        hint: {hint}")

print("\n" + "=" * 68)
print(f"  {passed}/{total} passing")
if todo:
    print(f"  not started: {', '.join(sorted(set(todo)))}")
if passed == total:
    print("  All green. Now go read ../lab/common.py and compare -- if yours is")
    print("  different but passes, yours is just as correct. Note the difference.")
print("=" * 68)
