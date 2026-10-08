# Predictions

The rule: **write the prediction before running anything.** Never edit a prediction after the fact —
append the result underneath. A wrong prediction is the single most valuable thing in this file,
because it marks the exact spot where my intuition is broken.

Scoring: `HIT` / `MISS` / `PARTIAL`. Count them up occasionally. A high hit rate on easy predictions
means I'm not predicting hard enough.

---

### 2026-10-08 · softmax, before running `lab/02`
**Predicted:** scores `[2,1,0]` → roughly `[0.6, 0.3, 0.1]`.
**Actual:** `[0.665, 0.245, 0.090]`.
**Score:** HIT (shape right, and the top one is higher than I'd have guessed).
**Learned:** `e` grows fast enough that a lead of 1 is worth more than it feels like. A gap of 1 is
already ~58% vs 21% in a 3-way race.

---

### 2026-10-08 · advantage, before running `lab/08`
**Predicted:** subtracting the group mean makes learning *faster*.
**Actual:** Flatly wrong. Without a baseline the learner moved faster (88.7% vs 72.9% on P(A) after
400 steps) — because raw rewards of ~10 act like a 20× larger learning rate. But shift every payout
by +100 and the no-baseline learner picks the **wrong** button on 6 of 12 seeds, while the baselined
one goes 12/12 and returns bit-identical results across all three payout tables.
**Score:** MISS.
**Learned:** The baseline isn't a speed trick. It buys **invariance** (nothing depends on where you
put "zero") and **lower variance** (stdev 8.2% vs 35.3% across seeds with noisy rewards). "Faster"
and "better" are different axes and I conflated them.

---

### TEMPLATE — copy this
### YYYY-MM-DD · <topic>, before running `lab/XX`
**Predicted:**
**Actual:**
**Score:**
**Learned:**
