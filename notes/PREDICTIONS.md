# Predictions

The rule: **write the prediction before running anything.** Never edit a prediction after the fact —
append the result underneath. A wrong prediction is the single most valuable thing in this file,
because it marks the exact spot where my intuition is broken.

Scoring: `HIT` / `MISS` / `PARTIAL`. Count them up occasionally. A high hit rate on easy predictions
means I'm not predicting hard enough.

---

### 2026-10-13 · over-optimization, before running `lab/10` — **Claude's miss, not yours**
*Logged in my column, per the working agreement: a number I read you that turns out wrong is a
MISS the same as one of yours.*

**Predicted:** I wrote into the script's own narration that the reader's true quality would
*"peak early, then fall off a cliff"* as optimisation pressure rose.

**Actual:** It climbed the whole way — `−0.20 → +0.57 → +0.90 → +1.20 → +1.31 → +1.43` — and
only dipped at 4096 candidates. No cliff.

**Score:** MISS.

**Learned (and it's a better lesson than the one I meant to teach):** Goodhart needs **a ceiling
on real quality and no ceiling on the gameable feature.** With substance drawn unbounded, hard
optimisation keeps finding genuinely better answers faster than it accumulates padding, so
quality rises. Cap substance at 1.0 — realistic, since there is a limit to how good a reply can
be — and it becomes a catastrophe: peak at 4 candidates, then `−0.26, −1.26, −2.57, −4.03,
−5.74`, while the reward model reports steady improvement the entire time.

So "optimise a proxy hard and it breaks" is not quite right. The sharper claim: **a proxy breaks
when the thing you want saturates and the thing it mismeasures does not.**

---

### 2026-10-12 · entropy & loss for `[3,1,0]`, before running `lab/04` and `lab/05`
*Reasoned aloud on a phone; scribed by Claude, wording checked back.*

**Predicted:** two parts.
- *Entropy:* goes down — "more likely that it's rain, so the model's willingness to get
  surprised goes down."
- *Loss (if rain happens):* goes **up** — "the loss is what happens when you're wrong... you're
  even more certain that it's rain."

**Actual:**
- Entropy `1.2009 → 0.7564` bits. **Down 37%.**
- Loss if rain `0.5881 → 0.2450` bits. **Down.** Loss if cloud or sun: `+1.0997` each.

**Score:** entropy **HIT** · loss **MISS**.

**Learned:** The loss is **the bill for what happened, not the penalty for being wrong.** There
is no notion of "wrong" anywhere in cross-entropy — you look up the belief you placed on the
event reality delivered, and that is the whole calculation. You pay a bill every time, including
when you're right; being right just makes it cheap. So "did the loss go up?" is unanswerable
until someone says what happened.

Second half of it: the *reasoning* was actually correct and landed on the wrong row. "You're
going to be more surprised when it's **not** rain" is exactly the `+1.0997` on cloud and sun.
Right mechanism, wrong question — which is the pattern to watch for (see `CURRICULUM.md`,
pattern 4).

Third: getting more confident is a **bet**. `−0.34` on rain bought at the price of `+1.10` on
each of the others. Forced, because the budget sums to 1 — you can't buy cheaper rain without
selling expensive cloud.

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
