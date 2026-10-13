# Session log

The handoff file. I don't carry memory between sessions — this is how I pick up where we
left off. Newest entry at the top. Keep it current or the continuity breaks.

---

## 2026-10-12 — entropy, cross-entropy, KL; curriculum built

**Covered:** softmax denominator (`Σe^x` = `Z`, and its magnitude is meaningless); adding vs
multiplying logits (no-op vs temperature, and why that licenses max-subtraction); entropy as
average surprise, its `log₂(n)` ceiling, the `p=1/e` peak, and that it takes **one** argument so
it cannot see outcomes; cross-entropy **is** the loss, `H(p,p)` = entropy, and
`H(p,q) = H(p) + KL(p‖q)` verified numerically; KL as **floor + waste**, its asymmetry via the
"first slot says which world is real" rule, and the RLHF leash.

**Settled:**
- **Verified the real book and lecture structure** via WebSearch. Direct fetch of `rlhfbook.com`
  is still blocked (DNS/proxy), but **search reaches it.** The old guessed lecture list in
  `README.md` was wrong from L3 on — corrected, with the correction kept in place.
- Working mode is now explicit: 🎧 phone/voice = learn and argue, 💻 computer = build and verify.
  Claude is scribe in voice mode. Written into `reason/00-HOW-WE-WORK.md`.
- Two standing rules: tie everything to language models; restate definitions every time.

**Built:**
- **`CURRICULUM.md`** — our own path. Carries the observed-learning-patterns list (each item
  backed by a real moment), the eight-field unit template, Part 1 status per unit with its
  language-model meaning, and Part 2 mapped to real confirmed chapter numbers.
- `lab/06_kl.py` — added DEMO 0: the Alice/Bob floor-and-waste day-count that actually landed,
  plus the same arithmetic on a next-token slot.

**Logged in PREDICTIONS.md:** entropy **HIT** / loss **MISS** on `[3,1,0]`. The misconception:
loss was thought to be *the penalty for being wrong*; it's *the bill for what happened*. Notably
the reasoning was right and landed on the wrong row — "more surprised when it's not rain" is
exactly the `+1.10` on cloud and sun.

**Open:**
- `practice/stubs.py` still **0/24**. Everything so far is recognition, not production.
- No lecture watched. `reason/` holds only the template. `GLOSSARY.md` RLHF half blank.
- Lecture 5's topic unidentified; chapters 2, 7, 11 numbers inferred not confirmed.
- Units 07 (log-probs) and 09 (calibration) untouched. 09 is closest to the RLCD work.
- **Nothing pushed.** This container is ephemeral — the work above is committed locally only.

**Next:** Lecture 1 (chapters 1–3) — prerequisites are met, and it's a phone task. On the
keyboard: `practice/stubs.py` to 24/24, which is the only unfaked measurement available.

---

## 2026-10-08 — setup

**Covered:** the maths vocabulary, end to end. Distributions as a belief budget, softmax,
temperature, entropy, cross-entropy, KL, log-probs, policy gradient, calibration.
Built `00-MATH-TLDR.md` plus nine runnable scripts in `lab/`, and the practice harness.

**Settled:** the course is Nathan Lambert's RLHF Book (`rlhfbook.com`) and its YouTube
lectures. Earlier confusion with "RLCF" is resolved — that's an unrelated Apple method
(Reinforcement Learning from *Checklist* Feedback) with no handbook.

**Logged in PREDICTIONS.md:** one MISS — I claimed the policy-gradient baseline makes
learning faster; the script showed it buys invariance and lower variance instead, and that
the no-baseline version can learn the *wrong* action 6 times in 12 seeds.

**Open:**
- Lecture checklist in `README.md` is from search results, not the source —
  `rlhfbook.com` is blocked by this environment's network policy. Needs 30 seconds in a
  browser to correct.
- No lecture watched yet. Nothing in `reason/` but the template.
- `practice/stubs.py` not started — 0/24.
- `GLOSSARY.md` RLHF section is deliberately blank.

**Next:** the ten-minute drill at the bottom of `00-MATH-TLDR.md`, then
`practice/check.py` to green, then lecture 1.

---

## TEMPLATE — copy this
## YYYY-MM-DD — <what we worked on>

**Covered:**
**Settled:**
**Logged in PREDICTIONS.md:**
**Open:**
**Next:**
