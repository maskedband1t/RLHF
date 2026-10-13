# Our curriculum

Not the book's table of contents. **Ours** — built from evidence of how you actually learn,
collected session by session, and mapped onto Nathan Lambert's RLHF Book so we follow the real
thing without being led by it.

Rewritten as we go. When a session reveals how a concept lands, the unit changes.

**Two rules, on every unit:**
- **Everything ties back to language models.** A concept that stays abstract hasn't landed.
- **Definitions get restated.** No unit assumes the previous one is still loaded in your head.

---

## How you learn — observed, not assumed

Each item is backed by something that actually happened. These drive the unit template.

1. **Mechanical grounding first.** You asked "where does 11.11 come from" before accepting
   softmax. What worked was `7.39 + 2.72 + 1.00`, not "the normalising constant."
   → *Every number traceable to arithmetic you can watch happen.*

2. **Countable beats formal.** "100 days, 50 rainy, 50 sunny" landed KL. `−Σ pᵢ log qᵢ` did not.
   → *Concrete example first. Formula second.*

3. **Terminology collisions are the wall — not the concepts.** Every miss so far was a word
   doing two jobs, never a failure to grasp an idea:
   - thought "loss" and "cross-entropy" were two objects (they're one number)
   - thought "loss" meant *penalty for being wrong* (it's *the bill for what happened*)
   → *Every unit needs "what this word does NOT mean."*

4. **Mechanism intuition is good; question-tracking slips.** On `[3,1,0]` you correctly derived
   "you'll be more surprised when it's *not* rain" — right — then answered the rain row with it.
   → *Drills state loudly which event is being asked about.*

5. **You generate structural questions unprompted.** "Does entropy stay the same no matter what
   actually happened?" came from you, off a table shown for another purpose. Highest signal
   that happens in these sessions.
   → *Capture them. They become units.*

6. **Reasoning aloud works.** Rambling through `[3,1,0]` produced a real, locatable miss.
   → *Predictions are spoken. Claude is the scribe.*

7. **Two modes.** 🎧 phone/voice = learn and argue. 💻 computer = build and verify.
   → *Every unit carries both, and neither blocks the other.*

---

## The unit template

| field | why |
|---|---|
| **Recap** | definitions restated in full (rule 2) |
| **In a language model** | what this *is* inside an LLM — never skipped (rule 1) |
| **Countable example** | real numbers, holdable in your head (pattern 2) |
| **Formula, decoded** | symbol by symbol, traced to the example (pattern 1) |
| **What it does NOT mean** | the terminology traps (pattern 3) |
| **Traps you actually hit** | filled from real misses — personalised |
| **🎧 Predict aloud** | spoken prompt, event stated explicitly (patterns 4, 6) |
| **💻 Build** | the keyboard task |
| **Done when** | you can say it cold, and the harness agrees |

---

## Part 1 — The maths vocabulary, as it appears in a language model

```
budget (01) ──► softmax (02) ──► temperature (03)
                   │
                   ├──► entropy (04) ──► cross-entropy (05) ──► KL (06)
                   │                          │
                   └──► log-probs (07) ───────┴──► policy gradient (08) ──► calibration (09)
```

| unit | in a language model, this *is* | status |
|---|---|---|
| **01 budget** | the model's output at every position: one belief number per token in a ~100,000-word vocabulary, summing to 1. Not a metaphor — literally the `probs` array. | ✅ |
| **02 softmax** | the **last operation in the network.** Final layer emits raw scores (logits), softmax turns them into next-token probabilities. | ✅ |
| **03 temperature** | the `temperature` knob in every API call. Same arithmetic: divide logits by `T` before softmax. | 🟡 |
| **04 entropy** | **how unsure the model is about the next word.** "United States of ___" → near-zero entropy, it's locked in. "My favourite ___" → high entropy, wide open. After RL, falling entropy = repetitive, samey output. | ✅ |
| **05 cross-entropy** | **the pretraining loss itself.** For every position in every document: raise belief on the word that actually came next. Perplexity is this in friendlier units. | ✅ |
| **06 KL** | the **leash** in RLHF — how far the tuned model drifted from the base model. Also the thing DPO is algebraically derived from. | ✅ |
| **07 log-probs** | the `logprobs` field the API returns. A sequence's probability is the **sum** of its token log-probs, because a 500-token answer's raw probability underflows to `0.0` and kills every gradient. | 🔴 |
| **08 policy gradient** | **how RLHF actually updates weights.** Sample an answer, score it, raise the log-probability of the tokens in the answers that scored above average. GRPO = sample a group per prompt, use the group's mean as the baseline. | 🟡 |
| **09 calibration** | does the model's 90% actually mean right-9-times-in-10? The hallucination question, and **the hand-off question in your RLCD work.** | 🔴 |

### What's settled and what's open

| unit | settled | open |
|---|---|---|
| 01 | 100 belief points, spend them all; zero-sum | — |
| 02 | `Σe^x` is the denominator (`Z`, the partition function) and its size is meaningless; **adding** to logits is a no-op; **multiplying** *is* temperature; shift-invariance is what licenses max-subtraction against overflow | — |
| 03 | scaling logits ≡ dividing by `T` | never driven directly; `T→0` untested by you |
| 04 | average surprise weighted by your own beliefs; ceiling is `log₂(n)`; driven by the **middle** of the distribution (peak at `p=1/e`); takes **one** argument so it cannot see outcomes | entropy collapse seen as a story, not numbers |
| 05 | the bill for what happened; one-hot truth collapses the sum to one lookup; **it IS the loss**; `H(p,p)` = entropy; `H(p,q) = H(p) + KL(p‖q)` | batch-averaging never run |
| 06 | bill = **floor + waste**, KL is the waste; `KL ≥ 0`; asymmetric because the **first slot says which world is real**; `π_θ` sits in the first slot of the RLHF leash | which direction to *minimise* (mode-seeking vs mass-covering) still hand-wavy |
| 07 | — | why logs at all; `1e-651 → 0.0` |
| 08 | advantage is signed; baseline buys invariance + low variance, **not speed** (logged MISS) | never re-derived by you; GRPO only as a story |
| 09 | — | entropy ↓ while loss ↑ = confidently wrong |

**The honest gap:** `practice/stubs.py` is `0/24`. All of the above is *recognition*. Production
is untested, and that's the only unfaked measurement available.

---

## Part 2 — The book, mapped

Sourced from the live site this session. **✅ = chapter number confirmed from a real URL or
lecture slug. ❓ = inferred, needs checking.**

| ch | title | lecture | our prerequisite |
|---|---|---|---|
| ✅ 1 | Introduction | **L1** (ch 1–3) | — |
| ❓ 2 | A Tiny History of RLHF | L1 | — |
| ✅ 3 | Training Overview | L1 | **06 KL** — the KL regularization penalty lives here |
| ✅ 4 | Instruction Fine-Tuning (SFT) | **L2** (ch 4, 5, 9) | **05 cross-entropy** — SFT *is* next-token training on curated answers |
| ✅ 5 | Reward Modeling | L2 | **02 softmax** — Bradley-Terry is a two-option softmax |
| ✅ 6 | Reinforcement Learning / Policy Gradients — PPO, REINFORCE, RLOO, GRPO | **L3** (ch 6 pt 1, the maths) · **L4** (ch 6 pt 2, implementation) | **08 policy gradient** |
| ❓ 7 | Reasoning & Inference-Time Scaling | ? | 03 temperature |
| ✅ 8 | Direct-Alignment Algorithms (DPO) | **L6** | **06 KL** — DPO is derived from the KL-constrained objective |
| ✅ 9 | Rejection Sampling | L2 | 01 budget |
| ✅ 10 | The Nature of Preferences | ? | — |
| ❓ 11 | Preference Data | ? | — |
| ✅ 12 | Synthetic Data & Modern Post-training (Constitutional AI is §12.5) | **L7** | — |
| ✅ 14 | Over-Optimization — Goodhart, reward hacking, sycophancy | ? | **06 KL** — this is what the leash prevents |
| ✅ | Appendix B — Style | ? | — |

Course also has **Q&A sessions** and runs past L7 (L10, L11 referenced). Lecture 5's topic
didn't surface — check [the course page](https://rlhfbook.com/course).

### The leverage this mapping buys

- **Ch 5 Reward Modeling should be nearly free.** Bradley-Terry is softmax on two options, and
  you own softmax completely — including why only the *gap* matters.
- **Ch 3 and Ch 8 and Ch 14 all rest on KL**, which you settled this session. Three chapters,
  one concept.
- **Ch 4 SFT is just cross-entropy** on curated data. Also settled.
- **L3 is the one to respect** — chapter 6 part 1 is the RL maths, and `lab/08` plus unit 08 is
  the preparation for exactly that lecture.

So the efficient route into the book is **L1 → L2 → L3**, because units 02, 05, 06 already
cover their prerequisites. The one real gap on that path is unit 08.

---

## Where things live

| file | role |
|---|---|
| `CURRICULUM.md` | this file — the path, and the record of how you learn |
| `00-MATH-TLDR.md` | one-page vocabulary reference |
| `lab/*.py` | one runnable script per concept; the referee |
| `practice/stubs.py` | production test — write the five functions cold |
| `notes/PREDICTIONS.md` | prediction before result; the misses are the value |
| `notes/LECTURES.md` | per-lecture term dumps, in your words |
| `reason/00-HOW-WE-WORK.md` | the loop and the division of labour |
| `SESSION-LOG.md` | handoff between sessions |

---

## Next, by mode

**🎧 Phone / voice**
- **Lecture 1** (ch 1–3). Your prerequisites are met. Guess two sentences aloud before play.
- Or unit **09 calibration** — closest to your RLCD hand-off work.

**💻 Computer**
- `practice/stubs.py` → 24/24. `softmax` and `entropy` you can write from this session alone;
  `cross_entropy` is the one-line collapse; `kl` is `cross_entropy − entropy`;
  `advantage` is `r − mean(r)`.
- Spot-check me: run `lab/04`, `05`, `06` and confirm the numbers I read you.
