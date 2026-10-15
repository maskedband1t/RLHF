# Our curriculum

Not the book's table of contents. **Ours** — built from evidence of how you actually learn,
collected session by session, and mapped onto Nathan Lambert's RLHF Book so we follow the real
thing without being led by it.

Rewritten as we go. When a session reveals how a concept lands, the unit changes.

**Three rules, on every unit:**
- **Every concept gets a visual.** Not prose describing a visual — an actual diagram or an
  interactive you can move. Walls of markdown are the wrong medium and they do not land.
- **Everything ties back to language models.** A concept that stays abstract hasn't landed.
- **Definitions get restated.** No unit assumes the previous one is still loaded in your head.

**Format, by default:** schoolbook. Visual, worked example, plain words, one thing at a time.
Not a paper, not a reference sheet, not a wall of markdown. Assume the textbook treatment every
time unless asked otherwise.

**What is asked of you, precisely:** *direction and rough size.* Never arithmetic. "Entropy goes
down, and by a lot" is a complete and correct answer. Claude computes; you predict. The six
anchor facts (screen 10 of the visual walkthrough) are the only things worth memorising, and
with them every direction question is doable in your head.

**And one on pacing:** one concept per screen, in dependency order, with a visible position in
the sequence. Jumping between points — however related they are — is the failure mode to avoid.

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
| **Visual** | a diagram or interactive, every unit, no exceptions. Abstract prose is the wrong medium for this material and the one-concept-per-screen walkthrough in `visual/` is where it lives |
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

## Adjacent reading that earned its place

**"Neural Networks Are Already Clean"** — Nick Cammarata, Goodfire. An informal research
essay (stated confidence ~85%) arguing that polysemanticity in early model organisms was
misread: not superposition, but *recursive generalization* — a neuron starting on one thing and
branching along related facets. The canonical cat/car neuron is a **V-shape** neuron; pointed
fires it, round pushes it negative. InceptionV1's last layer went from ~0–5% interpretable
neurons to **99–100%** with only residual connections and LayerNorm added, at identical
parameter count and data — so it was an ordinary learning failure, not a second learning
principle.

Three reasons it belongs in this curriculum, all honest rather than stretched:

1. **A real-world instance of unit 15 (`lab/10`).** The essay's own term for why the microscope
   misled the field is *"feature visualization reward hacking"* — the visualiser optimises to
   raise a neuron's activation, finds the cat route in, and drags whiskers along for the ride.
   An optimiser taking a path to a high score that misses the thing you wanted. Goodhart, in
   the instrument, for five years, with a theory then built on the artifact.
2. **A real-world instance of unit 14b (`lab/12`).** The field was confident and wrong in a
   place nobody had a reason to check. Same shape as a reward model at distance 4.0.
3. **It is this repo's own method at field scale.** The section "A History of Microscope
   Errors" keeps the wrong claim, names where and with whom it was made, and corrects it in
   public — which is `lab/08`'s honest correction and the MISS column of
   `notes/PREDICTIONS.md`, with five years of a research field as the stakes.

**CORRECTION, kept rather than edited away.** The paragraph above originally ended here with
*"a working hypothesis, not a paper... striking anecdotes"* and a verdict that it was far from
established. That judgment was formed after reading **~15,000 of 274,000 characters — about 5%**
— and it materially understated the work. Anurag pushed back with *"I think it's instead us who
haven't ingested this fully,"* which was correct. What a fuller read turned up:

- **A pre-registered falsification test.** 200 random neurons; an agent reads only the written
  explanation, never the neuron, and writes five examples that should fire it and five near
  misses that shouldn't — *then* they measure. New examples fired strongly **76%**, near misses
  **11%**; the explanation won 164, tied 15, **lost 4**. Guess before result, with the losses
  reported. This repo's own method, at scale.
- **"Blind maximum."** One agent explains a neuron; a second, seeing only that explanation,
  writes a sentence that fires it **harder than any example found in 10 billion tokens.**
- **A direct quantitative test against a specific superposition mechanism.** If features hide
  across many neurons' weak activations, you could not remove the **weakest 75%** of activations
  for a 72%→71% accuracy cost — while removing the **strongest 7%** (same total activation mass)
  costs 72%→**4%**. What is lost is fine distinctions the model already confused, arriving from
  semantically related neurons.
- **Circuit racing.** A blank 7.1M model reaches 48–50% alone and **~2%** trained alongside a
  pretrained twin that takes 93–97% of marginal credit. The "head start dominates" mechanism,
  made falsifiable — and with it **a theory of generalization** (broad features win from the
  first step, per Arpit 2017 and coherent gradients), which may matter more than the
  interpretability claim.
- **The definitional asymmetry.** The dominant formulation of superposition — *Toy Models*'
  "more features than dimensions" — is a claim about a count nobody can define or measure.
  Recursive generalization predicts a specific shape in semantic space and in training time.

Revised position: **if it survives independent attack, a leading candidate for the most
important interpretability result so far.** What genuinely remains open — and the author says it
himself, at a stated 85%: no independent replication yet (*"I would update strongly on even one
convincing example"*), and the developmental evidence needs training checkpoints that do not
exist for frontier models.

**The methodological lesson, which is why this correction stays in the file.** The essay reports
that the one prior framing closest to its own view is blocked to AI readers: it opens with a
large base64-embedded image, and every agent web reader they tried hit its length limit before
the argument began. Claude then read 5% of this essay and produced a calibrated-sounding
verdict. That is `lab/12` landing on the reader — the judge confident and wrong in a place
nobody checked — and it is now logged as learning pattern 11: **a skim plus a hedge is not
epistemics.** Read the evidence sections before grading the claim.

---

## Where things live

| file | role |
|---|---|
| `CURRICULUM.md` | this file — the path, and the record of how you learn |
| `visual/maths-chain.html` | **the stepped visual walkthrough.** One concept per screen, 01–09, with a live diagram for each. [Published here](https://claude.ai/artifact/2QNgXmggaJQd6W7Ys9v6ny) |
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
