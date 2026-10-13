# RLHF

Working through **Nathan Lambert's RLHF Book** (`rlhfbook.com`) and its YouTube lecture
course — slowly, in our own words, with everything checked against output.

This is not a notes repo. Notes let you reread something and feel like you understand it.
The rule here is:

> **Every concept gets a script that prints numbers, and the guess goes down before the
> script runs.**

Borrowed from my own [RLCD](https://github.com/maskedband1t/RLCD) methodology
(`notebook/CLAIMS.md`, prediction-before-result). It works on research; it works on
learning for the same reason — it's the only way to catch yourself *feeling* like you
understand something.

## Start here

| | |
|---|---|
| **[CURRICULUM.md](CURRICULUM.md)** | Our own path through the book, built from how you actually learn, mapped to the real chapters and lectures. **Start here.** |
| **[00-MATH-TLDR.md](00-MATH-TLDR.md)** | The entire maths vocabulary on one page, plus a notation decoder ring. **Read this first.** |
| **[reason/00-HOW-WE-WORK.md](reason/00-HOW-WE-WORK.md)** | The loop we run per chapter, and the division of labour. |
| **[practice/](practice/)** | Implement the five core functions yourself. Self-checking — 24 tests, hints on every failure. |
| [GLOSSARY.md](GLOSSARY.md) | Term → one line. Maths half filled; RLHF half blank, to fill from the lectures. |
| [lab/](lab/) | Nine runnable scripts, one per concept. No dependencies. |
| [SESSION-LOG.md](SESSION-LOG.md) | State of play. The handoff between sessions. |
| [QUESTIONS.md](QUESTIONS.md) | Dump half-formed confusions here. |
| [notes/PREDICTIONS.md](notes/PREDICTIONS.md) | Prediction before result. The actual method. |
| [reason/](reason/) | One worked file per chapter, from the template. |

## Run it

Needs nothing but `python3`. No pip, no venv, no numpy.

```bash
./run_all.sh                      # all nine concept scripts
cd practice && python3 check.py   # the practice harness: 0/24 to start
```

Each `lab/` script ends in `NOTICE:` lines — the takeaways — and is short enough to read
end to end.

## Learn it in this order

Not flat. It's a chain, and three links carry most of the weight:

```
budget (01) ──► softmax (02) ──► temperature (03)
                   │
                   ├──► entropy (04) ──► cross-entropy (05) ──► KL (06)
                   │                          │
                   └──► log-probs (07) ───────┴──► policy gradient (08) ──► calibration (09)
```

**If you only get three:** the budget, softmax, cross-entropy. With those you can read an
RLHF paper's loss function. Entropy and KL next. Everything else is assembly.

## The referee is the output, not either of us

`lab/08_policy_gradient.py` is in its current shape because the first explanation of
`advantage` written here was flatly contradicted by the script's own numbers — the
correction is kept in the file, and logged as a MISS in `notes/PREDICTIONS.md`. That's the
standard. Claims get checked against output, and wrong ones get kept with the correction
rather than quietly edited.

## Lecture progress

Verified against the live course site (2026-10-12). Chapter numbers in **bold** are confirmed
from real URLs or lecture slugs; `?` means still unchecked.

- [ ] **Lecture 1** — Overview (chapters **1–3**)
- [ ] **Lecture 2** — IFT, Reward Models & Rejection Sampling (chapters **4, 5, 9**)
- [ ] **Lecture 3** — Reinforcement Learning: motivation & maths (chapter **6** part 1)
- [ ] **Lecture 4** — RL Implementation & Practice (chapter **6** part 2)
- [ ] **Lecture 5** — topic not yet identified
- [ ] **Lecture 6** — Direct Preference Optimization (chapter **8**)
- [ ] **Lecture 7** — Synthetic Data & Modern Post-training (chapter **12**)
- [ ] Lectures 8+ — course runs past L7; Q&A sessions also published
- [x] Maths vocabulary — `00-MATH-TLDR.md`, all nine `lab/` scripts, `practice/` harness

**Correction kept on purpose.** The earlier version of this list was guessed from search
snippets and was wrong from Lecture 3 on: it had L3 as reward models, L4 as policy gradient and
L5 as implementation details. In fact reward models are in **L2** (bundled with IFT and
rejection sampling), and the RL material is split across **L3** (maths) and **L4**
(implementation). **Correction to the correction (same day):** a launch write-up of the course does mention a
planned prerequisite ML-foundations video, so the old list was right about that and this file
was wrong to call it invented. Kept here rather than edited away.

`rlhfbook.com` still cannot be fetched directly from this environment — DNS is blocked at the
proxy (`403 CONNECT tunnel failed`, `ENOTFOUND`) — but **WebSearch does reach it**, which is how
the list above was verified. See `CURRICULUM.md` for the full chapter→unit mapping.
