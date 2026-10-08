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

- [ ] Prereq: ML foundations video
- [ ] Lecture 1 — intro / overview
- [ ] Lecture 2 — instruction finetuning
- [ ] Lecture 3 — reward models
- [ ] Lecture 4 — policy gradient / RL maths
- [ ] Lecture 5 — implementation details
- [ ] Lectures 6–8
- [x] Maths vocabulary — `00-MATH-TLDR.md`, all nine `lab/` scripts, `practice/` harness

⚠️ **The lecture list above is unverified** — taken from search results, not from the course
page, because `rlhfbook.com` was blocked by the network policy when this was written.
Confirm against [rlhfbook.com/course](https://rlhfbook.com/course) and fix this table.
