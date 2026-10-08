# RLHF study log

Working through **Nathan Lambert's RLHF Book** (`rlhfbook.com`) and its YouTube lecture course.

This repo is not a notes dump. A notes dump is dead weight — you reread it and feel like you
understand things you can't actually use. The rule here is:

> **Every concept gets a script that prints numbers, and I write down what I think the numbers will
> be before I run it.**

That's borrowed from my own [RLCD](https://github.com/maskedband1t/rlcd) methodology
(`notebook/CLAIMS.md`, prediction-before-result). It works on research. It works on learning too,
for the same reason: it's the only way to catch yourself *feeling* like you understand something.

## Start here

| file | what it's for |
|---|---|
| **[00-MATH-TLDR.md](00-MATH-TLDR.md)** | The entire maths vocabulary on one page. Read this first. |
| [GLOSSARY.md](GLOSSARY.md) | Term → one line. Grows as I hit new ones in the lectures. |
| [lab/](lab/) | Nine runnable scripts, one per concept. No dependencies. |
| [notes/PREDICTIONS.md](notes/PREDICTIONS.md) | Prediction before result. The actual study method. |
| [notes/LECTURES.md](notes/LECTURES.md) | One row per lecture. Term I didn't know → my own TLDR. |

## Run the lab

Needs nothing but `python3`. No pip, no venv, no numpy.

```bash
cd lab
python3 01_budget.py          # what a "distribution" actually is
python3 02_softmax.py         # scores -> budget, by hand
python3 03_temperature.py     # one knob on the gaps
python3 04_entropy.py         # how wide is the hedge
python3 05_cross_entropy.py   # the training loss
python3 06_kl.py              # the leash
python3 07_logprobs.py        # why everything is in logs
python3 08_policy_gradient.py # REINFORCE + why "advantage" exists
python3 09_calibration.py     # confidence vs hit rate
```

Or `./run_all.sh`.

Each script ends in `NOTICE:` lines — the takeaways. Each one is short enough to read end to end.

## The study loop

1. Watch the lecture. Don't pause to look things up; let it wash over once.
2. Every term that didn't land → a row in `notes/LECTURES.md`.
3. For each row, write a TLDR **in my own words**. If I can't, I don't have it yet.
4. Open or write the matching `lab/` script. **Write the prediction down first** in
   `notes/PREDICTIONS.md`. Then run it.
5. Where the prediction was wrong, that's the real lesson. Log it. Don't edit the prediction.

Step 5 is the whole point. `lab/08_policy_gradient.py` carries a correction where my first
explanation was flatly contradicted by the script's own output — it's kept in the file on purpose.

## Honest status

- [ ] Prereq: ML foundations video
- [ ] Lecture 1 — intro / overview
- [ ] Lecture 2 — instruction finetuning
- [ ] Lecture 3 — reward models
- [ ] Lecture 4 — policy gradient / RL maths
- [ ] Lecture 5 — implementation details
- [ ] Lectures 6–8
- [x] Maths vocabulary — `00-MATH-TLDR.md` + all nine lab scripts

(Lecture list is approximate — confirm against `rlhfbook.com/course` and fix this table.)
