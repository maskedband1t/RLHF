# The whole vocabulary, on one page

You are not missing a maths education. You are missing **one object and six verbs**, written
in notation nobody decoded for you. The object takes ten minutes. The notation takes twenty.

---

## The one idea everything is built on

> **A probability distribution is a budget of 100 belief points, and you must spend all of them.**

That's it. `[70, 20, 10]` — 70 points on rain, 20 on cloud, 10 on sun. Written as `[0.7, 0.2, 0.1]`.

A model's output *is* this list. One number per option. For a language model, one number per word
in its vocabulary — a 100,000-item budget instead of a 3-item one. Nothing else changes.

Two consequences, and almost everything follows from them:

1. **Belief is zero-sum.** The total is pinned at 1. To raise one option you *must* lower others.
2. **You can be graded.** Reality happens, and we can check how many points you'd put on it.

Everything below is an operation on that budget. That's the whole subject.

---

## Decoder ring — read this before any formula

The ideas aren't hard. The symbols are unfamiliar. These nine lines are most of what blocks you:

| symbol | say it out loud as | in code |
|---|---|---|
| `Σᵢ` | "add up, for every i" | `sum(...)` — it's a for-loop |
| `Πᵢ` | "multiply together, for every i" | `math.prod(...)` |
| `𝔼[X]` or `E[X]` | "the average of X if you did it many times" | `.mean()` over a batch |
| `𝔼_{x∼p}[f(x)]` | "draw x from p, compute f(x), average it" | `mean(f(x) for x in samples)` |
| `x ∼ p` | "x is drawn at random from p" | `random.choices(..., weights=p)` |
| `p(y\|x)` | "chance of y **given** x already happened" | the `\|` is the word *given* |
| `θ` (theta) | "all the model's numbers" (its weights) | `model.parameters()` |
| `π_θ` (pi) | "the policy — the thing that decides — using those numbers" | `model` |
| `∇_θ` (nabla) | "which way to nudge θ to go uphill" | `loss.backward()` |

Plus: `log` means natural log unless stated. `argmax` = *which* option wins (`max` = the winning
score). `β`, `λ` = a knob a human picked. `1[...]` = 1 if true, else 0 — a switch.

**The single biggest unlock:** `𝔼[...]` means *"run it a lot and average."* Every loss function in
every RL paper is "average this quantity over the batch." Once `E` stops being scary, so do the papers.

---

## The six verbs

### 1. Softmax — the exchange rate from scores to budget
> **TLDR: exponentiate each score, divide by the total. Two lines of arithmetic.**

A model doesn't naturally emit numbers that sum to 1. It emits raw enthusiasm — **logits**, aka
scores — which can be any number: `rain 2.0, cloud 1.0, sun 0.0`. Softmax converts:

```
e^2 = 7.39      7.39 / 11.11 = 0.665
e^1 = 2.72  →   2.72 / 11.11 = 0.245     ← sums to 1. Done.
e^0 = 1.00      1.00 / 11.11 = 0.090
```

Why exponentiate? Two reasons, both practical: `e^x` is always positive (no "-10% chance of cloud"),
and it makes only the **gaps** between scores matter — `[2,1,0]` and `[102,101,100]` give the
identical answer.

**The gap *is* the confidence.** Winner leads by 0: `33/33/33`. By 1: `58/21/21`. By 4: `96/2/2`.

Why "*soft* max"? Hard max answers `[1,0,0]` — winner takes all, a cliff you can't do calculus on.
Softmax is the smooth version. Smooth means differentiable means trainable. That's the whole name.

**Temperature** is a knob on the gaps and nothing more: divide scores by `T` before softmax.
`T=0.5` doubles every gap (bolder). `T=2` halves them (more hedged). `T→0` = always pick the top one.

→ `lab/02_softmax.py`, `lab/03_temperature.py`

### 2. Entropy — how wide is your hedge
> **TLDR: the average number of yes/no questions needed to pin down the answer.**

Verify it on cases you can check by hand: 2 equally likely options = **1 bit**. 4 options = **2 bits**.
1024 options = **10 bits**. One certain option = **0 bits** — no questions needed, no doubt.

Where the formula comes from: the **surprise** of an event is `log(1/p)`. A certain event surprises
you 0. A 1-in-1000 event surprises you ~10 bits. Entropy is just your surprise *averaged over what
you yourself expect*:

```
H = Σ  p  ·  log(1/p)
    └ how likely ┘ └ how surprising ┘
```

**Why RLHF cares:** entropy is exploration. *"Entropy collapse"* = the model got so certain of one
answer that it stopped trying others, and learning stalls. *"Entropy bonus"* = a term in the loss
saying "stay a little unsure."

→ `lab/04_entropy.py`

### 3. Cross-entropy — the grade, and the training loss
> **TLDR: how surprised were you by what actually happened? Lower is better.**

It rained. Your bill:

| you said | loss |
|---|---|
| 90% rain | 0.15 bits |
| 70% rain | 0.52 bits |
| 10% rain | 3.32 bits |
| 1% rain | 6.64 bits |
| 0% rain | **infinite** |

**Only one number mattered: the belief you put on the thing that happened.** Look at the formula:

```
L = − Σᵢ yᵢ · log qᵢ
```

`y` is the truth as a one-hot budget — `[1,0,0]` — so **every term is multiplied by zero except one**.
The whole sum collapses to `−log q[the right answer]`. That's it. That's cross-entropy.

And that last row is why softmax matters: **softmax can never output exactly 0**, which is exactly
what keeps the loss finite. The pieces were built to fit.

**This is how language models are trained.** For every position, raise the belief on the word that
actually came next. Because the budget sums to 1 (zero-sum!), raising the right word *automatically*
lowers every wrong word. You never push the wrong answers down by hand.

**Perplexity** = `2^(average loss in bits)`. Perplexity 20 = "as lost as someone guessing uniformly
among 20 options at every word." It's cross-entropy in friendlier units.

**The decomposition** worth knowing: `cross-entropy = entropy(truth) + KL(truth ‖ you)`. Your grade is
irreducible uncertainty nobody could beat, *plus* how wrong your beliefs were. Only the second part
is yours to fix — so the loss never reaches zero, and that's correct, not a bug.

→ `lab/05_cross_entropy.py`

### 4. KL divergence — the leash
> **TLDR: how far has my new budget drifted from the old one?**

Base model believed `[0.70, 0.20, 0.10]`. After tuning you believe `[0.95, 0.04, 0.01]`.
KL puts a number on that move: `0.49 bits`.

It is **not symmetric** — `KL(p‖q) ≠ KL(q‖p)`. "Distance" is the wrong word. It's *"how surprised
one would be by the other"*, and that depends on who's doing the being-surprised. Which direction you
minimise decides your behaviour: one makes you cover everything, the other lets you pick one mode
and commit.

**What it's for.** Your reward model likes bullet points. Without a leash, your model discovers this
and emits bullet points forever, for every question. Reward: excellent. Model: useless. That's
**reward hacking**, and the KL term prices it — every bit of drift costs you.

→ `lab/06_kl.py`

### 5. Log-probs — why everything is in logarithms
> **TLDR: probabilities multiply and vanish. Logs add and survive.**

A 500-token answer at 5% belief per token. Multiply the probabilities: the true value is about
`1e-651`, far under the smallest number a float holds (`~1e-308`). **It silently becomes 0.0, every
gradient becomes zero, and training is dead.** In logs it's `-1497.87` — an ordinary number.

That's the entire reason. `log(a·b) = log(a) + log(b)` is doing all the work.

Reading the units: log-prob `0` is perfect, more negative is worse. And `−log(p) = log(1/p) =`
surprise — same quantity, three names, and papers use all three.

→ `lab/07_logprobs.py`

### 6. Policy gradient — make what worked more likely
> **TLDR: do the thing. See what you got. Raise the probability of what paid.**

REINFORCE is one line:

```
θ ← θ + lr · advantage · ∇_θ log π_θ(action)
```

Read it as: *"nudge the numbers in whichever direction raises the log-probability of the action I
took, scaled by how much better than usual it turned out."*

**"Advantage" is the part worth understanding.** Three buttons paying 10, 9.5, 9. Raw reward says
*push all three up hard* — which is self-cancelling, since the budget sums to 1. Subtract the
average (9.5) and the signal becomes **signed**: `+0.5 / 0.0 / −0.5`. Up, leave, down.

> **"Better than usual" is a real signal. "Good" is not.** If everything scores 9-ish, the absolute
> number carries no information; only the differences do.

The payoff, measured in `lab/08`: add 100 to every payout — which changes *nothing* about which
button is best — and the no-baseline learner picks the **wrong button 6 times out of 12 seeds**.
The baselined one gets it right 12/12 and returns *bit-identical* answers across all three payout
tables, because it only ever sees differences.

That subtraction has three names for one idea: **baseline**, **advantage**, `A = r − V(s)`.

**GRPO's whole trick:** sample a *group* of answers to the same prompt and use the group's own mean
as the baseline. No value network needed. That's the entire innovation, and it's 4 lines in `lab/08`.

→ `lab/08_policy_gradient.py`

---

## Now read the real objective

The main RLHF formula, which you can now decode word by word:

```
max_θ   𝔼_{x∼D, y∼π_θ(·|x)} [ r(x,y) ]  −  β · KL(π_θ ‖ π_ref)
```

> "Pick the model numbers (`θ`) that get the highest **average** (`𝔼`) reward (`r`), over prompts
> drawn from your dataset (`x∼D`) and answers drawn from your own model (`y∼π_θ`) — **minus** a
> penalty (`β`) for drifting from where you started (`π_ref`)."

Two terms. Earn reward; don't become a different animal. You already know both halves.

---

## Learn it in this order — the dependency graph

Don't study this flat. It's a chain, and three links carry most of the weight:

```
budget (01) ──► softmax (02) ──► temperature (03)
                   │
                   ├──► entropy (04) ──► cross-entropy (05) ──► KL (06)
                   │                          │
                   └──► log-probs (07) ───────┴──► policy gradient (08) ──► calibration (09)
```

**If you only get three:** the budget, softmax, and cross-entropy. With those you can read an RLHF
paper's loss function. Entropy and KL are the next two. Everything else is assembly.

---

## The ten-minute drill that converts "I get the vibe" into fluency

Reading a TLDR gets you to recognition. Recognition collapses the moment a real formula appears.
What closes the gap is doing the arithmetic **by hand, once, on three numbers**:

1. Write `[2, 1, 0]`. Softmax it with a calculator. Get `[0.665, 0.245, 0.090]`. Check it sums to 1.
2. Compute its entropy: `−(0.665·log₂0.665 + 0.245·log₂0.245 + 0.090·log₂0.090)`. Get ~1.20 bits.
3. Say rain happened. Your loss is `−log₂(0.665)` = 0.59 bits. Note you ignored the other two numbers.
4. Now `[3, 1, 0]`. **Predict before you compute:** does the loss go up or down? By roughly how much?
5. Check with `lab/02` and `lab/05`.

Step 4 is the one that matters. Thirty minutes of this, once, and the notation stops being a wall.
