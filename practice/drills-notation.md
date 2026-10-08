# Notation drills

Each drill is a formula. Your job: **write the plain-English sentence.** Not "what does
this compute" — what would you *say out loud* to explain it. Answers at the bottom; cover
them. Do these on paper, not in your head.

The point of these is that the ideas are already yours. Only the symbols aren't.

---

**1.** `Σᵢ pᵢ = 1`

**2.** `𝔼_{x∼D}[f(x)]`

**3.** `p(y | x)`

**4.** `H(p) = − Σᵢ pᵢ log pᵢ`

**5.** `L = − Σᵢ yᵢ log qᵢ`   (where `y` is one-hot)

**6.** `π_θ(a | s)`

**7.** `∇_θ log π_θ(a | s)`

**8.** `A = r − V(s)`

**9.** `KL(π_θ ‖ π_ref)`

**10.** `max_θ  𝔼_{x∼D, y∼π_θ(·|x)} [ r(x,y) ] − β · KL(π_θ ‖ π_ref)`

**11.** `∇_θ J(θ) = 𝔼[ ∇_θ log π_θ(a|s) · A ]`

**12.** `argmax_a π_θ(a | s)`  — and how is this different from `max_a π_θ(a | s)`?

---
---
---

## Answers

**1.** "The belief points add up to one." (You must spend the whole budget.)

**2.** "Draw an `x` from the dataset `D`, compute `f` of it, and average that over many
draws." In code: `mean(f(x) for x in batch)`.

**3.** "The probability of `y`, **given** that `x` already happened." The bar is the word
*given*. That's all it ever means.

**4.** "Entropy is the average surprise: for each outcome, how likely it is times how
surprising it would be." The minus sign is only there because `log` of a number below 1 is
negative — it's bookkeeping, not meaning.

**5.** "Minus the log of the probability you gave the right answer." Because `y` is one-hot,
every term is multiplied by zero except one, so the sum collapses to a single term. This is
cross-entropy loss, and it is the whole of next-token-prediction training.

**6.** "The chance my policy — using its current numbers `θ` — takes action `a` in
situation `s`."

**7.** "Which way to nudge the model's numbers to make the action `a` more likely in
situation `s`."

**8.** "How much better than usual this turned out." Advantage = what you got, minus what
you'd normally get.

**9.** "How far my tuned policy has drifted from the reference policy I started with."

**10.** "Pick the model numbers that get the highest average reward — over prompts drawn
from the dataset and answers drawn from my own model — minus a penalty for drifting from
where I started." Two terms: earn reward, don't become a different animal.

**11.** "To improve, nudge the numbers in whichever direction raises the log-probability of
the action I took, scaled by how much better than usual it went — averaged over a batch."
That's REINFORCE. Combine with **8** and you have the core of policy-gradient RLHF.

**12.** `argmax` = "**which** action scores highest" (returns the action).
`max` = "**what** is the highest score" (returns the number).
Mixing these up is one of the most common sources of confusion when reading pseudocode.

---

If any answer surprised you, that symbol goes in `QUESTIONS.md`.
