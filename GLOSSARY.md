# Glossary

One line each, in plain English. Add a row the moment a lecture uses a term I can't say back.
If I can't write the right column myself, I don't have it yet — that's the test.

## The maths vocabulary

| term | TLDR |
|---|---|
| distribution | A list of non-negative numbers that add to 1. A budget of belief. |
| logits / scores | The raw numbers *before* softmax. Any value, positive or negative. |
| softmax | Exponentiate the scores, divide by the total. Scores → budget. |
| temperature | A knob that stretches (`T<1`) or squashes (`T>1`) the gaps between scores. |
| greedy / `T=0` | Always take the top-scoring option. No sampling. |
| entropy | How spread out the budget is. In bits: yes/no questions to pin down the answer. |
| surprise | `log(1/p)`. Certain → 0. Rare → large. |
| cross-entropy | How surprised you were by what happened. The training loss. |
| perplexity | `2^(avg loss in bits)`. "As lost as guessing among N options." |
| KL divergence | How far one distribution sits from another. Not symmetric. |
| log-prob | `log(p)`. 0 is perfect, more negative is worse. Logs add, so they don't underflow. |
| expectation `𝔼` | Average over many tries. In code: `.mean()` over the batch. |
| gradient `∇` | Which way is uphill, and how steep. |
| policy `π` | The thing that decides. `π(a\|s)` = chance of action `a` in state `s`. |
| parameters `θ` | All the model's numbers. What training changes. |
| REINFORCE | `θ += lr · advantage · ∇ log π(action)`. Raise the log-prob of what paid. |
| baseline | A reference score you subtract from reward before using it. |
| advantage | `reward − baseline`. "Better than usual", which beats "good". |
| calibration | Of all the times you said 70%, did it happen 70% of the time? |
| ECE | Expected Calibration Error. Average gap between claimed confidence and hit rate. |

## RLHF-specific — fill these in from the lectures

| term | TLDR |
|---|---|
| SFT / instruction finetuning | |
| reward model | |
| preference pair | |
| Bradley-Terry | |
| PPO | |
| clipping / clip ratio | |
| GRPO | |
| DPO | |
| reference model (`π_ref`) | |
| KL coefficient `β` | |
| reward hacking | |
| rejection sampling / best-of-n | |
| value function / critic | |
| loss aggregation | |
| RLVR | |
