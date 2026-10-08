"""Practice: implement each one yourself, then run `python3 check.py`.

Rules of engagement:
  - Do NOT look at ../lab/common.py until check.py passes. That file has the answers.
  - Implementing these from the idea, not from memory of the code, is the whole point.
  - Pure stdlib. `import math` is all you need.
  - Partial credit is real: check.py reports each function separately, so do them
    one at a time and re-run after each.

Order matters. softmax -> entropy -> cross_entropy -> kl -> advantage.
"""
import math


def softmax(scores, temperature=1.0):
    """Turn raw scores into a budget of belief that sums to 1.

    The idea: belief is 100 points you must spend entirely. These scores are raw
    enthusiasm and can be any number, including negative. Convert them to a budget.

        softmax([2, 1, 0])  ->  [0.665, 0.245, 0.090]      (sums to 1.0)

    Three things to get right:
      1. Output must sum to 1.0
      2. Output must never be negative, even if every score is negative
      3. `temperature` divides the scores BEFORE the rest of the work.
         T < 1 widens the gaps (bolder). T > 1 narrows them (more hedged).

    Hint if stuck: what function turns any real number into a positive one,
    and grows fast enough that a lead of 1 becomes a big share?
    """
    raise NotImplementedError


def entropy(p):
    """How spread out the budget is, in bits.

    The idea: the average number of yes/no questions needed to pin down the answer.

        entropy([1, 0, 0])        -> 0.0    no doubt, no questions needed
        entropy([0.5, 0.5])       -> 1.0    one question
        entropy([0.25]*4)         -> 2.0    two questions
        entropy([0.7, 0.2, 0.1])  -> 1.157

    Build it from "surprise". The surprise of an outcome with probability p is
    log2(1/p). Entropy is your surprise AVERAGED over what you yourself expect.

    Watch out: an outcome with p=0 contributes nothing, but log2(0) explodes.
    Skip those terms.
    """
    raise NotImplementedError


def cross_entropy(p, outcome):
    """Your grade. `outcome` is the INDEX of the thing that actually happened.

    The idea: how surprised were you by what happened? Lower is better.

        cross_entropy([0.7, 0.2, 0.1], 0)  -> 0.515   you had 70% on it
        cross_entropy([0.7, 0.2, 0.1], 2)  -> 3.322   you only had 10% on it

    This is the loss language models are trained on. The thing to notice while
    you write it: how many of the numbers in `p` does your code actually touch?
    """
    raise NotImplementedError


def kl(p, q):
    """How far q sits from p, in bits. Not symmetric -- kl(p,q) != kl(q,p).

        kl([0.7,0.2,0.1], [0.7,0.2,0.1])  -> 0.0     identical, no gap
        kl([0.7,0.2,0.1], [0.3,0.3,0.4])  -> 0.539
        kl([0.3,0.3,0.4], [0.7,0.2,0.1])  -> 0.609   note: a DIFFERENT number

    You already wrote the pieces: this is cross_entropy(p, q) minus entropy(p),
    generalised to a soft `p` instead of a single outcome index. Weight each term
    by p, and skip terms where p is 0.
    """
    raise NotImplementedError


def advantage(rewards):
    """Turn raw rewards into "better than usual".

        advantage([10, 9.5, 9])        -> [0.5, 0.0, -0.5]
        advantage([110, 109.5, 109])   -> [0.5, 0.0, -0.5]   <- IDENTICAL. this is the point.
        advantage([5, 5, 5])           -> [0.0, 0.0, 0.0]    no information

    One line. The reason it matters is in ../lab/08_policy_gradient.py: without
    this, shifting every reward by a constant changes which action gets learned.
    This is also the entirety of GRPO's trick -- the group's own mean is the baseline.
    """
    raise NotImplementedError
