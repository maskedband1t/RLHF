# Open questions

Dump half-formed confusions here the moment they appear. Don't wait until they're
well-posed — a badly-worded question is still a located confusion, and locating it is most
of the work. I'll pick these up at the start of a session.

**Format:** one bullet, plus where it came from. Move it to `ANSWERED` with the answer in
your own words — not mine. If you can't write it in your own words, it isn't answered yet.

---

## Open

- *(nothing yet — add as you go)*

---

## Answered

### Why is it called "soft" max?
Because the hard version returns `[1, 0, 0]` — winner takes all, which is a cliff you
can't do calculus on. Softmax is the smooth version: the winner gets most, not all.
Smooth means differentiable means you can train through it. *(from `lab/02`)*

### Why is everything in logarithms?
Because probabilities multiply, and 500 of them multiplied together is about `1e-651`,
which a float stores as exactly `0.0`. Every gradient from it is then zero and training is
dead. Logs add instead: the same thing comes out `-1497.87`, an ordinary number.
*(from `lab/07`)*

### Why does the loss only look at one number?
Because the truth is one-hot, so in `L = − Σᵢ yᵢ log qᵢ` every term is multiplied by zero
except the one that happened. And it doesn't need the others: the budget sums to 1, so
raising the right answer lowers the wrong ones automatically. *(from `lab/05`)*
