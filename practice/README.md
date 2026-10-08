# Practice

Reading about softmax and writing softmax are different skills. This directory is the
second one. It is self-checking, so you are never blocked waiting on me.

```bash
cd practice
python3 check.py          # 0/24 to start -- that is correct
# implement one function in stubs.py
python3 check.py          # re-run after each one
```

**Order:** `softmax` → `entropy` → `cross_entropy` → `kl` → `advantage`. Each builds on
the last. `kl` in particular is just two things you already wrote, combined.

**Don't look at `../lab/common.py` until you're green.** It has the answers. Looking gives
you the feeling of understanding and none of it.

Every failure prints what you got, what was expected, and a hint aimed at the specific
mistake — a flipped sign, an ignored argument, a missing normalisation. Four of the 24
tests are not about getting a number right but about getting a *property* right:

- softmax must sum to 1
- softmax must survive `[1002, 1001, 1000]` (the overflow trap — worth hitting on purpose)
- `kl(p,q)` must differ from `kl(q,p)`
- `advantage` must give identical answers for `[10,9.5,9]` and `[110,109.5,109]`

Those four are the ones that teach something. If you pass the arithmetic and fail these,
you've implemented a formula without understanding what it's for.

When you're green: read `../lab/common.py` and compare. **If yours differs but passes,
yours is just as correct** — note the difference in `notes/PREDICTIONS.md`. There's usually
more than one honest way to write these.

## Also here

`drills-notation.md` — formulas, and your job is to write the English sentence. Answers at
the bottom. Do these on paper. Twenty minutes, and the notation stops being a wall.
