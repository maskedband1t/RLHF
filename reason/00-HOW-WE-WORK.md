# How we work through this

The goal is not to finish the book. It is to get to the point where a formula in an
RLHF paper reads as a sentence. That happens by going slowly and in our own words,
and it does not happen by reading faster.

## The loop, per chapter or lecture

1. **Guess first.** Before reading, write what you think the chapter is about in
   `reason/chNN-*.md`. Two sentences. It will often be wrong — that is the value.
2. **Read / watch once, without stopping.** No pausing to look things up. Let it wash over.
3. **Dump every term that didn't land** into the chapter file. Don't filter. A term you
   half-recognise counts as not landing.
4. **Write your own TLDR for each.** In your own words. If you can only produce the
   lecturer's phrasing back, you don't have it yet — mark it `STUCK` and move on.
5. **Decode the formulas symbol by symbol.** Fill in the table in the template. This is
   the single highest-value step and the one most tempting to skip.
6. **One `STUCK` item becomes a script.** We build a `lab/NN_*.py` that prints numbers.
   You write your prediction in `notes/PREDICTIONS.md` *before* running it.
7. **The Feynman check.** Write the "what I'd tell someone else" paragraph. If it comes
   out muddy, the understanding is muddy. Go back to 5.

Steps 1 and 6 are the ones that do the work. Steps 2–4 just feed them.

## Our division of labour

**You:** the guesses, the own-words TLDRs, the predictions, and saying plainly when
something doesn't land. That last one is not a failure mode, it's the input I need.

**Me:** turning any `STUCK` item into the smallest script that settles it; decoding
notation; finding the simple example; and telling you when I'm wrong.

**Non-negotiable both ways:** we check claims against output. `lab/08_policy_gradient.py`
exists in its current form because my first explanation of `advantage` was contradicted by
the script's own numbers. That correction is kept in the file. Expect more of them —
and push back when something I say doesn't sit right. The script is the referee, not me.

## Starting a new session

I don't carry memory between sessions. To get me up to speed in one line, paste:

> Read SESSION-LOG.md, QUESTIONS.md, and the newest file in reason/. We're on <chapter>.

`SESSION-LOG.md` is the handoff. Keep it current or this breaks.

## What counts as done

A chapter is done when:
- [ ] Every term in its table has a TLDR you wrote, with no `STUCK` left
- [ ] Every formula has a plain-English sentence next to it
- [ ] The "what I'd tell someone else" paragraph reads cleanly
- [ ] Any script it needed exists, and your prediction is logged — hit or miss

Not when you've read it. Reading is step 2 of 7.
