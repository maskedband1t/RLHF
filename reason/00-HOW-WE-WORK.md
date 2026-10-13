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


## Two modes: phone and keyboard

Added 2026-10-12, after discovering the plan assumed a keyboard that isn't always there.

**🎧 Phone / voice — learn and argue.** Lectures, prediction drills, Feynman checks, term
dumps, your-own-words TLDRs. Five of the seven loop steps above are phone-legal: 1 (guess),
2 (watch), 3 (dump terms), 4 (your TLDR), 7 (Feynman), plus the *prediction* half of 6.

**💻 Computer — build and verify.** `practice/stubs.py`, running `lab/` scripts to check my
numbers, writing the formula tables in `reason/`, committing.

Neither track blocks the other. The old plan had the keyboard-only harness sitting in front of
Lecture 1, which is pure listening — that was a sequencing bug, not discipline.

**Amendment to the division of labour.** The rule above gives you the predictions and TLDRs. In
voice mode the *substance* stays yours — your words, your reasoning, your guess — and the
**typing becomes mine.** I'm the scribe, not the author. If something I log back doesn't sound
like you said it, say so: a TLDR in my phrasing is worthless by your own rule.

**The honest cost of voice mode.** I run the scripts and read you the numbers, so the referee
is me-with-props rather than the output itself. Two mitigations: the numbers stay in the
transcript, and the computer track includes re-running the labs to spot-check me. A number I
read you that turns out wrong is a MISS in *my* column and gets logged the same as yours.

## Two standing rules for explanations

Added 2026-10-12, at your request.

1. **Tie everything back to language models.** A concept that stays abstract hasn't landed.
   Entropy isn't "spread of a distribution", it's *how unsure the model is about the next word*.
2. **Restate the definitions every time.** No explanation assumes the previous one is still
   loaded. Recap first, then the new thing.

Both are now required fields in the `CURRICULUM.md` unit template.


## Standing instructions — do not re-litigate these

Set 2026-10-13. Claude has no memory between sessions, so these live here.

- **Commits are always authored `maskedband1t <anurag.akkiraju@gmail.com>`.** Never
  `Claude <noreply@anthropic.com>`. Run
  `git config user.name maskedband1t && git config user.email anurag.akkiraju@gmail.com`
  at the start of a session. The repo's stop hook will flag these as "Unverified" and GitHub
  will show that badge — that is accepted and expected. Do not amend to a Claude author to
  satisfy the hook. `Co-Authored-By: Claude` in the message body is fine.
- **Always push.** No asking. This container is ephemeral; unpushed work dies with the session.

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
