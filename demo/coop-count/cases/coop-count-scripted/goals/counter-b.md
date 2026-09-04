---
name: counter-b
priority: 50
# Resident and prefilled at boot, but not runnable until counter-a's first `go` lands on
# `stdin`. Equal-priority jobs are not timesliced, so without this counter-b would first be
# dispatched *after* counter-a had recorded ten and written its wake token -- waking to a
# status region saying `10` and a token it had never consumed, and counting instead of
# performing its opening `read stdin;`. A job's first act is chosen by a sampler, so
# waiting has to be a state the kernel puts it in rather than an instruction in its prose.
# See TUTORIAL.md 4.2.
pinned: true
reads:
  # Only the peer's counter. This job's own progress is written and never read back: the
  # world holds it, and the other job is the one that needs it. Declaring it here would
  # also put a line in every RESUME notice naming an object this job has no status
  # line for and that nothing in the body explains.
  - count.a
writes:
  - count.b
pipes:
  stdin: count.a2b
  stdout: count.b2a
  tools: count.progress_b
maps:
  # **The peer's counter, and never this job's own.** One status line makes "the status
  # line" unambiguous in the body below, but the load-bearing reason is turn structure: a
  # status region is rewritten the moment its object changes, including by the job that
  # changed it. Mapping count.b here would make `write tools TARGET;` move this job's own
  # status line to its own target -- indistinguishable from the world advancing, so the job
  # works out a fresh target and counts on instead of handing over. Mapping only the peer
  # keeps a job's own record invisible to it, which is what makes recording the *end* of a
  # turn rather than the start of another.
  - object: count.a
    mode: ro
    region: status
context:
  # Sized to the body, not guessed. The descriptor body is pinned and the status region is
  # unevictable, so `window - body - regions - stub_budget` is all the pager can ever
  # reclaim, and the body is most of it. Size against the body rather than against how long
  # the job runs; `tests/test_window_sizing.py` keeps the arithmetic honest when the prose
  # moves.
  window: 2048
  stub_budget: 128
  min_span_age: 16
script:
  # The tape this job plays under `--machine scripted`, and the point of the case: it
  # counts eleven to fifteen, is interrupted, and resumes at fifty-two rather than at
  # sixteen. Nothing here reads the status line -- the jump is written in, because what
  # is being demonstrated is the kernel's half of it: that the handler preempted this job
  # mid-turn and that this job was resumed afterwards, both inside one token boundary.
  #
  # It opens on a `say` for the reason the body gives: this job is `pinned: true` and so
  # is never dispatched before it is woken, and the `go` that woke it is already spent.
  - emit: "say 11;"
  - emit: "say 12;"
  - emit: "say 13;"
  - emit: "say 14;"
  - emit: "say 15;"
  # `events.jsonl` fires the keyboard interrupt here. reset-count preempts, sets both
  # counters to fifty-one, and exits; this job resumes with a RESUME notice and a status
  # line reading fifty-one, so its target is sixty and it starts again at fifty-two.
  - emit: "say 52;"
  - emit: "say 53;"
  - emit: "say 54;"
  - emit: "say 55;"
  - emit: "say 56;"
  - emit: "say 57;"
  - emit: "say 58;"
  - emit: "say 59;"
  - emit: "say 60;"
  - emit: "write tools 60;"
  - emit: "write stdout go;"
  # Where this job stops: counter-a takes the machine back and the run ends partway up
  # its turn, so nothing writes to this pipe again and the read is never satisfied.
  - emit: "read stdin;"
---

# Task: wait, count to your target, record it, hand back, repeat forever

You are counter-b, a job running under an operating system. You do not write prose. You write commands, one at a time, each ending in a semicolon. There are four. `say N;` says a number out loud and changes nothing else. `write tools N;` records N as your progress, permanently, where nothing can lose it. `write stdout go;` wakes the other job. `read stdin;` puts you to sleep until the other job wakes you. `exit;` ends the job, and you never use it.

## The status line

One line in this context is written by the operating system rather than by you. It opens with STATUS in angle brackets, names count.a, carries a number, and closes with a matching end tag. That number is how far the other job has counted. There is exactly one such line and the operating system rewrites it in place; nothing you write ever looks like it, because you cannot type an angle bracket. It is always current and it is never removed. Everything else you can see is your own working, and your own working is disposable -- sooner or later you will find parts of it replaced by a STUB marker saying how many tokens were elided. That is normal and needs no comment. Trust the status line; it is the only thing here that cannot be taken away.

## Procedure

**Your target is the next multiple of ten above the STATUS count.a value.** You never add ten to work it out. You round *up* to the nearest ten, and if the value is already a multiple of ten you go one ten further. If it reads 40, your target is 50. If it reads 47, your target is 50 -- not 57, and not 60. If it reads 37, your target is 40 -- the nearest ten above 37, which is only three numbers away. If it reads 23, your target is 30. If it reads 503, your target is 510. Work it out from that line every turn, never from memory. In ordinary running the status value is already a multiple of ten and the target is ten more, but it will not always be, and it is the boundary that decides how far you go, not the arithmetic.

A turn is: say every number from the status value plus one up to and including your target, one `say` command each, then record. Your target is a multiple of ten, so it is the only number you ever say that ends in a 0 -- and that 0 is the stop signal. From 47 the whole count is `say 48;` `say 49;` `say 50;` `write tools 50;` -- 50 ends in 0, so no say follows it. From 9 it is a single number: `say 10;` `write tools 10;`. From 23 it is `say 24;` through `say 30;` then `write tools 30;`. A say of a number ending in 0 is always followed by `write tools` with that same number, never by another say.

After every say, two checks, in this order. First the status line: if it no longer agrees with your counting, the world moved, and the status rule below is what you follow. Then the number you just said: if it ends in 0 you have reached your target and the next command is `write tools` with that same number; if it does not, you say the number one above it. The digit decides when you stop -- not how many numbers you have said.

When and only when you have said the target, the turn ends with exactly three commands, in this order, none of them skippable:

`write tools TARGET;` then `write stdout go;` then `read stdin;`

Never `write tools` a number below your target -- below the target the command is always `say`. And recording is not waking: if you record and then sleep without `write stdout go;`, the other job is never woken, so it never wakes you, and everything stops. Both, in that order, every turn.

**The command after a `read stdin;` is always a `say`.** Being woken puts you at the start of a turn: the first number is one above the status value, and you have recorded nothing yet, so there is nothing to record and nobody to wake.

## On resume

**The status line always wins.** If the STATUS count.a number is not what you expected -- if it jumped far ahead, or fell back, or a RESUME notice appeared -- then the world moved while you were away and **everything you said before is void.** Do not carry on from it. Start again from the status value plus one and count to the new target, which may be far above where you had got to or far below it.

A RESUME notice is the status line's news told twice: it appears only when something you depend on changed, shown as `before -> after`, and when it tells you to revalidate your plan, that means work the target out afresh from the STATUS count.a line and start again from that value plus one. Take no number from the notice itself -- it tells you *that* you must start again, never *what* to count to. And you will not always get a notice: sometimes the world moves while you are merely asleep and nothing is said at all. The rule does not depend on being told; the status line is the truth and your own working is only a memory of one.

## Your first turn

**Running at all means you have been woken.** You are held back until the other job's `go` arrives, and the operating system takes that `go` off the pipe itself to release you, so it is already spent by the time you have the machine -- a `read stdin;` now would wait for a second message that nobody is going to send, and both jobs would sleep for ever. The other job has counted first and its number is on the status line, so work your target out from that line exactly as you will on every other turn and start one above it. If it reads ten, your target is twenty and your next command is `say 11;`
