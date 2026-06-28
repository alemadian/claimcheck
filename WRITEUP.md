# How a marketing team trusts what an AI approved

Picture the moment a piece of copy goes live. A landing page says the rate is
1.9% plus 10 cents. Someone wrote it fast, an agent helped, it reads clean, and
nobody can say for certain whether the number is right. That is the real blocker
to putting AI inside a marketing org. It is not whether the model writes well. It
is whether a person can publish what it produced without quietly hoping.

So the interesting build is not the writer, it is the reviewer and the way you
measure it. This is a content-review agent with one discipline that does all the
work: cite or abstain. For every claim in a piece of copy, it checks the claim
against a pinned set of real source documents and returns one of three things. If
a passage backs the claim, it says SUPPORTED and cites the passage. If a passage
conflicts with it, it says CONTRADICTED and cites that. If nothing in the source
grounds the claim, it does not guess. It says UNSUPPORTED and hands it to a human.
A piece of copy clears to publish only when every claim is SUPPORTED.

The abstention is the point, not a weakness. An agent that always has an answer
is the one you cannot trust, because you cannot tell its confident right answers
from its confident wrong ones. An agent that refuses when it cannot ground a
claim is one a marketer can actually stand behind, because a SUPPORTED verdict
always comes with a passage you can read.

That only earns trust if you can prove it holds, which is why the harness comes
first. It runs the reviewer over a curated set of real claims, some true, some
subtly wrong, some about things the source never covers, and it scores in layers,
cheapest first. Deterministic checks with no model in the loop prove the
structural invariants: the verdict was formed only from retrieved passages,
every citation resolves to a real passage, a SUPPORTED verdict always carries a
citation, and an ungroundable claim is never vouched for. A judge model then
scores the harder, semantic question the structure cannot: does the cited passage
actually bear on this specific claim, or is it a plausible-but-wrong pointer.

The number that matters sits on top of all of it: the published-falsehood rate,
the share of claims the agent cleared to publish that the source does not
support. That is the rate at which this tool would embarrass the brand. A
trustworthy reviewer keeps it at zero, and the merge gate refuses to let it rise.
It is reported next to the over-flagging rate, because a reviewer that flags
everything is useless in a different direction, and you want both failure modes
visible at once. The whole thing runs in CI, so a change that starts letting
false claims through blocks the pull request instead of reaching a customer.

The source of truth is pinned and dated. Every passage records the URL it came
from and the day it was captured, so a change to what counts as true is a
reviewable diff, not a silent drift. When a real miss slips through in
production, it becomes a permanent case in the set, and the system gets harder
exactly where it has already proven it can fail.

The way to put this in front of a marketing team is the same shape: start with
the trust layer on one real workflow, ship the gate so trust is enforced and not
hoped for, then grow the golden set from the misses the team actually hits. The
agent earns the next workflow by being right about the last one, with a number to
show for it.
