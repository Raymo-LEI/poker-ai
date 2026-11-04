# Search mathematics v1 (2026-11-04)

This contract supersedes the unversioned V13/V14/V15 search, value labels and
preflop tables. It does not certify any frozen policy's playing strength.

## State and utility

Amounts are integer **additional chips**, never raise-to. Pot includes the
outstanding opponent bet. Stacks exclude all chips already in the pot. A
successor uses the platform minimum (`2*round_raise-own_street_put`), positive
bets strictly below stack, and all-in as a separate response. No raises follow
an opponent all-in. Heads-up unmatched chips are refunded at showdown; equal
initial stacks on Botzone make this equivalent to the referee's two all-ins.
Unequal-stack tests use the explicitly extended heads-up refund convention.

Search utility is `(final_stack - root_stack - root_pot/2)/BB`, for each player.
It is zero sum. Uncentered root delta adds `root_pot/(2*BB)`. Whole-hand delta
also adds `(root_stack-starting_stack)/BB`. Moving a leaf value to a root adds
`(leaf_stack-root_stack + (leaf_pot-root_pot)/2)/BB`. Do not charge an outstanding
bet twice. Terminal states have no action slots.

## Beliefs and values

Use public combo weights w0,w1 and joint chance `w0[h]*w1[o]*compatible(h,o)`.
Joint normalization is global; each conditional value divides by its own
`Z(h)=sum_o w1[o]*compatible(h,o)`. Traversal carries **unnormalized** opponent
behavior reach, and removes own type prior only as an information-set constant.
Unreachable conditional values are NaN with an explicit reachability mask.
Zeros stay zero; empty ranges raise an error. No epsilon invents range mass.

History likelihood is the frozen policy forward for each candidate hand at the
**event-time** public state. Legal probabilities normalize together. Observed
noncanonical sizes are explicitly coarsened to the aggression category, not
presented as exact size likelihood. Context is a declared neutral-context
observation model, not hindsight HUD. Board reveals filter combo weights before
any aggregation. The joint deal's remaining-card count is constant, so the
likelihood of an observed board cancels for compatible pairs. Replaying the
whole history is equivalent to migrating the same combo posterior; do not
restart uniformly at a street boundary. Candidate combos may share cards;
only a realized pair/runout must be disjoint.

## DCFR

Default alpha=1.5, beta=0, gamma=2. At iteration t>=1, freeze all strategies
from the signed regrets stored at the preceding iteration end (uniform legal
fallback at zero positive mass). Traverse that one snapshot, then store
`R_t = discount_t(R_(t-1)) + instantaneous_regret_t`, where positive discount
is `t^alpha/(t^alpha+1)` and negative discount is `t^beta/(t^beta+1)=1/2`.
Only regret matching clips to positive. Average sums start at zero and add
`t^gamma * own_behavior_reach * sigma_t`. Regrets include opponent/chance
reach, never own behavior reach. Initial strategy is a snapshot initializer,
not restored historical regret. All legal domains are explicit enumerations.
Returned policies are DCFR average strategies, not a KL response.

An iteration budget is a stopping budget, not an equilibrium certificate.
Reports include iterations and external best-response gap where available.
Original external-sampling MCCFR is separate and unchanged: sampling visitation
already accounts for opponent/chance reach.

## Abstraction and value schema

Root actions are exactly the supplied executable mapping. Duplicate chip
amounts share the same successor; they are never assigned a maximum over other
sizes. Continuation uses check/call, pot and all-in with an explicit raise cap;
at the cap only check/call/fold remain. This is a restricted betting game.
Runtime pre-river continuation is a named check-down approximation, never an
old neural leaf. Tiny full turn labels enumerate chance then river betting.

`combo_value_v1`: exact board card IDs, street (0/1/2/3), pot/BB, both remaining
stacks/BB, both independent combo lists and unnormalized range masses. No
backbone placeholder or untrained feature columns. Outputs are conditional
centered root BB values for each side, with validity masks. Encoding/decoding
outputs is identity in BB. Weighted values obey zero sum under the compatible
joint deal, **not** under the unconditioned single-side marginals.

Legacy `vleaf.*`, `cfv_turn.*`, `cfv_river.*`, `preflop_freq` without this exact
contract and generator provenance are rejected as value/table artifacts.
Frozen policy parameters remain usable as a statistical prior only. New labels
must solve their actual input ranges; leaf builders use actual successor state.
Production labels, production training and target-platform acceptance remain
separate follow-up tasks.

The complete value state also carries actor, first actor, pending decisions,
per-street contributions, platform round_raise and BB chip size; pot and stacks
alone do not distinguish a checked leaf from a facing-bet state.

DCFR reference: [Brown and Sandholm](https://arxiv.org/abs/1809.04040).
The explicit iteration/storage convention above is part of this implementation;
no convergence claim for the runtime's limited budgets is inferred from it.
