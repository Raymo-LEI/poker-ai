# Open-source Stage 2 opponent shortlist

Checked 2026-10-29 against upstream repositories, source and model cards.
This is a source/asset assessment, not a local playing-strength reproduction.
No external opponent has been installed into a live training pool.

## Best first investigation: DecisionHoldem blueprint

[DecisionHoldem](https://github.com/AI-Decision/DecisionHoldem) supplies a
blueprint-only `blueprint.so` interface and a separate search-enabled
`AlascasiaHoldem.so` interface. Its upstream
[Slumbot adapter](https://github.com/AI-Decision/DecisionHoldem/blob/main/pypokergui/play_with_slumbot.py)
uses 50/100 blinds and 20,000-chip stacks, matching our standard 200BB game.
The authors report >730 mbb/hand over about 20,000 Slumbot hands for their
system; that does not establish the same strength for blueprint-only play.

The README requires six external files, including street clusters,
`sevencards_strength.bin` and `blueprint_strategy.dat`, hosted on Baidu Netdisk.
Their present downloadability, hashes and inference throughput have not been
verified. The source declares AGPL-3.0. The supplied wrapper has mutable native
state: use a dedicated persistent worker, not one shared singleton across
concurrent matches. Rebuild/audit the native library and define the ctypes ABI
explicitly rather than copying the demonstration wrapper unchanged.

Recommendation: first test **blueprint-only** as a frozen local opponent, then
keep the more expensive search version for periodic evaluation. No retraining
is needed *if* the complete supplied artifacts work. This is not yet a ready
drop-in actor.

## Best matching published HU200 model: CFR-NLHE-6max v0.5.5

Despite the historical name, the upstream
[v0.5.5 release](https://github.com/dberweger2017/CFR-NLHE-6max/releases/tag/v0.5.5)
is specifically heads-up 200BB. Its
[model card](https://github.com/dberweger2017/CFR-NLHE-6max/blob/main/docs/releases/v0.5.5/MODEL_CARD.md)
describes a three-seed CFR average, public-history translation for off-menu
bets and downloadable frozen assets. The authors report **−19.67 BB/100**
with CI [−38.57, −0.76] over 30,000 Slumbot hands: **−196.7 mbb/hand**,
CI [−385.7, −7.6]. It loses to Slumbot; these are upstream results, not ours.

The release needs about 7.67 GB of downloads, roughly 25 GB of disk with its
derived play cache, and recommends a 16 GB machine. Its own license audit
records unresolved licensing for the upstream `pokers` engine, despite MIT
for the repository's code. Preserve that distinction; do not import the
engine as though all upstream components had a verified MIT grant.

Recommendation: a promising frozen strong-reference opponent, initially in
one resident process with bounded concurrency. Before any training integration,
verify the tagged asset hashes, adapter information boundary, engine rights,
fallback frequency and measured action latency. Do not download these large
assets into or run their initialization on the active training machine casually.

## Lightweight engineering candidate: GTO Poker Overlay

[Upstream](https://github.com/hellomate2/gto-poker-overlay) provides a C++ MCCFR
blueprint, a roughly 7 MB browser policy representation, and a
[JSON-lines policy service](https://github.com/hellomate2/gto-poker-overlay/blob/main/blueprint/README.md)
(`bp serve`). Headless blueprint play is the relevant component; no browser
automation or live-site integration is needed for our training.

The shipped model assumes **100BB**. Its own status section reports no external
Slumbot/GTO Wizard benchmark and a substantial LBR vulnerability. MIT is the
repository license; preserve third-party notices separately. The compact
policy size does not include every possible runtime memory cost.

Recommendation: convenient for a fast independent style/blueprint opponent,
but the shipped 100BB blueprint is **not** a faithful 200BB opponent. Use a
separate 100BB benchmark, or obtain/train a matching 200BB policy. Scaling
chip units cannot preserve both blinds and stack depth.

## Interesting but not cleared: Saheb/rl-poker

[Upstream](https://github.com/Saheb/rl-poker) implements a full CFR blueprint
plus Cython turn/river endgame solving and documents 200BB Slumbot experiments.
Its README reports broad small-sample CIs, and a larger run was still in
progress in the inspected version. The example uses 50,000 turn iterations
and 300,000 river iterations, so runtime cost needs measurement. Strategy
filenames are documented, but actual downloadable asset availability and an
explicit reuse license were not verified. Do not equate a public repository
with an unrestricted reuse grant.

Recommendation: inspect algorithms and evidence, pending license and asset
verification; not a first production training dependency.

## Why the familiar names are not immediate training bots

- [Slumbot2019](https://github.com/ericgjackson/slumbot2019) is MIT CFR/MCCFR
  infrastructure with abstractions, resolving and head-to-head tools. The
  inspected README gives training/build instructions, not a verified ready
  download of the website's strong policy. The online service chooses its
  own deals and opponent; it cannot simply act as a local opponent inside our
  PPO environment. Keep the working online adapter for external evaluation.
- [TexasSolver](https://github.com/bupticybee/TexasSolver) exposes a console
  solver and JSON strategy export, under AGPL-3.0. It needs ranges and a
  subgame setup, rather than supplying a ready all-street training bot.
  Useful later for sampled postflop verification or solver-generated policy
  data, with solve costs measured first.
- [Official ReBeL](https://github.com/facebookresearch/rebel) explicitly
  contains **Liar's Dice only**. Its published poker results do not mean the
  official repository provides a runnable HUNL opponent.

## Proposed next experiment, not a current configuration change

Start with the DecisionHoldem artifact/interface investigation. HU200 v0.5.5
is a second independent strong-reference candidate if its asset and dependency
requirements are acceptable. Measure them against our frozen students on the
same deals before calling them stronger. Track exact state replay, full raise
sizes, fallback/translation rates, illegal actions, per-action p50/p99 and
complete-match throughput. Never give an external actor the referee object,
opponent hole cards or future board.

For a later explicitly launched Stage 2 experiment, use an opt-in external
family with pinned code/assets, a resident process and bounded sampling share.
Timeout or invalid output should fail the experiment visibly, not silently
substitute check/call. Keep external held-out policies and deal seeds separate
from training. The current run and its nine-family configuration are unchanged.
