# R08 plan

Status: **NOT_RUN**. The Teacher gate is **PENDING_TEACHER**.

This directory is a handoff stub for after Route 1 R07 finishes. It does not
start SD-CFR, distillation, encoder training, or PPO. It does not read AutoDL
outputs or Botzone upload paths. It does not edit `eval/r07` manifests,
including `a6_seed303`.

`--execute` on the stub scripts exits 2 before any job. Distill stays
unauthorized until a later change, outside this prep scaffold, records a real
Teacher pass. No mbb, chip, or AIVAT number is filled in here.

Intended later input, still empty:

- frozen R07 teacher out directory for A6 seed 303
- the published manifest path `eval/r07/manifests/a6_seed303.json` (not vendored
  by this branch)
- schema `botzone_a6_v2` with `quarter_pot` kept at width 6

GTO Wizard numbers, when they exist, come from the offline fixture tool in
`docs/gto_wizard_eval.md`. They are not a training reward.
