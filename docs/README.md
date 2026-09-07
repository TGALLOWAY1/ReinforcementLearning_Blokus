# Documentation Index

Last updated: July 2026 (post-audit — see the root `AUDIT_REPORT.md`).

The root `README.md` explains what the repo does and how to run everything;
`AUDIT_REPORT.md` records the July 2026 cleanup, the maxⁿ MCTS fix, and the
training-plateau diagnosis. The directories below hold the surviving detailed
docs.

> ⚠️ Historical strength conclusions (layer experiments, KEY_FINDINGS-era
> verdicts, pre-July-2026 champion narratives) predate the maxⁿ reward fix and
> are **not valid** as evidence about which MCTS features help.

- **agent-strength-rebuild/DIAGNOSTIC_ASSESSMENT_2026-09-06.md** — the September 2026
  diagnostic assessment (findings, root cause, recommended path, milestones); read before
  any agent-strength work.
- **agent-strength-rebuild/** — **governing plan for all agent-strength work
  (July 2026 rescue): start at
  [`agent-strength-rebuild/MASTER_PLAN.md`](agent-strength-rebuild/MASTER_PLAN.md)**;
  supersedes `05-planning/CONTINUOUS_TRAINING_PLAN.md` for strategy.
- **00-overview/** — documentation map and orientation.
- **01-product/** — feature inventory and current-behavior notes.
- **02-architecture/** — engine / MCTS / webapi / frontend architecture.
- **03-implementation/** — implementation details, incl. nightly training
  (`TRAINING_AND_OVERNIGHT_RUNS.md`).
- **04-quality/** — known issues, risks, test strategy.
- **05-frontend/** — frontend design notes.
- **05-planning/** — prioritized TODO / next tasks. **Start here:**
  [`CONTINUOUS_TRAINING_PLAN.md`](05-planning/CONTINUOUS_TRAINING_PLAN.md) —
  why the nightly loop doesn't yet improve the agent and the prioritized plan
  to make it ratchet (handoff for the next session).
- **06-history/** — project history (context only).
- **07-ai-context/** — agent workflow and context-loading protocol.
- **08-visuals/** — diagrams and screenshots.

*To run the project, start at the root [README.md](../README.md).*
