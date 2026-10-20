# GitHub Copilot instructions

This file is the entry point GitHub Copilot reads automatically.

**The authoritative agent guide for this repository is
[`AGENTS.md`](../AGENTS.md).** Read it before making any code change.

## Copilot-specific notes

- **Repository memory**: `/memories/repo/*.md` contains durable
  architecture / API notes accumulated across sessions. Consult them
  when you need deeper context than `AGENTS.md` provides. Notable
  entries (non-exhaustive):
  - `SFI_core_api_thorough.md` — full public API inventory.
  - `integrate_framework_thorough_exploration.md` — integrate engine.
  - `overdamped_strang_thorough_exploration.md`,
    `strang_band_implementation.md`,
    `strang_complete_inventory_v10.md` — Strang estimator internals.
  - `mask_handling_thorough_exploration.md` — masked trajectories.
  - `deprecated_api_cleanup_status.md` — which methods/files were
    removed or renamed (avoid reintroducing them).
- **Session memory**: `/memories/session/` holds per-conversation
  plans. If you resume work, check there first.
- **Skills loaded by default**: `agent-customization`,
  `get-search-view-results`, plus the GitHub-PR skills. Use them when
  applicable.

## Required reading (in order)

1. [`AGENTS.md`](../AGENTS.md) — canonical imports, playbook links,
   "do not re-implement" list.
2. Playbook matching the task
   ([`docs/source/agent_playbooks/`](../docs/source/agent_playbooks/)).
3. [`GALLERY_STYLE_GUIDE.md`](../GALLERY_STYLE_GUIDE.md) (plotting
   tasks).
