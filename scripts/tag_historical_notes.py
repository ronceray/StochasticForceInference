#!/usr/bin/env python3
# TODO: review this file
"""Prepend a "historical" banner to every ``design_notes_history/*.md`` file.

Idempotent: detects the banner marker and skips already-tagged files.

Usage
-----

   python scripts/tag_historical_notes.py            # apply
   python scripts/tag_historical_notes.py --dry-run  # preview

The banner reads:

    > ⚠ **Historical design note** — retained for traceability only.
    > May not reflect the current API. For current behaviour, see the
    > source code under ``SFI/`` and the user guides in
    > ``docs/source/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTES_DIR = ROOT / "design_notes_history"
MARKER = "<!-- historical-note-banner -->"
BANNER = (
    f"{MARKER}\n"
    "> ⚠ **Historical design note** — retained for traceability only.\n"
    "> May not reflect the current API. For current behaviour, see the\n"
    "> source code under `SFI/` and the user guides in `docs/source/`.\n"
    "\n"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not NOTES_DIR.is_dir():
        print(f"No such directory: {NOTES_DIR}")
        return 1

    tagged = 0
    skipped = 0
    for md in sorted(NOTES_DIR.glob("*.md")):
        if md.name.upper() == "INDEX.MD":
            skipped += 1
            continue
        text = md.read_text()
        if MARKER in text:
            skipped += 1
            continue
        new_text = BANNER + text
        action = "WOULD TAG" if args.dry_run else "TAG"
        print(f"  {action}  {md.relative_to(ROOT)}")
        if not args.dry_run:
            md.write_text(new_text)
        tagged += 1

    print(f"Tagged: {tagged}    Skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
