"""Allow ``python -m dbs_eeg_sync`` to open the graphical launcher."""

from __future__ import annotations

from .gui_launcher import main

if __name__ == "__main__":
    raise SystemExit(main())
