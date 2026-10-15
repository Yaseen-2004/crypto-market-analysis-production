"""Build the data file the static HTML interfaces read.

Run after ``python main.py`` to refresh the pages when opening them from disk:

    python ui/build_data.py

Writes ``ui/analysis_data.js``, which ``volatility_deck.html`` (phone) and
``risk_room.html`` (big screen) load as a plain script - no fetch, so both pages
also work straight from the filesystem. The Django app (``webapp/``) builds the
same payload itself after every scheduled refresh and does not need this file.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.ui_payload import build_payload, to_script  # noqa: E402


def main() -> int:
    sources = (config.COIN_STATISTICS_CSV, config.ANALYSIS_DATASET_CSV, config.OUTLIER_REPORT_CSV)
    missing = [p.name for p in sources if not p.exists()]
    if missing:
        print(f"Missing pipeline output(s): {', '.join(missing)}. Run 'python main.py' first.")
        return 1
    payload = build_payload(
        pd.read_csv(config.COIN_STATISTICS_CSV),
        pd.read_csv(config.ANALYSIS_DATASET_CSV),
        pd.read_csv(config.OUTLIER_REPORT_CSV),
        refreshed_at=datetime.fromtimestamp(config.COIN_STATISTICS_CSV.stat().st_mtime, timezone.utc),
    )
    target = Path(__file__).resolve().parent / "analysis_data.js"
    target.write_text(to_script(payload), encoding="utf-8")
    meta = payload["meta"]
    print(f"Wrote {target.relative_to(ROOT).as_posix()} ({target.stat().st_size / 1024:.0f} KB): "
          f"{meta['coins']} coins, {meta['totalOutliers']} outliers, window ending {meta['windowEnd']} UTC.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
