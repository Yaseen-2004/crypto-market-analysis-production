"""Entry point for the Cryptocurrency Market Volatility & Trend Analysis System.

    python main.py            # fetch fresh data and run the full analysis
    python main.py --help     # all options
"""

import sys

from src.main import main

if __name__ == "__main__":
    sys.exit(main())
