"""Human-readable number formatting shared by charts, reports and the dashboard."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value)


def format_price(value) -> str:
    """$76,350 / $1.23 / $0.00001234 - keeps 4 significant digits for sub-dollar prices."""
    if _missing(value):
        return "n/a"
    value = float(value)
    if abs(value) >= 1000:
        return f"${value:,.0f}"
    if abs(value) >= 1:
        return f"${value:,.2f}"
    return "$" + np.format_float_positional(value, precision=4, unique=False, fractional=False, trim="-")


def compact_usd(value) -> str:
    """$1.53T / $850.20M / $12.40K."""
    if _missing(value):
        return "n/a"
    value = float(value)
    for divisor, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= divisor:
            return f"${value / divisor:,.2f}{suffix}"
    return f"${value:,.2f}"


def pct(value, digits: int = 2, signed: bool = True) -> str:
    if _missing(value):
        return "n/a"
    return f"{float(value):+.{digits}f}%" if signed else f"{float(value):.{digits}f}%"


def num(value, digits: int = 2) -> str:
    return "n/a" if _missing(value) else f"{float(value):.{digits}f}"


def interval_label(minutes: float) -> str:
    """'hourly' for ~60-minute data, otherwise e.g. '5-minute' / 'daily'."""
    if _missing(minutes) or minutes <= 0:
        return "periodic"
    if 50 <= minutes <= 70:
        return "hourly"
    if 1380 <= minutes <= 1500:
        return "daily"
    return f"{minutes:.0f}-minute"
