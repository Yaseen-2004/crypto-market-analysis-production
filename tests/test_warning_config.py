"""Guards the pytest warning configuration itself."""

import warnings

import pytest


def test_deprecations_from_our_own_code_are_errors():
    """`filterwarnings` in pyproject.toml must escalate DeprecationWarnings raised in src.*"""
    with pytest.raises(DeprecationWarning):
        warnings.warn_explicit("probe", DeprecationWarning, "src/probe.py", 1, module="src.probe")


def test_third_party_deprecations_do_not_fail_the_suite():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn_explicit("elsewhere", DeprecationWarning, "other/probe.py", 1, module="other.probe")
    assert len(caught) == 1
