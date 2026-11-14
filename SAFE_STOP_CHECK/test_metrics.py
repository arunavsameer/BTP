"""Shim so `python test_metrics.py` still works from this folder."""

from __future__ import annotations

import unittest

import tests.test_metrics as _tests  # noqa: F401

if __name__ == "__main__":
    unittest.main(module=_tests)
