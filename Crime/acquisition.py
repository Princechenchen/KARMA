"""Frozen method registry and paired discrete-LHS acquisition designs."""
from __future__ import annotations

import numpy as np

from region_recommender import maximin_lhs_indices

CORE_METHODS = ["C1", "C2", "C12", "C10", "ROI-D", "C6", "C7", "KARMA"]
OPTIONAL_METHODS = ["C3", "C4", "C5", "C9", "C11"]
METHODS = CORE_METHODS + OPTIONAL_METHODS
EXPERIMENTAL_METHODS = ["C10-Approx", "KARMA-HG", "KARMA-BND", "KARMA-Fixed"]


def parse_methods(text):
    value = str(text).strip()
    if value.lower() == "core":
        return CORE_METHODS.copy()
    if value.lower() == "all":
        return METHODS.copy()
    methods = [item.strip() for item in value.split(",") if item.strip()]
    unknown = set(methods).difference(METHODS + EXPERIMENTAL_METHODS)
    if unknown or not methods:
        raise ValueError(f"unknown or empty methods: {sorted(unknown)}")
    if len(set(methods)) != len(methods):
        raise ValueError("duplicate methods are not allowed")
    return methods


def lhs_sequence(units, eligible, seed, total=100):
    eligible = np.asarray(eligible, dtype=int).reshape(-1)
    if len(np.unique(eligible)) != len(eligible):
        raise ValueError("eligible indices must be unique")
    if not total <= len(eligible):
        raise ValueError("not enough eligible cells for the requested budget")
    xy = units[["x", "y"]].to_numpy(float)
    local = maximin_lhs_indices(xy[eligible], total, seed)
    return eligible[local]


def initial_design(units, eligible, seed, n=20):
    return lhs_sequence(units, eligible, seed, total=100)[:n]
