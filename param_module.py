#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Default parameters and state initialization for the pricing optimization app.

This module provides default values for demand parameters and results storage
used by the Streamlit application.
"""

# Default demand parameters for each demand type
demand_params = {
    "linear": {"a": 100, "b": -2},
    "quadratic": {"a": 50, "b": -1, "c": 0.01},
    "seasonal": {"a": 70, "b": -0.5, "period": 30},
    "shifted": {"a1": 60, "b1": -0.3, "a2": 40, "b2": -0.2, "shift_time": 50},
    "stepwise": {"thresholds": [20, 50, 80], "demands": [100, 50, 20, 5]},
    "combined": {
        "segments": [
            {"type": "quadratic", "params": {"a": 50, "b": -1, "c": 0.01}},
            {"type": "seasonal", "params": {"a": 70, "b": -0.5, "period": 30}},
        ],
        "time_thresholds": [25, 50]
    },
    "csv": {}
}

# Results storage (populated after simulation runs)
results = {}
