"""Shared evaluation harness: case loading, target adapters, timing, results.

Both runners (`strands_runner` and `langsmith_runner`) sit on top of this, so a
case is invoked identically whichever judge scores it. That is what makes the
side-by-side comparison in the dashboard meaningful: the only variable between
the two runners is the evaluator, not the way the system under test was driven.
"""
