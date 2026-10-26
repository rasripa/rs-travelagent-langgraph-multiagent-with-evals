"""The two evaluation runners.

Both drive the system through `evaluation.harness.targets.run_case` and write
the same result schema, so the dashboard can compare them directly. The only
intended difference between them is the judge.
"""
