"""Verification pipeline.

Stage modules are added in later phases. Runtime execution order is:

    Stage 1 -> Stage 2 -> Stage 3 -> Stage 5 -> Stage 4 -> Stage 6

Stage numbers are conceptual/rubric labels, not execution order. Stage 5
(source assessment) must run before Stage 4 (verification) because the
deterministic aggregation in Stage 4 consumes the source weights that Stage 5
produces.
"""
