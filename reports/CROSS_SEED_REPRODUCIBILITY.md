# Cross-seed reproducibility

## Run12 / appendix 192 vs re-eval 193

The Run~7 appendix reported `sum_max_g = 192` with `make[axe]` `max_g = 10`.
Re-evaluation under the same outcomes, CFG, final_functions, and synthesis grids
gives `sum_max_g = 193` with `make[axe]` `max_g = 9` (fails seed 15).

**Root cause (axe):** after collecting wood+iron, `NAVIGATE_TO_WORKSHOP` returns
`[]` because workshop0 is unreachable on empty cells. `CraftState.step` cannot
walk into occupied cells (`if self.grid[n_x, n_y, :].any(): stay`). Seed 15 was
never solved in synthesis for `make[axe]`. Claiming g=10 required an impossible
success on that seed.

**Conclusion:** use the re-eval matrix (193), not appendix 192. Other runs already
share this re-eval pipeline (2026-08-21).

## Eval rules

1. `scripts/eval_dsl1_cross_seed.py` reuses existing coverage JSON unless `--force`
2. `PYTHONHASHSEED=0`, `reuse_environments=True`, appendix `norm_prog` uniqueness
3. Timeout / out-of-steps = failure
