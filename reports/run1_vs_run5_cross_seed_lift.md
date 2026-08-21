# Why Run~1 gains (+62) and Run~5 loses (−15) on cross-seed lift

Lift $= \mathrm{sum\_max\_g} - \mathrm{seeds\_solved}$.
HF DSL~1 (`pipeline_hf_20260611_151047_run{1,5}_2104814_dsl1_fs500`):

| Run | seeds_solved | sum_max_g | lift |
|-----|-------------:|----------:|-----:|
| Run~1 | 65 | 127 | **+62** |
| Run~5 | 126 | 111 | **−15** |

## Program form (decision rule)

- **Locating**: terminal searches the grid for its target (`MOVE_TO`, `USE_TOOL*`, `CRAFT`, and on Run~5 `PICKUP_AUTO`).
- **Explicit**: layout-tied motion (`MOVE`, `TURN`, and on Run~5 `TURN_ONLY` / `WAIT`).
- **Mixed**: both nonempty.

Front-only actuators (`PICKUP`, `PICKUP_IF_PRESENT`, `CRAFT_AT`) do not by themselves set the form.

### Counts

| Form | Run~1 unique / $(t,s)$ | Run~5 unique / $(t,s)$ |
|------|------------------------:|------------------------:|
| Locating only | 0 / 0 | 37 / 92 |
| Mixed | 47 / 49 | 24 / 24 |
| Explicit only | 15 / 16 | 10 / 10 |
| **Total** | **62 / 65** | **71 / 126** |

### Instance-weighted mean $g$

| Form | Run~1 | Run~5 |
|------|------:|------:|
| Locating | — | 6.02 |
| Mixed | **9.39** | 3.88 |
| Explicit | 8.00 | 1.30 |

Unique strings with $g{=}10$: Run~1 **41/62**, Run~5 **3/71**.

## Locating form ≠ high $g$

Run~5 is 73% locating but locating mean $g\approx 6$. Locating only means “searches”; it does not mean the search is seed-robust.

### Run~5 `PICKUP_AUTO` (brittle)

Implementation:

1. Choose the **Manhattan-nearest** item cell (`argmin` of distances).
2. Walk with **`_manhattan_path` (X then Y)** — **no BFS, no obstacle check**.

Demo: `PICKUP_AUTO(GRASS)` (coverage $g{=}8$).

| seed | success |
|------|---------|
| 0–35 | yes |
| **40, 45** | **no** |

On seed 40: agent `(2,1)`, nearest grass `(2,7)`, stand `(2,6)`. Manhattan path length 5 passes through occupied cell `(2,5)`. Other grass cells have free neighbors; nearest-greedy + straight-line walk still fails.

Demo: `PICKUP_AUTO(WOOD); PICKUP_AUTO(IRON); CRAFT(AXE, WORKSHOP0)` → live $g{=}5$, fails on `{0,15,25,35,45}`.

### Run~1 `CRAFT` (planning; carries mixed programs)

Implementation: `_ensure` ingredients + `_bfs_path(..., blocked_mask)` to a workshop. CFG says “craft at current workshop”; impl is a mini-planner.

Demo ablation (same FunSearch terminals, synth-matched grids):

| Program | $g$ |
|---------|----:|
| `TURN(NORTH); MOVE(NORTH,ONE); CRAFT(BED)` | 10 |
| **`CRAFT(BED)` only** | **10** |
| `MOVE(SOUTH,SIX); MOVE(SOUTH,TWO); TURN(EAST); CRAFT(STICK)` | 10 |
| **`CRAFT(STICK)` only** | **10** |

Explicit scaffolding is not what transfers; divergent locating `CRAFT` is.

## Bottom line

| | Run~5 | Run~1 |
|--|-------|-------|
| Form label | more locating | more mixed |
| Locating mechanism | nearest + Manhattan (no BFS) | BFS `CRAFT` that gathers |
| Synth density | high (126) | low (65) |
| Story | many seed-tied locating wins | few wins, each high-$g$ |

More locating programs help only when the locator plans around the grid. Run~5’s locating mass is greedy nearest-pickup; Run~1’s high lift comes from planning `CRAFT` inside mixed strings.
