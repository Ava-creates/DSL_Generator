Craft Environment Quick Reference
=================================

What this covers
----------------
- How worlds are built (cookbook, tasks, scenarios).
- How to map inventory ids to item names.
- Grid tensor layout, agent position/direction.
- Actions and stepping.
- Pass checks/evaluation hooks used in downstream scripts.
- Snippets to inspect state quickly.

Cookbook and Item Index
-----------------------
- Loaded from `craft/resources/recipes.yaml` via `craft.cookbook.Cookbook`.
- Item ids are 1-based; 0 is reserved/invalid.
- Name <-> id:
  - `cookbook.index.index(name)` → id
  - `cookbook.index.get(id)` → name
  - `len(cookbook.index)` → n_kinds + 1 (because ids start at 1).
- Types:
  - `cookbook.environment`: set of ids for environment tiles.
  - `cookbook.primitives`: set of ids for primitive items.
  - `cookbook.recipes`: dict of output_id → {ingredient_id/count or meta}.
- Helpers:
  - `cookbook.primitives_for(goal_id)`: minimal primitives needed.
  - `cookbook.primitives_for_reward(goal_id)`: expanded counts including intermediates.

Environment Basics
------------------
- Envs are created through `craft.env_factory.EnvironmentFactory.sample_environment`.
- The runtime env is `CraftLab` (`craft/env.py`); current state is `env._current_state`.
- Key state fields:
  - `pos` (tuple[int, int]): agent (x, y)
  - `dir` (int): agent facing (implementation uses numeric; see Actions for mapping)
  - `inventory` (array-like[float]): counts per item id (aligned to cookbook ids)
  - `grid` (np.ndarray): world tensor shaped `(width, height, n_kinds)`
- Factory inputs (default in scripts):
  - `recipes_path`, `hints_path`, `env_type` (int), `max_steps`, `visualise`, `reuse_environments`, `custom_grid_path`.
  - Tasks are derived from hints (see `craft/env_factory.py`).
- Scenario:
  - If `custom_grid_path` provided, uses that JSON spec; else sampled by goal.
  - Scenario is attached to env as `env.scenario`, with `env.scenario.spec` available for custom metadata (e.g., `pass_check`).

Coordinate System and Grid
--------------------------
- Grid tensor: `grid[x, y, kind]` is 1 if item `kind` is present at `(x, y)`.
- Shape: `(width, height, n_kinds)`, where `n_kinds = len(cookbook.index) - 1`.
- Coordinate order: x is column, y is row.
- Agent position: `env._current_state.pos` (x, y).
- Agent direction: `env._current_state.dir` numeric; typical mapping aligns with action ids (DOWN=0, UP=1, LEFT=2, RIGHT=3) when facing is used.
- Converting grid cells to names (one-hot to readable) shown below in “Grid and Positions”.

Inventory Semantics
-------------------
- Type: list/array (often numpy) length `n_kinds`; index 0 unused.
- Entry i (1-based) corresponds to `cookbook.index.get(i)`; value is the count.
- Access raw: `inv = env._current_state.inventory`
- Map to names (non-zero only):
  ```python
  inv = env._current_state.inventory
  cb = env.world.cookbook
  named = [(cb.get(i), float(inv[i-1])) for i in range(1, len(cb)) if float(inv[i-1]) != 0.0]
  ```
- Full map (including zeros):
  ```python
  full = {cb.get(i): float(inv[i-1]) for i in range(1, len(cb))}
  ```

Grid and Positions
------------------
- `env._current_state.grid` shape: `(width, height, n_kinds)` one-hot per cell.
- `env._current_state.pos`: agent position.
- `env._current_state.dir`: facing direction.
- Building readable cell names (like in `smth.py`):
  ```python
  grid = env._current_state.grid
  cb = env.world.cookbook
  names = []
  for y in range(grid.shape[1]):
      row = []
      for x in range(grid.shape[0]):
          cell = grid[x, y]
          indices = [i for i, v in enumerate(cell) if v]
          row.append(cb.get(indices[0]).lower() if indices else "")
      names.append(row)
  ```

Actions
-------
- `env.action_specs()` returns ids for `DOWN=0, UP=1, LEFT=2, RIGHT=3, USE=4`.
- Step: `reward, done, obs = env.step(action_id)`.

Pass Checks (used in smth.py)
-----------------------------
- `env.scenario.spec` may include `pass_check` string evaluated against:
  - `pos_before/pos_after`
  - `inventory_before/inventory_after`
  - `grid_before_cells/grid_after_cells`
  - `actions_count`, `dir_before/dir_after`
- `pass_check` is a Python expression; ensure values are sanitized if using untrusted specs.

Handy Inventory Printer
-----------------------
```python
def describe_inventory(env):
    inv = env._current_state.inventory
    cb = env.world.cookbook
    return {cb.get(i): float(inv[i-1]) for i in range(1, len(cb))}
```

Handy State Snapshot
--------------------
```python
def snapshot(env):
    state = env._current_state
    cb = env.world.cookbook
    return {
        "pos": state.pos,
        "dir": state.dir,
        "inventory": {cb.get(i): float(state.inventory[i-1]) for i in range(1, len(cb))},
        "grid_shape": getattr(state.grid, "shape", None),
    }
```

Minimal Run Example
-------------------
```python
from craft import env_factory

env = env_factory.EnvironmentFactory(
    "craft/resources/recipes.yaml",
    "craft/resources/hints.yaml",
    7,
    max_steps=300,
    visualise=False,
    reuse_environments=False,
).sample_environment(task_name="make[stick]")

env.reset()
print(snapshot(env))
reward, done, _ = env.step(env.action_specs()["RIGHT"])
print("reward", reward, "done", done)
```

