"""Crafter observation helpers.

Crafter's ``info['semantic']`` is a uint8 array of tile ids covering the whole
world. This module converts that array into a symbolic *local grid* centered
on the agent, plus utilities to decode ids to names, so that:

* ``pass_check`` expressions can reference ``grid_before_cells`` /
  ``grid_after_cells`` the same way they do on Craft.
* ``render_state_markdown`` can draw a proper grid similar to Craft.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def semantic_id_to_name(env: Any) -> Dict[int, str]:
    """Return the full id → name mapping used by Crafter's ``semantic`` view.

    Works with either a :class:`CrafterEnvWrapper` or the raw ``crafter.Env``.
    Raises if the env has not been initialised (no ``_world`` / ``_sem_view``).
    """
    inner = getattr(env, "_env", env)
    world = getattr(inner, "_world", None)
    sem_view = getattr(inner, "_sem_view", None)
    if world is None or sem_view is None:
        raise AttributeError(
            "semantic_id_to_name requires an initialised crafter.Env with "
            "_world and _sem_view attributes."
        )

    id_to_name: Dict[int, str] = {}
    for name, idx in world._mat_ids.items():
        id_to_name[int(idx)] = "empty" if name is None else str(name)
    for cls, idx in sem_view._obj_ids.items():
        id_to_name[int(idx)] = cls.__name__.lower()
    return id_to_name


def facing_name(facing: Tuple[int, int]) -> str:
    mapping = {
        (0, -1): "up",
        (0, 1): "down",
        (-1, 0): "left",
        (1, 0): "right",
    }
    return mapping.get(tuple(facing), "unknown")


def local_grid_cells(
    env: Any,
    *,
    radius: int = 4,
) -> Dict[str, Any]:
    """Extract an ``(2r+1) x (2r+1)`` symbolic grid centered on the player.

    Returns a dict containing::

        {
          "cells":  List[List[str]],   # tile names, empty string for out-of-world
          "origin": [x0, y0],          # world coords of the top-left cell
          "player_pos": [px, py],      # world coords of the agent
          "player_local": [rx, ry],    # agent position within the local grid
          "facing":  "up"|"down"|"left"|"right"|"unknown",
        }
    """
    inner = getattr(env, "_env", env)
    info = getattr(env, "info", {}) or {}
    semantic = info.get("semantic")
    if semantic is None:
        semantic = inner._sem_view()

    semantic = np.asarray(semantic)
    id_to_name = semantic_id_to_name(env)

    pos = info.get("player_pos")
    if pos is None:
        pos = inner._player.pos
    pos = [int(v) for v in pos]
    px, py = pos

    r = int(radius)
    x0, y0 = px - r, py - r
    width, height = semantic.shape

    cells: List[List[str]] = []
    for j in range(2 * r + 1):
        row: List[str] = []
        for i in range(2 * r + 1):
            wx, wy = x0 + i, y0 + j
            if 0 <= wx < width and 0 <= wy < height:
                tile_id = int(semantic[wx, wy])
                if tile_id not in id_to_name:
                    raise KeyError(
                        f"Unknown crafter tile id {tile_id} at world ({wx},{wy}); "
                        f"known ids: {sorted(id_to_name)}"
                    )
                row.append(id_to_name[tile_id])
            else:
                row.append("")
        cells.append(row)

    facing = tuple(inner._player.facing)

    return {
        "cells": cells,
        "origin": [x0, y0],
        "player_pos": [px, py],
        "player_local": [r, r],
        "facing": facing_name(facing),
    }


def grid_to_markdown(
    grid: Dict[str, Any],
    *,
    include_header: bool = True,
) -> str:
    """Render a :func:`local_grid_cells` dict as a compact markdown grid.

    The agent cell is suffixed with a facing indicator (^ v < >).
    """
    cells = grid.get("cells") or []
    if not cells:
        return "(no grid available)"

    arrows = {"up": "^", "down": "v", "left": "<", "right": ">"}
    arrow = arrows.get(grid.get("facing", "unknown"), "*")
    player_local = grid.get("player_local")

    rendered: List[List[str]] = []
    col_widths: List[int] = [0] * len(cells[0])
    for j, row in enumerate(cells):
        out_row: List[str] = []
        for i, tile in enumerate(row):
            cell = tile if tile else "."
            if player_local and [i, j] == player_local:
                cell = f"{cell}{arrow}" if cell != "." else f"@{arrow}"
            out_row.append(cell)
            col_widths[i] = max(col_widths[i], len(cell))
        rendered.append(out_row)

    lines: List[str] = []
    if include_header:
        origin = grid.get("origin", [0, 0])
        lines.append(
            f"Local grid radius {len(cells) // 2} around player "
            f"(origin=({origin[0]},{origin[1]}), facing={grid.get('facing','?')}):"
        )
    for row in rendered:
        lines.append(" | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)


def find_nearest(
    grid: Dict[str, Any],
    target: str,
) -> Optional[Tuple[int, int, int]]:
    """Return ``(dx, dy, manhattan)`` for the nearest cell named ``target``.

    Offsets are relative to the player (``dx = wx - player_x``). Returns
    ``None`` when the target is not visible in the local grid.
    """
    cells = grid.get("cells") or []
    player_local = grid.get("player_local")
    if not cells or not player_local:
        return None
    pr, pc = player_local[1], player_local[0]
    best: Optional[Tuple[int, int, int]] = None
    for j, row in enumerate(cells):
        for i, tile in enumerate(row):
            if tile == target:
                dx, dy = i - pc, j - pr
                dist = abs(dx) + abs(dy)
                if best is None or dist < best[2]:
                    best = (dx, dy, dist)
    return best
