"""Apply a custom Crafter *scenario* on top of a freshly-reset env.

Crafter is procedurally generated, so tests that rely on specific world
content (e.g. ``defeat_zombie`` needs a zombie nearby) would be flaky on a
pure seed. Instead, we let users attach a scenario dict to their test case
that the wrapper applies right after :py:meth:`CrafterEnvWrapper.reset`.

Schema (all keys optional)::

    {
      "player": {
        "pos":      [x, y],               # move agent to this tile
        "facing":   "up"|"down"|"left"|"right",
        "inventory": {"wood": 2, "wood_sword": 1, ...},
        "health":   9, "food": 9, "drink": 9, "energy": 9
      },
      "clear_area": {
        "center": [x, y],                 # default: player pos after override
        "radius": 2,                      # make these tiles plain grass
        "material": "grass"
      },
      "tiles": [
        {"pos": [x, y], "material": "tree"},
        {"pos": [x, y], "material": "coal"},
        ...
      ],
      "entities": [
        {"type": "zombie",   "pos": [x, y]},   # relative to player if "relative": true
        {"type": "cow",      "pos": [1, 0], "relative": true},
        {"type": "skeleton", "pos": [x, y]},
        {"type": "plant",    "pos": [x, y]},
      ],
      "remove_entities_within": 6            # clear all spawned mobs within N tiles
                                              # of the player before adding new ones
    }

The wrapper calls :func:`apply_scenario(env, scenario)` after ``env.reset()``
and before any ``init_actions`` are replayed.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

_FACING_VECTORS = {
    "up":    (0, -1),
    "down":  (0, 1),
    "left":  (-1, 0),
    "right": (1, 0),
}


def _inner(env: Any) -> Any:
    return getattr(env, "_env", env)


def _entity_class(name: str):
    import crafter.objects as O

    mapping = {
        "zombie":   O.Zombie,
        "skeleton": O.Skeleton,
        "cow":      O.Cow,
        "plant":    O.Plant,
        "arrow":    O.Arrow,
        "fence":    O.Fence,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unknown Crafter entity type: {name}")
    return mapping[key]


def _set_tile(world: Any, pos: Tuple[int, int], material: str) -> None:
    world[tuple(int(v) for v in pos)] = material


def _resolve_pos(
    pos: Iterable[int],
    *,
    relative: bool,
    player_pos: Tuple[int, int],
) -> Tuple[int, int]:
    x, y = int(pos[0]), int(pos[1])
    if relative:
        x += int(player_pos[0])
        y += int(player_pos[1])
    return x, y


def _safe_move_player(inner: Any, new_pos: Tuple[int, int]) -> None:
    """Move the player object to ``new_pos`` without tripping the obj_map asserts."""
    player = inner._player
    world = inner._world
    if tuple(player.pos) == tuple(new_pos):
        return
    # Clear the destination cell if some other object is there.
    occupant = world._objects[world._obj_map[tuple(new_pos)]]
    if occupant is not None and occupant is not player:
        world.remove(occupant)
    world.move(player, np.array(new_pos))


def _clear_entities_within(inner: Any, center: Tuple[int, int], radius: int) -> None:
    import crafter.objects as O

    player = inner._player
    world = inner._world
    keep = (O.Player,)
    for obj in list(world.objects):
        if isinstance(obj, keep):
            continue
        if obj is player:
            continue
        dx = int(obj.pos[0]) - int(center[0])
        dy = int(obj.pos[1]) - int(center[1])
        if abs(dx) <= radius and abs(dy) <= radius:
            world.remove(obj)


def apply_scenario(env: Any, scenario: Optional[Dict[str, Any]]) -> None:
    """Mutate ``env`` in place according to ``scenario``.

    No-op when ``scenario`` is falsy. Assumes ``env.reset()`` has already run.
    """
    if not scenario:
        return

    inner = _inner(env)
    world = inner._world
    player = inner._player

    # Player overrides first, so relative positions are anchored correctly.
    player_cfg = scenario.get("player") or {}
    if "pos" in player_cfg:
        _safe_move_player(inner, tuple(int(v) for v in player_cfg["pos"]))
    if "facing" in player_cfg:
        key = str(player_cfg["facing"]).lower()
        if key not in _FACING_VECTORS:
            raise ValueError(
                f"scenario.player.facing must be one of "
                f"{sorted(_FACING_VECTORS)}; got {player_cfg['facing']!r}."
            )
        player.facing = _FACING_VECTORS[key]
    inventory = player_cfg.get("inventory") or {}
    for item, count in inventory.items():
        player.inventory[item] = int(count)
    for stat in ("health", "food", "drink", "energy"):
        if stat in player_cfg:
            player.inventory[stat] = int(player_cfg[stat])
            if stat == "health":
                inner._last_health = int(player_cfg[stat])

    player_pos = tuple(int(v) for v in player.pos)

    clear_cfg = scenario.get("clear_area")
    if clear_cfg:
        center = tuple(int(v) for v in clear_cfg.get("center", player_pos))
        radius = int(clear_cfg.get("radius", 2))
        material = str(clear_cfg.get("material", "grass"))
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                pos = (center[0] + dx, center[1] + dy)
                if 0 <= pos[0] < world.area[0] and 0 <= pos[1] < world.area[1]:
                    _set_tile(world, pos, material)

    for tile in scenario.get("tiles") or []:
        pos = _resolve_pos(
            tile["pos"],
            relative=bool(tile.get("relative", False)),
            player_pos=player_pos,
        )
        if not (0 <= pos[0] < world.area[0] and 0 <= pos[1] < world.area[1]):
            raise ValueError(
                f"Tile position {pos} is out of bounds for world of "
                f"size {tuple(world.area)}."
            )
        _set_tile(world, pos, str(tile["material"]))

    remove_radius = scenario.get("remove_entities_within")
    if remove_radius is not None:
        _clear_entities_within(inner, player_pos, int(remove_radius))

    for ent in scenario.get("entities") or []:
        pos = _resolve_pos(
            ent["pos"],
            relative=bool(ent.get("relative", False)),
            player_pos=player_pos,
        )
        if not (0 <= pos[0] < world.area[0] and 0 <= pos[1] < world.area[1]):
            raise ValueError(
                f"Entity position {pos} is out of bounds for world of "
                f"size {tuple(world.area)}."
            )
        if tuple(pos) == tuple(player.pos):
            raise ValueError(
                f"Entity position {pos} collides with the player. "
                "Use a different offset or move the player first."
            )
        existing = world._objects[world._obj_map[tuple(pos)]]
        if existing is not None:
            world.remove(existing)
        cls = _entity_class(ent["type"])
        import crafter.objects as O

        if cls in (O.Zombie, O.Skeleton):
            obj = cls(world, np.array(pos), player)
        else:
            obj = cls(world, np.array(pos))
        world.add(obj)
