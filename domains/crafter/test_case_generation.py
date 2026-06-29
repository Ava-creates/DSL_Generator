"""Per-terminal test-case generator for the Crafter domain.

Each spec is a JSON object with the following keys::

    {
      "task_name":    <crafter achievement>,
      "test_type":    "positive" | "negative" | "edge",
      "max_steps":    <int>,
      "init_actions": [<int action code>, ...],   # optional bootstrap actions
      "scenario":     {<scenario dict>},           # REQUIRED world overrides
      "pass_check":   "<python expr referencing inventory_before/after, ...>",
      "arg_values":   {<func_arg_name>: <literal>, ...}  # optional
    }

The schema mirrors the Craft grid spec shape: the solve() template reads
``pass_check`` and ``arg_values`` off the same keys. Tests never depend on
procgen seeds; each spec must declare a non-empty ``scenario`` that fully
determines the starting world.

Fail-fast policy: every invalid input (missing prompt, no LLM, malformed
LLM output after all retries) raises rather than silently returning a
fallback spec.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from vllm import SamplingParams


_VALID_TEST_TYPES = {"positive", "negative", "edge"}
_DEFAULT_MAX_STEPS = 400

# Must match domains/crafter/scenario.py::_entity_class and Crafter's
# tile materials. Kept in sync manually; validation is authoritative.
_VALID_ENTITY_TYPES = {"zombie", "skeleton", "cow", "plant", "arrow", "fence"}
_VALID_TILE_MATERIALS = {
    "coal", "diamond", "furnace", "grass", "iron", "lava",
    "path", "sand", "stone", "table", "tree", "water",
}
_VALID_FACINGS = {"up", "down", "left", "right"}


def _validate_scenario(scenario: Dict[str, Any]) -> Optional[str]:
    """Return an error string if the scenario is malformed, else None.

    Catches the common failure modes we've seen from LLMs:
      * placing a tile material (e.g. ``table``, ``tree``) under ``entities``;
      * placing an entity type under ``tiles``;
      * unknown facings; non-list tiles/entities; malformed positions.
    """
    player = scenario.get("player")
    if player is not None and not isinstance(player, dict):
        return "scenario.player must be an object"
    if isinstance(player, dict):
        facing = player.get("facing")
        if facing is not None and str(facing).lower() not in _VALID_FACINGS:
            return (
                f"scenario.player.facing must be one of {sorted(_VALID_FACINGS)}; "
                f"got {facing!r}"
            )
        pos = player.get("pos")
        if pos is not None and (
            not isinstance(pos, (list, tuple)) or len(pos) != 2
            or not all(isinstance(v, int) for v in pos)
        ):
            return "scenario.player.pos must be [x, y] of two integers"
        inv = player.get("inventory")
        if inv is not None and not isinstance(inv, dict):
            return "scenario.player.inventory must be an object of item->count"

    tiles = scenario.get("tiles")
    if tiles is not None:
        if not isinstance(tiles, list):
            return "scenario.tiles must be a list"
        for i, tile in enumerate(tiles):
            if not isinstance(tile, dict):
                return f"scenario.tiles[{i}] must be an object"
            material = tile.get("material")
            if material not in _VALID_TILE_MATERIALS:
                return (
                    f"scenario.tiles[{i}].material={material!r} is not a valid tile; "
                    f"must be one of {sorted(_VALID_TILE_MATERIALS)}. Entities like "
                    f"zombie/cow/skeleton belong in scenario.entities, not scenario.tiles."
                )
            pos = tile.get("pos")
            if not isinstance(pos, (list, tuple)) or len(pos) != 2 \
                    or not all(isinstance(v, int) for v in pos):
                return f"scenario.tiles[{i}].pos must be [x, y] of two integers"

    entities = scenario.get("entities")
    if entities is not None:
        if not isinstance(entities, list):
            return "scenario.entities must be a list"
        for i, ent in enumerate(entities):
            if not isinstance(ent, dict):
                return f"scenario.entities[{i}] must be an object"
            etype = ent.get("type")
            if etype not in _VALID_ENTITY_TYPES:
                return (
                    f"scenario.entities[{i}].type={etype!r} is not a valid entity; "
                    f"must be one of {sorted(_VALID_ENTITY_TYPES)}. Tile materials "
                    f"like table/furnace/tree/stone belong in scenario.tiles, not "
                    f"scenario.entities."
                )
            pos = ent.get("pos")
            if not isinstance(pos, (list, tuple)) or len(pos) != 2 \
                    or not all(isinstance(v, int) for v in pos):
                return f"scenario.entities[{i}].pos must be [x, y] of two integers"

    clear = scenario.get("clear_area")
    if clear is not None:
        if not isinstance(clear, dict):
            return "scenario.clear_area must be an object"
        material = clear.get("material", "grass")
        if material not in _VALID_TILE_MATERIALS:
            return (
                f"scenario.clear_area.material={material!r} is not a valid tile; "
                f"must be one of {sorted(_VALID_TILE_MATERIALS)}"
            )
    return None


def _extract_json_object(text: str) -> str:
    if not text:
        return ""
    for marker in ("assistantfinal", "JSON.assistantfinal"):
        if marker in text:
            text = text.split(marker, 1)[1]
            break
    start = text.find("{")
    if start == -1:
        return text.strip()
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1].strip()
    return text[start:].strip()


def _render_prompt(
    *,
    template: str,
    func_name: str,
    description: str,
    func_args: str,
    env_description: str,
    domain_text: str,
    task_list: List[str],
    valid_actions: List[Dict[str, Any]],
    existing_cases: Optional[List[Dict[str, Any]]],
    positive_grids: int,
    negative_grids: int,
    edge_grids: int,
    codebase_text: str,
    init_check_failure: str,
) -> str:
    existing_block = json.dumps(existing_cases or [], indent=2)
    action_block = "\n".join(
        f"  {a.get('code')}: {a.get('name')} - {a.get('description','')}"
        for a in valid_actions
    )
    task_block = ", ".join(task_list)
    # Use plain string replacement (not str.format) because the template
    # contains JSON examples with literal '{'/'}' that would be misread as
    # format specifiers.
    replacements = {
        "{FUNC_NAME}": func_name,
        "{DESCRIPTION}": description,
        "{FUNC_ARGS}": func_args or "None",
        "{NLD}": env_description,
        "{DOMAIN_TEXT}": domain_text,
        "{TASKS}": task_block,
        "{ACTIONS}": action_block,
        "{EXISTING_CASES}": existing_block,
        "{POSITIVE_COUNT}": str(positive_grids),
        "{NEGATIVE_COUNT}": str(negative_grids),
        "{EDGE_COUNT}": str(edge_grids),
        "{CODEBASE_TEXT}": codebase_text,
        "{INIT_CHECK_FAILURE}": init_check_failure,
    }
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def _validate_spec(
    spec: Dict[str, Any],
    *,
    valid_tasks: List[str],
    valid_actions: set[int],
    require_test_type: bool,
) -> Optional[str]:
    if not isinstance(spec, dict):
        return "spec must be a JSON object"
    task = spec.get("task_name")
    if not task or task not in valid_tasks:
        return f"task_name '{task}' must be one of {valid_tasks}"
    if "pass_check" not in spec or not isinstance(spec["pass_check"], str):
        return "pass_check must be a Python expression string"
    if "max_steps" in spec and not isinstance(spec["max_steps"], int):
        return "max_steps must be an integer"
    if "init_actions" in spec:
        acts = spec["init_actions"]
        if not isinstance(acts, list) or any(
            not isinstance(a, int) or a not in valid_actions for a in acts
        ):
            return "init_actions must be a list of valid integer action codes"
    test_type = spec.get("test_type")
    if require_test_type:
        if test_type not in _VALID_TEST_TYPES:
            return f"test_type must be one of {_VALID_TEST_TYPES}"
    if "arg_values" in spec and not isinstance(spec["arg_values"], dict):
        return "arg_values must be an object"
    scenario = spec.get("scenario")
    if not isinstance(scenario, dict) or not scenario:
        return "scenario is required and must be a non-empty object"
    scenario_err = _validate_scenario(scenario)
    if scenario_err is not None:
        return scenario_err
    return None


def ensure_function_test_case(
    *,
    func_name: str,
    description: str,
    output_path: str,
    valid_tasks: List[str],
    valid_actions: List[Dict[str, Any]],
    env_description: str,
    domain_text: str,
    func_args: str = "",
    default_task_name: Optional[str] = None,
    shared_vllm: Any = None,
    prompt_path: str = "prompt_specifications/crafter_testcase_prompt.txt",
    attempts: int = 5,
    existing_cases: Optional[List[Dict[str, Any]]] = None,
    codebase_text: str = "",
    require_test_type: bool = True,
    skip_positive_grids: bool = False,
    positive_grids: int = 10,
    negative_grids: int = 4,
    edge_grids: int = 1,
) -> Dict[str, Any]:
    """LLM-based generation of a single Crafter test-case JSON.

    Raises on any failure (no LLM, missing prompt, invalid JSON after all
    retries) so problems surface immediately instead of silently writing a
    broken spec.
    """
    if shared_vllm is None:
        raise RuntimeError(
            "Crafter test-case generation requires a vLLM instance; "
            "`shared_vllm` is None."
        )
    if not valid_tasks:
        raise ValueError("No valid task names available for Crafter test-case generation.")

    valid_action_codes = {int(a["code"]) for a in valid_actions}
    task = default_task_name or valid_tasks[0]
    if task not in valid_tasks:
        raise ValueError(
            f"default_task_name '{task}' is not in the adapter's task list {valid_tasks}."
        )

    if not os.path.isfile(prompt_path):
        raise FileNotFoundError(
            f"Crafter test-case prompt template not found at '{prompt_path}'. "
            "Create it or pass a valid prompt_path."
        )
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    base_prompt = _render_prompt(
        template=template,
        func_name=func_name,
        description=description,
        func_args=func_args,
        env_description=env_description,
        domain_text=domain_text,
        task_list=valid_tasks,
        valid_actions=valid_actions,
        existing_cases=existing_cases,
        positive_grids=positive_grids,
        negative_grids=negative_grids,
        edge_grids=edge_grids,
        codebase_text=codebase_text,
        init_check_failure="",
    )
    params = SamplingParams(temperature=0.3, max_tokens=2000)

    last_error = ""
    last_candidate = ""
    for _ in range(max(1, int(attempts))):
        prompt = base_prompt
        if last_error:
            prompt += (
                f"\n\nThe previous candidate failed validation: {last_error}."
                " Regenerate a valid JSON spec."
            )
        output = shared_vllm.generate([prompt], sampling_params=params)
        raw = output[0].outputs[0].text.strip()
        last_candidate = raw
        json_text = _extract_json_object(raw)
        spec = json.loads(json_text)
        err = _validate_spec(
            spec,
            valid_tasks=valid_tasks,
            valid_actions=valid_action_codes,
            require_test_type=require_test_type,
        )
        if err is None:
            if skip_positive_grids and spec.get("test_type") == "positive":
                last_error = "positive grids disabled for this call"
                continue
            _write_spec(spec, output_path)
            return spec
        last_error = err

    raise RuntimeError(
        f"Failed to generate a valid Crafter test-case spec for '{func_name}' "
        f"after {attempts} attempts. Last error: {last_error}. "
        f"Last candidate:\n{last_candidate}"
    )


def _write_spec(spec: Dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
