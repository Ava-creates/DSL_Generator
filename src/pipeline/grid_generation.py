#!/usr/bin/env python3
"""
Utilities for generating and validating Craft grid JSON specs via LLM.
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
from typing import Dict, Optional, Tuple

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None

try:
    from craft.cookbook import Cookbook
    from craft.craft import WIDTH, HEIGHT, validate_grid_spec
except Exception:
    Cookbook = None
    WIDTH, HEIGHT = 12, 12


def _get_grid_size() -> Tuple[int, int]:
    return int(WIDTH), int(HEIGHT)


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
    end = text.find("}", start)
    if end == -1:
        return text[start:].strip()
    # Expand to full JSON object by tracking braces from the first '{'
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1].strip()
    return text[start:].strip()


def _load_prompt_template(prompt_path: str) -> str:
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return (
            "Return ONLY valid JSON for a Craft grid spec with keys:\n"
            "task_name, width, height, include_boundary, init_pos, init_dir, grid.\n"
            "Output JSON only."
        )


def _extract_allowed_arg_values_from_cfg(func_args: str, cfg_text: str) -> dict[str, set[str]]:
    """Build allowed terminal values per argument symbol from CFG text.

    Example:
      func_args: "craftable:float, tool:str"
      cfg_text contains:
        CRAFTABLE ::= AXE | BOW
        TOOL ::= AXE | HAMMER
      returns:
        {"craftable": {"AXE", "BOW"}, "tool": {"AXE", "HAMMER"}}
    """
    if not func_args or not cfg_text:
        return {}

    arg_names = []
    for raw in func_args.split(","):
        token = raw.strip()
        if not token:
            continue
        name = token.split(":", 1)[0].strip()
        if name:
            arg_names.append(name)

    if not arg_names:
        return {}

    grammar_map: dict[str, set[str]] = {}
    rule_pattern = re.compile(r"^([A-Z_][A-Z0-9_]*)\s*::=\s*(.*)$")

    def _record_rhs_values(symbol: str, rhs: str) -> None:
        if not symbol or rhs is None:
            return
        values = grammar_map.setdefault(symbol, set())
        for part in rhs.split("|"):
            v = part.strip().strip("'").strip('"')
            if not v:
                continue
            if (
                re.fullmatch(r"[A-Z_][A-Z0-9_]*", v)
                or re.fullmatch(r"-?\d+", v)
                or re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)", v)
            ):
                values.add(v)

    current_symbol: str | None = None
    for raw_line in cfg_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = rule_pattern.match(line)
        if m:
            current_symbol = m.group(1).strip()
            _record_rhs_values(current_symbol, m.group(2).strip())
            continue

        if current_symbol and line.startswith("|"):
            continuation_rhs = re.sub(r"^\|\s*", "", line)
            _record_rhs_values(current_symbol, continuation_rhs)
            continue

        current_symbol = None

    allowed: dict[str, set[str]] = {}
    for arg in arg_names:
        symbol = arg.upper()
        if symbol in grammar_map:
            allowed[arg] = grammar_map[symbol]
    return allowed


def _validate_arg_values_against_cfg(arg_values: dict, allowed_map: dict[str, set[str]]) -> tuple[bool, str]:
    """Validate arg_values are in per-arg allowed terminal set from CFG."""
    if not allowed_map:
        return True, ""
    if not isinstance(arg_values, dict):
        return False, "arg_values must be a JSON object"

    for arg_name, allowed in allowed_map.items():
        if arg_name not in arg_values:
            return False, f"arg_values missing required key '{arg_name}'"
        raw = arg_values[arg_name]
        candidate = str(raw).strip().upper()
        if candidate not in allowed:
            allowed_sorted = ", ".join(sorted(allowed))
            return (
                False,
                f"arg_values.{arg_name}='{raw}' is invalid; allowed values from CFG are: {allowed_sorted}",
            )
    return True, ""


def _render_prompt(
    template: str,
    func_name: str,
    description: str,
    func_args: str,
    env_description: str,
    recipes_text: str,
    width: int,
    height: int,
    valid_items: list[str],
    valid_tasks: list[str],
    existing_cases: list[Dict] | None = None,
) -> str:
    if existing_cases:
        summaries = []
        for i, case in enumerate(existing_cases):
            parts = [f"Case {i}: task_name={case.get('task_name', '?')}"]
            if case.get("arg_values"):
                parts.append(f"arg_values={json.dumps(case['arg_values'])}")
            if case.get("init_pos"):
                parts.append(f"init_pos={case['init_pos']}")
            if case.get("init_dir") is not None:
                parts.append(f"init_dir={case['init_dir']}")
            if case.get("inventory"):
                parts.append(f"inventory={json.dumps(case['inventory'])}")
            if case.get("pass_check"):
                parts.append(f"pass_check={case['pass_check']}")
            summaries.append(", ".join(parts))
        existing_text = (
            "The following test cases have ALREADY been generated. "
            "Generate a NEW test case that is DIFFERENT from all of these — "
            "use different arg_values, init_pos, init_dir, or grid layout:\n"
            + "\n".join(summaries)
        )
    else:
        existing_text = ""

    replacements = {
        "<<FUNCTION_NAME>>": func_name,
        "<<DESCRIPTION>>": description,
        "<<ARGUMENTS>>": func_args,
        "<<ENV_DESCRIPTION>>": env_description,
        "<<RECIPES>>": recipes_text,
        "<<WIDTH>>": str(width),
        "<<HEIGHT>>": str(height),
        "<<VALID_ITEMS>>": ", ".join(sorted(valid_items)),
        "<<VALID_TASKS>>": ", ".join(sorted(valid_tasks)),
        "<<EXISTING_CASES>>": existing_text,
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def _get_cookbook(recipes_path: str) -> Optional[Cookbook]:
    if Cookbook is None:
        return None
    try:
        return Cookbook(recipes_path)
    except Exception as e:
        print(f"  Warning: Could not load cookbook from {recipes_path}: {e}")
        return None


def _get_item_names(cookbook: Cookbook) -> set[str]:
    items = set()
    for idx in cookbook.environment:
        items.add(cookbook.index.get(idx))
    for idx in cookbook.primitives:
        items.add(cookbook.index.get(idx))
    for idx in cookbook.recipes.keys():
        items.add(cookbook.index.get(idx))
    return {str(item).strip().lower() for item in items if item}


def _get_task_names(cookbook: Cookbook) -> list[str]:
    primitives = {cookbook.index.get(idx) for idx in cookbook.primitives}
    recipes = {cookbook.index.get(idx) for idx in cookbook.recipes.keys()}
    primitives = {p for p in primitives if p}
    recipes = {r for r in recipes if r}
    tasks = {f"get[{p}]" for p in primitives}
    tasks.update({f"make[{p}]" for p in primitives})
    tasks.update({f"make[{r}]" for r in recipes})
    return sorted(tasks)


def _validate_task_name(task_name: str, cookbook: Cookbook, default_task: Optional[str]) -> str:
    if not task_name or not isinstance(task_name, str):
        return default_task or ""
    match = re.match(r"^(get|make)\[([^\]]+)\]$", task_name.strip())
    if not match:
        return default_task or ""
    action, item = match.group(1), match.group(2).strip().lower()
    primitives = {cookbook.index.get(idx) for idx in cookbook.primitives}
    recipes = {cookbook.index.get(idx) for idx in cookbook.recipes.keys()}
    primitives = {p.lower() for p in primitives if p}
    recipes = {r.lower() for r in recipes if r}
    if action == "get":
        return f"get[{item}]" if item in primitives else (default_task or "")
    if action == "make":
        return f"make[{item}]" if item in recipes or item in primitives else (default_task or "")
    return default_task or ""


def _normalize_grid_spec(
    spec: Dict,
    width: int,
    height: int,
    valid_items: set[str],
    cookbook: Cookbook,
    default_task_name: str,
) -> Dict:
    normalized = {}
    grid = spec.get("grid")
    if not isinstance(grid, list) or len(grid) != height:
        grid = [["" for _ in range(width)] for _ in range(height)]
    normalized_grid = []
    for row in grid:
        if not isinstance(row, list) or len(row) != width:
            row = ["" for _ in range(width)]
        norm_row = []
        for cell in row:
            if cell is None or cell == "":
                norm_row.append("")
                continue
            name = str(cell).strip().lower()
            norm_row.append(name if name in valid_items else "")
        normalized_grid.append(norm_row)
    normalized["grid"] = normalized_grid
    normalized["width"] = width
    normalized["height"] = height
    normalized["include_boundary"] = bool(spec.get("include_boundary", True))

    init_pos = spec.get("init_pos", [1, 1])
    if not isinstance(init_pos, (list, tuple)) or len(init_pos) != 2:
        init_pos = [1, 1]
    try:
        x = int(init_pos[0])
        y = int(init_pos[1])
    except Exception:
        x, y = 1, 1
    x = max(1, min(width - 2, x))
    y = max(1, min(height - 2, y))
    normalized["init_pos"] = [x, y]

    init_dir = spec.get("init_dir", 0)
    try:
        init_dir = int(init_dir)
    except Exception:
        init_dir = 0
    if init_dir not in [0, 1, 2, 3]:
        init_dir = 0
    normalized["init_dir"] = init_dir

    task_name = spec.get("task_name")
    normalized["task_name"] = _validate_task_name(task_name, cookbook, default_task_name)
    raw_inventory = spec.get("inventory", {})
    normalized_inventory = {}
    if isinstance(raw_inventory, dict):
        for name, count in raw_inventory.items():
            if name is None:
                continue
            item = str(name).strip().lower()
            if item not in valid_items:
                continue
            try:
                count_val = int(count)
            except Exception:
                continue
            if count_val > 0:
                normalized_inventory[item] = count_val
    normalized["inventory"] = normalized_inventory
    pass_check = spec.get("pass_check")
    if isinstance(pass_check, str) and pass_check.strip():
        normalized["pass_check"] = pass_check.strip()
    raw_arg_values = spec.get("arg_values", {})
    if isinstance(raw_arg_values, dict):
        normalized["arg_values"] = raw_arg_values
    else:
        normalized["arg_values"] = {}
    return normalized


def ensure_function_grid_spec(
    func_name: str,
    description: str,
    recipes_path: str,
    output_path: str,
    shared_vllm=None,
    default_task_name: Optional[str] = None,
    prompt_path: str = "prompt_specifications/grid_prompt.txt",
    func_args: str = "",
    env_description: str = "",
    recipes_text: str = "",
    attempts: int = 3,
    existing_cases: list[Dict] | None = None,
    cfg_text: str = "",
) -> Optional[Dict]:
    cookbook = _get_cookbook(recipes_path)
    if cookbook is None:
        return None

    valid_items = _get_item_names(cookbook)
    valid_tasks = _get_task_names(cookbook)
    width, height = _get_grid_size()

    allowed_arg_values = _extract_allowed_arg_values_from_cfg(func_args or "", cfg_text or "")

    if shared_vllm is not None and SamplingParams is not None:
        template = _load_prompt_template(prompt_path)
        base_prompt = _render_prompt(
            template=template,
            func_name=func_name,
            description=description,
            func_args=func_args or "None",
            env_description=env_description or "",
            recipes_text=recipes_text or "",
            width=width,
            height=height,
            valid_items=list(valid_items),
            valid_tasks=valid_tasks,
            existing_cases=existing_cases,
        )
        print(f"[grid_generation] Prompt for {func_name}:\n{base_prompt}\n")
        print(f"[grid_generation] Args for {func_name}: {func_args or 'None'}")
        params = SamplingParams(temperature=0.2, max_tokens=5000)
        attempts = max(1, int(attempts))
        last_error = ""
        for attempt in range(attempts):
            prompt = base_prompt
            if last_error:
                prompt = (
                    base_prompt
                    + "\n\nThe previous grid spec was invalid because: "
                    + last_error
                    + "\nRegenerate a valid JSON spec."
                )
            output = shared_vllm.generate([prompt], sampling_params=params)
            raw = output[0].outputs[0].text.strip()
            print(f"[grid_generation] Raw LLM output for {func_name}:\n{raw}\n")
            json_text = _extract_json_object(raw)
            print("json_text", json_text)
            spec = {}
            try:
                spec = json.loads(json_text)
            except Exception as e:
                try:
                    spec = ast.literal_eval(json_text)
                except Exception:
                    print(f"  Warning: grid JSON parse failed for {func_name}: {e}")
                    spec = {}
            if not isinstance(spec, dict):
                last_error = "response is not a JSON object"
                continue
            required_keys = {"task_name", "width", "height", "grid"}
            if not required_keys.issubset(spec.keys()):
                last_error = f"missing required keys: {sorted(required_keys - set(spec.keys()))}"
                continue
            normalized = _normalize_grid_spec(
                spec=spec if isinstance(spec, dict) else {},
                width=width,
                height=height,
                valid_items=valid_items,
                cookbook=cookbook,
                default_task_name=default_task_name,
            )
            args_ok, args_error = _validate_arg_values_against_cfg(
                normalized.get("arg_values", {}),
                allowed_arg_values,
            )
            if not args_ok:
                last_error = args_error
                continue
            print(normalized)
            is_valid = True
            if 'validate_grid_spec' in globals():
                is_valid, last_error = validate_grid_spec(normalized, cookbook)
            if is_valid:
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(normalized, f, indent=2)
                return normalized
    else:
        raise RuntimeError("shared_vllm/SamplingParams unavailable; cannot generate grid spec")

    reason = last_error or "unknown validation error"
    print(
        f"  Warning: Grid spec invalid for {func_name} after {attempts} attempts: {reason}"
    )
    return None


