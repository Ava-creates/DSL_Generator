#!/usr/bin/env python3
"""
Utilities for generating and validating Craft grid JSON specs via LLM.
"""

from __future__ import annotations

import ast
import json
import os
import re
from typing import Dict, Optional, Tuple

from src.pipeline.cfg_symbol_utils import (
    DEFAULT_EXCLUDED_SYMBOLS,
    build_cfg_rule_map,
    expand_symbol_to_terminals,
    _clean_token,
)

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


def _json_for_log(obj) -> str:
    """Serialize a grid spec for failure logs; never raises on Ellipsis etc."""
    if isinstance(obj, str):
        return obj.strip()
    return json.dumps(obj, ensure_ascii=True, default=repr)


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
    except Exception as e:
        raise RuntimeError(
            f"Failed to load grid prompt template '{prompt_path}': {e}. "
            "Stopping stage to avoid using an unintended fallback prompt."
        ) from e


def _parse_func_arg_names(func_args: str) -> list[str]:
    """Parse 'integer:int, item:str' -> ['integer', 'item']."""
    names: list[str] = []
    for raw in (func_args or "").split(","):
        token = raw.strip()
        if not token or token.lower() == "none":
            continue
        name = token.split(":", 1)[0].strip()
        if name:
            names.append(name)
    return names


def _referenced_cfg_symbols(rule_map: dict[str, list[str]]) -> set[str]:
    """Symbols that appear on the RHS of any rule."""
    referenced: set[str] = set()
    token_re = re.compile(r"^[A-Z_][A-Z0-9_]*$")
    for rhs_list in rule_map.values():
        for part in rhs_list:
            for raw in re.split(r"\s*\|\s*|\s+", part.strip()):
                tok = _clean_token(raw).upper()
                if tok and token_re.match(tok) and tok not in DEFAULT_EXCLUDED_SYMBOLS:
                    referenced.add(tok)
    return referenced


def _unbound_cfg_symbols(rule_map: dict[str, list[str]]) -> set[str]:
    """RHS symbols with no defining ::= rule (e.g. INTEGER referenced by STEPS ::= INTEGER)."""
    return _referenced_cfg_symbols(rule_map) - set(rule_map.keys())


def _symbol_has_cfg_enumeration(symbol: str, rule_map: dict[str, list[str]]) -> bool:
    """True when the CFG defines concrete terminal values for this symbol."""
    clean = _clean_token(symbol).upper()
    if clean not in rule_map:
        return False

    alts: list[str] = []
    for part in rule_map[clean]:
        for alt in part.split("|"):
            tok = _clean_token(alt).upper()
            if tok and tok not in DEFAULT_EXCLUDED_SYMBOLS:
                alts.append(tok)
    if not alts:
        return False

    unbound = _unbound_cfg_symbols(rule_map)
    # Alias to an undefined non-terminal (e.g. STEPS ::= INTEGER) is not an enumeration.
    if len(alts) == 1 and alts[0] in unbound:
        return False

    if all(tok not in rule_map for tok in alts):
        return True

    expanded = expand_symbol_to_terminals(clean, rule_map)
    concrete = {t.upper() for t in expanded if t.upper() not in rule_map}
    return len(concrete) > 0


def _build_arg_schema(func_args: str, cfg_text: str) -> str:
    """Human-readable schema for the grid LLM (keys + allowed values from CFG)."""
    arg_names = _parse_func_arg_names(func_args)
    if not arg_names:
        return "This function has no arguments besides env."

    rule_map = build_cfg_rule_map(cfg_text or "")
    lines = [
        "Required arg_values keys — use EXACTLY these names (from the CFG), no synonyms:",
    ]
    for name in arg_names:
        symbol = name.upper()
        if _symbol_has_cfg_enumeration(symbol, rule_map):
            allowed = sorted(expand_symbol_to_terminals(symbol, rule_map))
            lines.append(f'  - "{name}": one of {", ".join(allowed)}')
        else:
            lines.append(
                f'  - "{name}": no terminal enumeration in CFG for {symbol}; '
                "use a plain integer when the argument is numeric, otherwise a string token"
            )
    return "\n".join(lines)


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

    arg_names = _parse_func_arg_names(func_args)
    if not arg_names:
        return {}

    rule_map = build_cfg_rule_map(cfg_text)

    allowed: dict[str, set[str]] = {}
    for arg in arg_names:
        symbol = arg.upper()
        if _symbol_has_cfg_enumeration(symbol, rule_map):
            allowed[arg] = expand_symbol_to_terminals(symbol, rule_map)
    return allowed


def _validate_arg_values_keys(arg_values: dict, expected_names: list[str]) -> tuple[bool, str]:
    """Require arg_values keys to match CFG-derived parameter names exactly."""
    if not expected_names:
        return True, ""
    if not isinstance(arg_values, dict):
        return False, "arg_values must be a JSON object"
    expected = set(expected_names)
    actual = set(arg_values.keys())
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        return False, f"arg_values missing required keys {missing} (expected {sorted(expected)})"
    if extra:
        return False, f"arg_values has unexpected keys {extra} (expected only {sorted(expected)})"
    return True, ""


def _validate_unenumerated_arg_value(raw) -> tuple[bool, str]:
    """Accept plain integers for CFG symbols without a terminal enumeration."""
    if isinstance(raw, bool):
        return False, "boolean is not a valid argument value"
    if isinstance(raw, int) and not isinstance(raw, bool):
        return True, ""
    if isinstance(raw, float) and raw.is_integer():
        return True, ""
    if isinstance(raw, str) and raw.strip().lstrip("-").isdigit():
        return True, ""
    return False, f"expected integer value, got {raw!r}"


def _validate_arg_values_against_cfg(
    arg_values: dict,
    allowed_map: dict[str, set[str]],
    expected_names: list[str],
) -> tuple[bool, str]:
    """Validate arg_values keys and values against CFG-derived constraints."""
    keys_ok, keys_error = _validate_arg_values_keys(arg_values, expected_names)
    if not keys_ok:
        return False, keys_error
    if not expected_names:
        return True, ""
    if not isinstance(arg_values, dict):
        return False, "arg_values must be a JSON object"

    for arg_name in expected_names:
        raw = arg_values[arg_name]
        allowed = allowed_map.get(arg_name)
        if allowed:
            candidate = str(raw).strip().upper()
            if candidate not in allowed:
                allowed_sorted = ", ".join(sorted(allowed))
                return (
                    False,
                    f"arg_values.{arg_name}='{raw}' is invalid; allowed values from CFG are: {allowed_sorted}",
                )
            continue
        ok, err = _validate_unenumerated_arg_value(raw)
        if not ok:
            return False, f"arg_values.{arg_name}: {err}"
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
    codebase_text: str = "",
    positive_grids: int = 10,
    negative_grids: int = 4,
    edge_grids: int = 1,
    init_check_failure: str = "",
    cfg_text: str = "",
) -> str:
    if existing_cases:
        summaries = []
        for i, case in enumerate(existing_cases):
            parts = [f"Case {i}: task_name={case.get('task_name', '?')}"]
            if case.get("test_type"):
                parts.append(f"test_type={case['test_type']}")
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
        "<<ARG_SCHEMA>>": _build_arg_schema(func_args, cfg_text),
        "<<ENV_DESCRIPTION>>": env_description,
        "<<CODEBASE>>": codebase_text,
        "<<RECIPES>>": recipes_text,
        "<<WIDTH>>": str(width),
        "<<HEIGHT>>": str(height),
        "<<VALID_ITEMS>>": ", ".join(sorted(valid_items)),
        "<<VALID_TASKS>>": ", ".join(sorted(valid_tasks)),
        "<<EXISTING_CASES>>": existing_text,
        "<<POSITIVE_GRIDS>>": str(positive_grids),
        "<<NEGATIVE_GRIDS>>": str(negative_grids),
        "<<EDGE_GRIDS>>": str(edge_grids),
        "<<TOTAL_GRIDS>>": str(positive_grids + negative_grids + edge_grids),
        "<<INIT_CHECK_FAILURE>>": init_check_failure if init_check_failure.strip() else "",
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
    require_test_type: bool = True,
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
    
    # Add test_type field
    test_type = spec.get("test_type")
    if test_type is None:
        if require_test_type:
            raise ValueError("test_type is required and must be one of: negative, positive, edge")
        else:
            test_type = "positive"  # default for old prompts that don't include test_type
    elif test_type not in ["negative", "positive", "edge"]:
        if require_test_type:
            raise ValueError(f"test_type must be one of 'negative', 'positive', 'edge', got: {test_type}")
        else:
            test_type = "positive"
    normalized["test_type"] = test_type
    
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
    codebase_text: str = "",
    require_test_type: bool = True,
    skip_positive_grids: bool = False,
    positive_grids: int = 10,
    negative_grids: int = 4,
    edge_grids: int = 1,
    init_check_failure: str = "",
) -> Optional[Dict]:
    cookbook = _get_cookbook(recipes_path)
    if cookbook is None:
        return None

    valid_items = _get_item_names(cookbook)
    valid_tasks = _get_task_names(cookbook)
    width, height = _get_grid_size()

    allowed_arg_values = _extract_allowed_arg_values_from_cfg(func_args or "", cfg_text or "")
    expected_arg_names = _parse_func_arg_names(func_args or "")

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
            codebase_text=codebase_text or "",
            positive_grids=positive_grids,
            negative_grids=negative_grids,
            edge_grids=edge_grids,
            init_check_failure=init_check_failure or "",
            cfg_text=cfg_text or "",
        )
        # print(f"[grid_generation] Prompt for {func_name}:\n{base_prompt}\n")
        # print(f"[grid_generation] Args for {func_name}: {func_args or 'None'}")
        params = SamplingParams(temperature=0.2, max_tokens=5000)
        attempts = max(1, int(attempts))
        last_error = ""
        failed_attempts: list[dict] = []
        
        def _record_failed_attempt(reason: str, candidate: str | dict) -> None:
            failed_attempts.append({
                "reason": reason,
                "candidate": _json_for_log(candidate),
            })

        def _format_failed_attempts() -> str:
            if not failed_attempts:
                return ""
            recent = failed_attempts[-5:]
            older = failed_attempts[:-5]
            lines = ["\n\nRecent failed attempts (last 5). fix these issues in the next attempt:"]
            for idx, item in enumerate(recent, start=1):
                candidate = item.get("candidate", "")
                if len(candidate) > 1200:
                    candidate = candidate[:1200] + "... [truncated]"
                lines.append(
                    f"\nAttempt context {idx} reason: {item.get('reason', 'unknown')}\n"
                    f"Attempt context {idx} candidate:\n{candidate}"
                )
            if older:
                lines.append("\nOther failure reasons:")
                for idx, item in enumerate(older, start=1):
                    lines.append(f"- Earlier failure {idx}: {item.get('reason', 'unknown')}")
            lines.append("\nUse these failures to avoid repeating mistakes.")
            return "\n".join(lines)

        for attempt in range(attempts):
            prompt = base_prompt
            if last_error:
                prompt = (
                    base_prompt
                    + _format_failed_attempts()
                    + "\nRegenerate a valid JSON spec."
                )
            print("prompt", prompt)
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
                    last_error = f"grid JSON parse failed: {e}"
                    _record_failed_attempt(last_error, raw)
                    print(f"[grid_generation] {func_name} attempt {attempt + 1}/{attempts} failed: {last_error}")
                    continue
            if not isinstance(spec, dict):
                last_error = "response is not a JSON object"
                _record_failed_attempt(last_error, json_text)
                print(f"[grid_generation] {func_name} attempt {attempt + 1}/{attempts} failed: {last_error}")
                continue
            required_keys = {"task_name", "width", "height", "grid"}
            if require_test_type:
                required_keys.add("test_type")
            if not required_keys.issubset(spec.keys()):
                last_error = f"missing required keys: {sorted(required_keys - set(spec.keys()))}"
                _record_failed_attempt(last_error, spec)
                print(f"[grid_generation] {func_name} attempt {attempt + 1}/{attempts} failed: {last_error}")
                continue
            try:
                normalized = _normalize_grid_spec(
                    spec=spec if isinstance(spec, dict) else {},
                    width=width,
                    height=height,
                    valid_items=valid_items,
                    cookbook=cookbook,
                    default_task_name=default_task_name,
                    require_test_type=require_test_type,
                )
            except ValueError as e:
                last_error = f"invalid test case: {e}"
                _record_failed_attempt(last_error, spec)
                print(f"[grid_generation] {func_name} attempt {attempt + 1}/{attempts} failed: {last_error}")
                continue
            args_ok, args_error = _validate_arg_values_against_cfg(
                normalized.get("arg_values", {}),
                allowed_arg_values,
                expected_arg_names,
            )
            if not args_ok:
                last_error = args_error
                _record_failed_attempt(last_error, normalized)
                print(f"[grid_generation] {func_name} attempt {attempt + 1}/{attempts} failed: {last_error}")
                continue
            print(normalized)
            is_valid = True
            if 'validate_grid_spec' in globals():
                is_valid, last_error = validate_grid_spec(normalized, cookbook)
            if not is_valid:
                _record_failed_attempt(last_error, normalized)
                print(f"[grid_generation] {func_name} attempt {attempt + 1}/{attempts} failed (validate_grid_spec): {last_error}")
            if is_valid:
                if skip_positive_grids and normalized.get('test_type') == 'positive':
                    # Don't write to disk; return spec so caller can use it as LLM context
                    return normalized
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
        f"[grid_generation] FAILED to generate valid grid for {func_name} after {attempts} attempts. Last error: {reason}"
    )
    return None


