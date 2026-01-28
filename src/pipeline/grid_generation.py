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

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None

try:
    from craft.cookbook import Cookbook
    from craft.craft import WIDTH, HEIGHT
except Exception:
    Cookbook = None
    WIDTH, HEIGHT = 12, 12


def _get_grid_size() -> Tuple[int, int]:
    return int(WIDTH), int(HEIGHT)


def _extract_json_object(text: str) -> str:
    if not text:
        return ""
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


def _render_prompt(
    template: str,
    func_name: str,
    description: str,
    width: int,
    height: int,
    valid_items: list[str],
    valid_tasks: list[str],
) -> str:
    replacements = {
        "<<FUNCTION_NAME>>": func_name,
        "<<DESCRIPTION>>": description,
        "<<WIDTH>>": str(width),
        "<<HEIGHT>>": str(height),
        "<<VALID_ITEMS>>": ", ".join(sorted(valid_items)),
        "<<VALID_TASKS>>": ", ".join(sorted(valid_tasks)),
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


def _validate_task_name(task_name: str, cookbook: Cookbook, default_task: str) -> str:
    if not task_name or not isinstance(task_name, str):
        return default_task
    match = re.match(r"^(get|make)\[([^\]]+)\]$", task_name.strip())
    if not match:
        return default_task
    action, item = match.group(1), match.group(2).strip().lower()
    primitives = {cookbook.index.get(idx) for idx in cookbook.primitives}
    recipes = {cookbook.index.get(idx) for idx in cookbook.recipes.keys()}
    primitives = {p.lower() for p in primitives if p}
    recipes = {r.lower() for r in recipes if r}
    if action == "get":
        return f"get[{item}]" if item in primitives else default_task
    if action == "make":
        return f"make[{item}]" if item in recipes or item in primitives else default_task
    return default_task


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

    task_name = spec.get("task_name", default_task_name)
    normalized["task_name"] = _validate_task_name(task_name, cookbook, default_task_name)
    return normalized


def ensure_function_grid_spec(
    func_name: str,
    description: str,
    recipes_path: str,
    output_path: str,
    shared_vllm=None,
    default_task_name: str = "make[goldarrow]",
    prompt_path: str = "prompt_specifications/grid_prompt.txt",
) -> Optional[Dict]:
    cookbook = _get_cookbook(recipes_path)
    if cookbook is None:
        return None

    valid_items = _get_item_names(cookbook)
    valid_tasks = _get_task_names(cookbook)
    width, height = _get_grid_size()

    spec = {}

    if shared_vllm is not None and SamplingParams is not None:
        template = _load_prompt_template(prompt_path)
        prompt = _render_prompt(
            template=template,
            func_name=func_name,
            description=description,
            width=width,
            height=height,
            valid_items=list(valid_items),
            valid_tasks=valid_tasks,
        )
        params = SamplingParams(temperature=0.2, max_tokens=2000)
        output = shared_vllm.generate([prompt], sampling_params=params)
        raw = output[0].outputs[0].text.strip()
        print(f"[grid_generation] Raw LLM output for {func_name}:\n{raw}\n")
        json_text = _extract_json_object(raw)
        try:
            spec = json.loads(json_text)
        except Exception as e:
            try:
                spec = ast.literal_eval(json_text)
            except Exception:
                print(f"  Warning: grid JSON parse failed for {func_name}: {e}")
                spec = {}
    else:
        print("  Warning: shared_vllm/SamplingParams unavailable, using empty grid spec")

    normalized = _normalize_grid_spec(
        spec=spec if isinstance(spec, dict) else {},
        width=width,
        height=height,
        valid_items=valid_items,
        cookbook=cookbook,
        default_task_name=default_task_name,
    )
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2)
    return normalized


def build_env_setup(
    recipes_path: str,
    hints_path: str,
    task_name: str,
    grid_spec_path: str,
) -> str:
    return f"""
  import os
  import json
  recipes_path = "{recipes_path}"
  hints_path = "{hints_path}"
  grid_spec_path = r"{grid_spec_path}"
  task_name = "{task_name}"
  if os.path.exists(grid_spec_path):
    try:
      with open(grid_spec_path, "r", encoding="utf-8") as f:
        grid_spec = json.load(f)
      task_name = grid_spec.get("task_name", task_name) or task_name
    except Exception:
      pass
  custom_grid_path = grid_spec_path if os.path.exists(grid_spec_path) else None
  env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 7, max_steps=300, reuse_environments=False,
            visualise=visualise, custom_grid_path=custom_grid_path)
  env = env_sampler.sample_environment(task_name=task_name)
  """
