"""Domain-agnostic version of ``src/pipeline/cfg_to_funsearch_pipeline.py``.

Most of the module is imported verbatim from :mod:`src.pipeline.cfg_to_funsearch_pipeline`;
only the functions that mention craft-specific files or templates are
overridden here to route through a :class:`~domains.base.DomainAdapter`.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Tuple

from domains.base import DomainAdapter

# Re-export helpers unchanged from the original module.
from src.pipeline.cfg_to_funsearch_pipeline import (  # noqa: F401
    apply_specification_template_placeholders,
    convert_tokenized_to_program_format,
    create_evaluation_file,
    create_experiment_directory,
    determine_inputs,
    extract_function_args,
    find_funsearch_log_file,
    generate_func_init,
    infer_argument_type,
    infer_return_type,
    parse_function_name_and_args,
    replace_codebase_placeholder_in_specification,
    replace_dsl_section_in_specification,
    replace_nld_placeholder_in_specification,
    run_explicit_feedback_generation,
    sanitize_function_name,
    validate_cfg,
    validate_terminal_descriptions,
    _load_seed_body,
    _versioned_name,
)
from src.pipeline.cfg_parser import CFGParser
from src.utils.config_loader import (
    funsearch_grid_regen_kwargs_from_config,
    load_config,
)
from src.utils.file_utils import version_file
from src.utils.results_tracker import plot_funsearch_reward_vs_interactions
from funsearch.implementation import config as config_lib
from funsearch.implementation.funsearch import FunSearch

try:
    from vllm import LLM as vLLM
except ImportError:  # pragma: no cover - vLLM optional at import time
    vLLM = None


def _resolve_codebase_text(codebase_path: Optional[str]) -> str:
    if not codebase_path:
        return ""
    if os.path.isfile(codebase_path):
        with open(codebase_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def generate_function_prompt(
    func_name: str,
    description: str,
    cfg: str,
    *,
    adapter: DomainAdapter,
    specification: str = "",
    experiment_dir: Optional[str] = None,
    dsl_round: Optional[int] = None,
    func_evolution_round: Optional[int] = None,
    shared_vllm=None,
    forced_task_name: Optional[str] = None,
    use_task_env: bool = False,
    grid_prompt_path: str = "prompt_specifications/grid_prompt.txt",
    require_test_type: bool = True,
    skip_positive_grids: bool = False,
    positive_grids: int = 10,
    negative_grids: int = 4,
    edge_grids: int = 1,
    codebase_path: Optional[str] = "prompt_specifications/codebase.txt",
) -> tuple[str, str]:
    """Domain-aware counterpart of the craft-only ``generate_function_prompt``.

    Every craft-specific file lookup and template call is replaced with a
    call into ``adapter``. Behavior for the Craft domain is identical.
    """
    func_dir = os.path.join(experiment_dir, "function_specific_prompts") if experiment_dir else "function_specific_prompts"
    os.makedirs(func_dir, exist_ok=True)

    base_name, args_list = parse_function_name_and_args(func_name)
    safe_name = sanitize_function_name(func_name)

    if args_list:
        args = ", ".join([a.strip().lower() for a in args_list if a.strip()])
    elif cfg and str(cfg).strip():
        args = extract_function_args(func_name, cfg)
    else:
        args = ""

    if (args == "arg" or not args) and cfg and str(cfg).strip():
        parser = CFGParser(cfg)
        for fname, fargs in parser.get_terminal_functions():
            if fname.strip().lower() == base_name.strip().lower():
                if fargs:
                    args = ", ".join([a.strip().lower() for a in fargs if a.strip()])
                break

    has_args = bool(args and args != "arg" and args.strip())
    display_name = base_name
    func_file = os.path.join(
        func_dir,
        f"{_versioned_name(safe_name, dsl_round, func_evolution_round)}.txt",
    )

    _, default_return = infer_return_type(description)

    arg_type = "str"
    typed_args = args if args else ""
    arg_list: list[str] = []
    if has_args:
        arg_list = [a.strip() for a in args.split(",") if a.strip()]
        inferred_types = [infer_argument_type(a, cfg, description) for a in arg_list]
        if inferred_types:
            typed_args = ", ".join([f"{n}:{t}" for n, t in zip(arg_list, inferred_types)])
            arg_type = inferred_types[0]

    if has_args:
        func_params = f"env, {args}"
        func_call_args = f"env, {args}"
        args_docstring = f"      {args} ({arg_type}): Function-specific argument(s).\n  "
    else:
        func_params = "env"
        func_call_args = "env"
        args_docstring = ""

    default_task_name = forced_task_name
    env_description = adapter.spec.nld
    recipes_text = adapter.domain_text_for_prompt()
    codebase_text = _resolve_codebase_text(codebase_path)

    grid_spec_paths: list[str] = []
    grid_spec: Optional[dict] = None

    if use_task_env:
        if not default_task_name:
            raise ValueError(f"Missing forced_task_name for task-env mode: {func_name}")
        task_name_for_env = default_task_name
        grid_spec_path = None
    else:
        grid_dir_override = os.environ.get("GRID_SPEC_DIR")
        grid_dir = grid_dir_override if grid_dir_override else (
            os.path.join(experiment_dir, "grids") if experiment_dir else "grids"
        )
        os.makedirs(grid_dir, exist_ok=True)
        num_grid_tests = (
            negative_grids + edge_grids
            if skip_positive_grids
            else (positive_grids + negative_grids + edge_grids)
        )
        total_grid_generation_attempts = 200

        use_existing_grids = str(os.environ.get("USE_EXISTING_GRID_SPECS", "")).lower() in {"1", "true", "yes"}
        if use_existing_grids and os.path.isdir(grid_dir):
            def _match(fname: str) -> bool:
                if not fname.lower().endswith(".json"):
                    return False
                if dsl_round is not None:
                    return f"{safe_name}_dsl{dsl_round}_" in fname
                return f"{safe_name}_" in fname

            for fname in sorted(os.listdir(grid_dir)):
                if _match(fname):
                    grid_spec_paths.append(os.path.join(grid_dir, fname))
            if not grid_spec_paths:
                for fname in sorted(os.listdir(grid_dir)):
                    if fname.lower().endswith(".json") and f"{safe_name}_" in fname:
                        grid_spec_paths.append(os.path.join(grid_dir, fname))
            if grid_spec_paths:
                with open(grid_spec_paths[0], "r", encoding="utf-8") as f:
                    grid_spec = json.load(f)

        if not grid_spec_paths and shared_vllm is not None:
            generated_cases: list[dict] = []
            saved_count = 0
            attempts_for_case = max(1, total_grid_generation_attempts // max(1, num_grid_tests))
            max_total_iters = num_grid_tests * 8 if skip_positive_grids else num_grid_tests * 2
            total_iters = 0
            if dsl_round is not None:
                _prefix = f"{safe_name}_dsl{dsl_round}_case"
            else:
                _prefix = f"{safe_name}_case"
            _existing_count = len([
                f for f in os.listdir(grid_dir)
                if f.startswith(_prefix) and f.endswith(".json")
            ]) if os.path.isdir(grid_dir) else 0

            def _case_num(fname: str) -> int:
                m = re.search(r"_case(\d+)\.json$", fname)
                return int(m.group(1)) if m else -1

            if _existing_count > 0:
                for _ef in sorted(os.listdir(grid_dir), key=_case_num):
                    if _ef.startswith(_prefix) and _ef.endswith(".json"):
                        with open(os.path.join(grid_dir, _ef), "r", encoding="utf-8") as _fh:
                            _loaded = json.load(_fh)
                        generated_cases.append(_loaded)
                        grid_spec_paths.append(os.path.join(grid_dir, _ef))
                        if skip_positive_grids and _loaded.get("test_type") != "positive":
                            saved_count += 1
                        elif not skip_positive_grids:
                            saved_count += 1
                if grid_spec_paths and grid_spec is None:
                    with open(grid_spec_paths[0], "r", encoding="utf-8") as _fh:
                        grid_spec = json.load(_fh)

            _new_saved_count = 0
            while saved_count < num_grid_tests and total_iters < max_total_iters:
                total_iters += 1
                generated_count = _existing_count + _new_saved_count
                if dsl_round is not None:
                    grid_filename = f"{safe_name}_dsl{dsl_round}_case{generated_count}.json"
                else:
                    grid_filename = f"{safe_name}_case{generated_count}.json"
                grid_spec_path = os.path.join(grid_dir, grid_filename)

                grid_spec = adapter.generate_test_case(
                    func_name=func_name,
                    description=description,
                    func_args=typed_args if has_args else "None",
                    output_path=grid_spec_path,
                    shared_vllm=shared_vllm,
                    prompt_path=grid_prompt_path,
                    existing_cases=generated_cases or None,
                    attempts=attempts_for_case,
                    require_test_type=require_test_type,
                    skip_positive_grids=skip_positive_grids,
                    positive_grids=positive_grids,
                    negative_grids=negative_grids,
                    edge_grids=edge_grids,
                    cfg_text=cfg,
                    codebase_text=codebase_text,
                    default_task_name=default_task_name,
                )
                if isinstance(grid_spec, dict):
                    generated_cases.append(grid_spec)
                    if skip_positive_grids and grid_spec.get("test_type") == "positive":
                        print(
                            f"[grid_generation] Skipping positive case for {func_name} "
                            f"(skip_positive_grids=True); using as LLM context only."
                        )
                    else:
                        grid_spec_paths.append(grid_spec_path)
                        saved_count += 1
                        _new_saved_count += 1
                else:
                    print(
                        f"[grid_generation] No valid grid for {func_name} case "
                        f"{generated_count}; will retry."
                    )

            if total_iters >= max_total_iters and saved_count < num_grid_tests:
                print(
                    f"[grid_generation] Warning: hit max iterations ({max_total_iters}) "
                    f"for {func_name}; only {saved_count}/{num_grid_tests} cases saved."
                )

        if not grid_spec_paths:
            raise ValueError(
                f"No grid specs available for {func_name}; shared_vllm unavailable "
                "and no reusable specs found."
            )

        grid_spec_path = grid_spec_paths[0]
        task_name_for_env = None
        if isinstance(grid_spec, dict):
            task_name_for_env = grid_spec.get("task_name") or None
        if not task_name_for_env:
            raise ValueError(
                f"Missing task_name in grid spec for {func_name}; adapter must supply a valid task."
            )

    if use_task_env:
        solve_func = adapter.solve_template_task_env(
            func_name=func_name,
            func_params=func_params,
            func_call_args=func_call_args,
        ) if hasattr(adapter, "solve_template_task_env") else adapter.solve_template(
            func_name=func_name,
            func_params=func_params,
            func_call_args=func_call_args,
        )
    else:
        solve_func = adapter.solve_template(
            func_name=func_name,
            func_params=func_params,
            func_call_args=func_call_args,
        )

    seed_body = _load_seed_body(experiment_dir, safe_name, dsl_round, func_evolution_round)
    if seed_body:
        seed_body = "\n".join([f"  {line}" if line.strip() else "" for line in seed_body.splitlines()])

    evolve_func = f'''@funsearch.evolve
def {safe_name}({func_params}):
  """
  {description}
  
  Args:
      env: The current environment instance.
  {args_docstring}  
      Returns: List[int]: A sequence of raw integer action codes accepted by env.step().

  """
'''

    if use_task_env:
        env_setup = ""
    else:
        env_setup = adapter.env_setup_code(
            task_name=task_name_for_env,
            test_case_path=grid_spec_path,
        )

    args_definitions = ""
    if has_args:
        arg_list = [a.strip() for a in args.split(",")] if "," in args else [args.strip()]
        args_def_lines = [
            '  arg_values = grid_spec["arg_values"] if isinstance(grid_spec, dict) else {}'
        ]
        for arg_name in arg_list:
            if not arg_name:
                continue
            args_def_lines.append(f'  {arg_name} = arg_values["{arg_name}"]')
            args_def_lines.append(f'  if isinstance({arg_name}, str):')
            args_def_lines.append(f'    {arg_name} = {arg_name}.lower()')
        args_definitions = "\n".join(args_def_lines) + "\n" if args_def_lines else ""

    eval_func = adapter.evaluate_template(
        display_name=display_name,
        env_setup=env_setup,
        args_definitions=args_definitions,
        func_call_args=func_call_args,
        test_case_paths_var=repr(grid_spec_paths) if grid_spec_paths else None,
    )

    if solve_func is None:
        raise ValueError(
            f"Adapter returned None for solve template of {func_name}."
        )

    prompt_content = solve_func + "\n" + eval_func + "\n" + evolve_func
    with open(func_file, "w", encoding="utf-8") as f:
        f.write(prompt_content)

    func_signature = f"def {safe_name}({func_params})"
    print(f"Generated function prompt: {func_file}")
    return func_file, func_signature


# ---------------------------------------------------------------------------
# CFG generation + implement_cfg delegation
# ---------------------------------------------------------------------------

def get_cfg(
    *,
    adapter: DomainAdapter,
    experiment_dir: str,
    skip_cfg_generation: bool = False,
    cfg_output_file: Optional[str] = None,
    max_cfg_retries: int = 10,
    cfg_generator_prompt_path: str = "prompt_specifications/cfg_generator.txt",
    domain_context_template_path: Optional[str] = None,
    shared_vllm=None,
) -> Tuple[str, Dict[str, str], Optional[str], bool]:
    """Domain-aware CFG retrieval.

    ``adapter.spec.nld`` and :meth:`adapter.domain_text_for_prompt` are
    materialized to temporary files so the existing CFG generator (which
    still expects paths) can be reused without changes.
    """
    print(f"\n{'='*80}")
    print(f"Getting CFG for domain '{adapter.spec.name}'")
    print(f"{'='*80}")

    cfg_path = os.path.join(experiment_dir, "cfg", "cfg_output.json")
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)

    if skip_cfg_generation and cfg_output_file and os.path.exists(cfg_output_file):
        with open(cfg_output_file, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
        cfg = cfg_data.get("cfg", "")
        terminals = cfg_data.get("terminals", {})
        example = cfg_data.get("example")
        is_valid, msg = validate_cfg(cfg, example)
        if not is_valid:
            print(f"ERROR: Loaded CFG validation failed: {msg}", file=sys.stderr)
            return "", {}, None, False
        t_valid, t_msg = validate_terminal_descriptions(terminals)
        if not t_valid:
            print(f"ERROR: Loaded terminal descriptions failed validation: {t_msg}", file=sys.stderr)
            return "", {}, None, False
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_data, f, indent=2, ensure_ascii=False)
        versioned_path = os.path.join(os.path.dirname(cfg_path), "cfg_output_0.json")
        with open(versioned_path, "w", encoding="utf-8") as f:
            json.dump(cfg_data, f, indent=2, ensure_ascii=False)
        return cfg, terminals, example, True

    if skip_cfg_generation and os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
        cfg = cfg_data.get("cfg", "")
        terminals = cfg_data.get("terminals", {})
        example = cfg_data.get("example")
        is_valid, msg = validate_cfg(cfg, example)
        if not is_valid:
            print(f"ERROR: Loaded CFG validation failed: {msg}", file=sys.stderr)
            return "", {}, None, False
        t_valid, t_msg = validate_terminal_descriptions(terminals)
        if not t_valid:
            print(f"ERROR: Loaded terminal descriptions failed validation: {t_msg}", file=sys.stderr)
            return "", {}, None, False
        return cfg, terminals, example, True

    domain_assets_dir = os.path.join(experiment_dir, "domain_assets")
    os.makedirs(domain_assets_dir, exist_ok=True)
    nld_path = adapter.write_nld_file(os.path.join(domain_assets_dir, f"nld_{adapter.spec.name}.txt"))
    domain_context_path = adapter.write_domain_context_file(
        os.path.join(domain_assets_dir, f"domain_context_{adapter.spec.name}.txt")
    )

    for attempt in range(1, max_cfg_retries + 1):
        if attempt > 1:
            print(f"\n[Generating CFG] Retry attempt {attempt}/{max_cfg_retries}...")

        from src.pipeline.getting_cfg import generate_and_parse_cfg
        cfg, terminals, example = generate_and_parse_cfg(
            vllm_instance=shared_vllm,
            nld_path=nld_path,
            recipes_path=domain_context_path,
            prompt_template_path=cfg_generator_prompt_path,
            domain_context_template_path=domain_context_template_path,
        )

        is_valid, msg = validate_cfg(cfg, example)
        if not is_valid:
            print(f" CFG validation failed: {msg}")
            if attempt < max_cfg_retries:
                continue
            return "", {}, None, False

        from src.pipeline.integrated_pipeline import ensure_terminals_match_cfg
        terminals = ensure_terminals_match_cfg(cfg, terminals or {}, shared_vllm=shared_vllm)
        t_valid, t_msg = validate_terminal_descriptions(terminals)
        if not t_valid:
            print(f" Terminal description validation failed: {t_msg}")
            if attempt < max_cfg_retries:
                continue
            return "", {}, None, False

        cfg_data = {"cfg": cfg, "terminals": terminals, "example": example}
        if os.path.exists(cfg_path):
            version_file(cfg_path)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_data, f, indent=2, ensure_ascii=False)
        versioned_path_0 = os.path.join(experiment_dir, "cfg", "cfg_output_0.json")
        if not os.path.exists(versioned_path_0):
            import shutil

            shutil.copy2(cfg_path, versioned_path_0)
        return cfg, terminals, example, True

    print(f"\nERROR: Failed to generate valid CFG after {max_cfg_retries} attempts", file=sys.stderr)
    return "", {}, None, False


class _AdapterContext:
    """Module-level context used to thread the active adapter into helpers."""

    current: Optional[DomainAdapter] = None
    shared_vllm = None


def _adapter_aware_generate_function_prompt(
    func_name,
    description,
    cfg,
    specification="",
    experiment_dir=None,
    dsl_round=None,
    func_evolution_round=None,
    shared_vllm=None,
    forced_task_name=None,
    use_task_env=False,
    grid_prompt_path="prompt_specifications/grid_prompt.txt",
    require_test_type=True,
    skip_positive_grids=False,
    positive_grids=10,
    negative_grids=4,
    edge_grids=1,
    codebase_path="prompt_specifications/codebase.txt",
):
    """Drop-in replacement for the craft-only ``generate_function_prompt``."""
    adapter = _AdapterContext.current
    if adapter is None:
        raise RuntimeError(
            "No DomainAdapter bound. Use `with_adapter(adapter):` around the call."
        )
    if shared_vllm is None:
        shared_vllm = _AdapterContext.shared_vllm
    adapter_prompt = getattr(adapter, "test_case_prompt_path", None)
    if adapter_prompt:
        grid_prompt_path = adapter_prompt
    return generate_function_prompt(
        func_name=func_name,
        description=description,
        cfg=cfg,
        adapter=adapter,
        specification=specification,
        experiment_dir=experiment_dir,
        dsl_round=dsl_round,
        func_evolution_round=func_evolution_round,
        shared_vllm=shared_vllm,
        forced_task_name=forced_task_name,
        use_task_env=use_task_env,
        grid_prompt_path=grid_prompt_path,
        require_test_type=require_test_type,
        skip_positive_grids=skip_positive_grids,
        positive_grids=positive_grids,
        negative_grids=negative_grids,
        edge_grids=edge_grids,
        codebase_path=codebase_path,
    )


class with_adapter:
    """Context manager that rebinds craft-specific hooks to an adapter.

    Inside the ``with`` block, calls to
    :func:`src.pipeline.cfg_to_funsearch_pipeline.generate_function_prompt`
    (including those performed by ``implement_cfg`` and friends) are routed
    through ``adapter``.
    """

    def __init__(self, adapter: DomainAdapter, shared_vllm=None) -> None:
        self.adapter = adapter
        self.shared_vllm = shared_vllm
        self._prev_generate_function_prompt = None
        self._prev_adapter = None
        self._prev_shared_vllm = None

    def __enter__(self) -> DomainAdapter:
        import src.pipeline.cfg_to_funsearch_pipeline as upstream

        self._prev_adapter = _AdapterContext.current
        self._prev_shared_vllm = _AdapterContext.shared_vllm
        _AdapterContext.current = self.adapter
        _AdapterContext.shared_vllm = self.shared_vllm
        self._prev_generate_function_prompt = upstream.generate_function_prompt
        upstream.generate_function_prompt = _adapter_aware_generate_function_prompt
        return self.adapter

    def __exit__(self, exc_type, exc, tb) -> None:
        import src.pipeline.cfg_to_funsearch_pipeline as upstream

        upstream.generate_function_prompt = self._prev_generate_function_prompt
        _AdapterContext.current = self._prev_adapter
        _AdapterContext.shared_vllm = self._prev_shared_vllm


def implement_cfg(
    cfg: str,
    terminals: Dict[str, str],
    example: Optional[str],
    spec_file: str,
    experiment_dir: str,
    *,
    adapter: DomainAdapter,
    model_type: str = "huggingface",
    shared_vllm=None,
    results_tracker=None,
    dsl_round: Optional[int] = None,
    func_evolution_round: Optional[int] = None,
    nld_path: Optional[str] = None,
    codebase_path: Optional[str] = None,
) -> Tuple[bool, Dict[str, str]]:
    """Domain-aware ``implement_cfg``.

    Rebinds craft-specific templates to ``adapter`` and then delegates to the
    original :func:`src.pipeline.cfg_to_funsearch_pipeline.implement_cfg` so
    the parallel FunSearch/explicit-feedback loop is preserved verbatim.
    """
    if nld_path is None:
        domain_assets_dir = os.path.join(experiment_dir, "domain_assets")
        os.makedirs(domain_assets_dir, exist_ok=True)
        nld_path = adapter.write_nld_file(
            os.path.join(domain_assets_dir, f"nld_{adapter.spec.name}.txt")
        )

    from src.pipeline.cfg_to_funsearch_pipeline import implement_cfg as upstream_impl

    with with_adapter(adapter, shared_vllm=shared_vllm):
        return upstream_impl(
            cfg,
            terminals,
            example,
            spec_file,
            experiment_dir,
            model_type=model_type,
            shared_vllm=shared_vllm,
            results_tracker=results_tracker,
            dsl_round=dsl_round,
            func_evolution_round=func_evolution_round,
            nld_path=nld_path,
            codebase_path=codebase_path,
        )
