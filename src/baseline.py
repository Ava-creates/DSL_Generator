#!/usr/bin/env python3
"""
Baseline runner: build prompts for env_task_list tasks, run FunSearch,
and generate explicit feedback outputs without CFG generation.
"""

import argparse
import importlib.util
import json
import os
import re
import shlex
import sys
import subprocess
from numbers import Real

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from src.pipeline.cfg_to_funsearch_pipeline import (
    generate_baseline_function_prompt,
    generate_func_init,
    sanitize_function_name,
    apply_specification_template_placeholders,
)
from src.utils.pipeline_state import update_state
from src.utils.status_manager import write_status
from src.utils.config_loader import load_config
from craft import env_factory

try:
    from vllm import LLM as vLLM
except ImportError:
    vLLM = None


TASKS = [
    {"title": "Get wood", "description": "Get wood."},
    {"title": "Get grass", "description": "Get grass."},
    {"title": "Get iron", "description": "Get iron."},
    {"title": "Make plank", "description": "Get primitives required as mentioned in the recipe and then make a plank at workshop."},
    {"title": "Make stick", "description": "Get primitives required as mentioned in the recipe and then make a stick at workshop."},
    {"title": "Make cloth", "description": "Get primitives required as mentioned in the recipe and then make cloth at workshop."},
    {"title": "Make rope", "description": "Get primitives required as mentioned in the recipe and then make rope at workshop."},
    {"title": "Make bridge", "description": "Get primitives required as mentioned in the recipe and then make a bridge at workshop."},
    {"title": "Make bundle", "description": "Get primitives required as mentioned in the recipe and then make a bundle at workshop."},
    {"title": "Get gold", "description": "Get primitives required as mentioned in the recipe, make item to cross obstacle around gold, and then get gold."},
    {"title": "Make flag", "description": "Get primitives required as mentioned in the recipe and then make a flag at workshop."},
    {"title": "Make bed", "description": "Get primitives required as mentioned in the recipe and then make a bed at workshop."},
    {"title": "Make axe", "description": "Get primitives required as mentioned in the recipe and then make an axe at workshop."},
    {"title": "Make shears", "description": "Get primitives required as mentioned in the recipe and then make shears at workshop."},
    {"title": "Make ladder", "description": "Get primitives required as mentioned in the recipe and then make a ladder at workshop."},
    {"title": "Get gem", "description": "Figure out what item is needed to cross obstacle to get gem and then make that item and cross obstacle and get gem."},
    {"title": "Make golden arrow", "description": "Figure out what item is needed to cross obstacle to get golden arrow."},
]


ITEM_OVERRIDES = {
    "golden arrow": "goldarrow",
}

BASELINE_VARIANT_TASK_ENV = "task_env"
BASELINE_VARIANT_TESTCASE = "testcase"
BASELINE_VARIANT_TWO_PHASE = "two_phase_seeded_random"
DEFAULT_TEST_SEEDS = list(range(0, 50, 5))


def _latest_func_round(experiment_dir: str) -> int | None:
    """Return latest func round present in final_functions (baseline semantics)."""
    final_dir = os.path.join(experiment_dir, "final_functions")
    if not os.path.isdir(final_dir):
        return None

    # Support both legacy names (..._dsl0_funcN.py) and baseline names (..._funcN.py).
    pat = re.compile(r"(?:_dsl\d+)?_func(\d+)\.py$")
    found = []
    for name in os.listdir(final_dir):
        m = pat.search(name)
        if m:
            found.append(int(m.group(1)))
    return max(found) if found else None


def task_name_from_title(title: str) -> str:
    title = title.strip()
    lower = title.lower()
    if lower.startswith("get "):
        action = "get"
        item = lower[len("get "):].strip()
    elif lower.startswith("make "):
        action = "make"
        item = lower[len("make "):].strip()
    else:
        raise ValueError(f"Unsupported task title format: {title}")
    item_token = ITEM_OVERRIDES.get(item, item.replace(" ", ""))
    return f"{action}[{item_token}]"


def _bash_wrap(command: str) -> str:
    """Run wrapped SLURM command with bash so `source` is available."""
    return f"bash -lc {shlex.quote(command)}"


def _load_function_from_file(file_path: str, function_name: str):
    """Load a Python function from a file path."""
    module_name = f"baseline_eval_{function_name}_{abs(hash(file_path))}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, function_name):
        raise AttributeError(f"Function {function_name} not found in {file_path}")
    return getattr(module, function_name)


def _is_bool_like(value) -> bool:
    """Return True for Python and numpy boolean scalars."""
    if isinstance(value, bool):
        return True
    return type(value).__name__ in {"bool_", "bool8"}


def run_baseline_final_evaluation(
    experiment_dir: str,
    dsl_round: int,
    func_evolution_round: int,
    recipes_path: str = "craft/resources/recipes.yaml",
    hints_path: str = "craft/resources/hints.yaml",
    max_steps: int = 400,
    test_seeds=None,
) -> int:
    """Evaluate final baseline functions directly on their matching tasks.

    Returns 0 when all tasks are solved, else 1.
    """
    final_functions_dir = os.path.join(experiment_dir, "final_functions")
    if not os.path.isdir(final_functions_dir):
        print(f" Final functions directory not found: {final_functions_dir}", file=sys.stderr)
        return 1

    selected_seeds = [int(s) for s in (test_seeds if test_seeds else DEFAULT_TEST_SEEDS)]
    print(f"[Baseline Final Eval] Using test seeds: {selected_seeds}")

    results = []
    solved_count = 0

    for task in TASKS:
        title = task["title"]
        safe_name = sanitize_function_name(title)
        task_name = task_name_from_title(title)

        candidate_paths = [
            # Baseline-preferred naming (dsl-free)
            os.path.join(final_functions_dir, f"{safe_name}_func{func_evolution_round}.py"),
            os.path.join(final_functions_dir, f"{safe_name}_func0.py"),
            # Legacy naming retained for compatibility
            os.path.join(final_functions_dir, f"{safe_name}_dsl0_func{func_evolution_round}.py"),
            os.path.join(final_functions_dir, f"{safe_name}_dsl0_func0.py"),
            os.path.join(final_functions_dir, f"{safe_name}.py"),
        ]
        func_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if func_path is None:
            results.append({
                "task": task_name,
                "function": safe_name,
                "solved": False,
                "reason": "missing_final_function",
                "seed_results": {},
            })
            continue

        try:
            func = _load_function_from_file(func_path, safe_name)
        except Exception as e:
            results.append({
                "task": task_name,
                "function": safe_name,
                "solved": False,
                "reason": f"load_error: {e}",
                "seed_results": {},
            })
            continue

        solved = False
        solved_seed = None
        seeds_tested = 0
        seed_results = {}
        best_reward = float("-inf")
        best_actions = 0
        last_runtime_error = None

        for seed in selected_seeds:
            seed_key = str(int(seed))
            seeds_tested += 1
            try:
                sampler = env_factory.EnvironmentFactory(
                    recipes_path,
                    hints_path,
                    7,
                    max_steps=max_steps,
                    seed=int(seed),
                    reuse_environments=False,
                    visualise=False,
                )

                env = sampler.sample_environment(task_name=task_name)
                env.reset()

                actions = func(env)
                if actions is None:
                    actions = []
                if not isinstance(actions, list):
                    raise TypeError(f"Expected list actions, got {type(actions).__name__}")

                total_reward = 0.0
                done = False
                executed = 0
                for a in actions:
                    step_out = env.step(int(a))
                    executed += 1
                    if isinstance(step_out, tuple) and len(step_out) >= 3:
                        # Craft env returns (reward, done, observations) where reward
                        # may be a numpy scalar (e.g., np.float32). Detect tuple layout
                        # via boolean slot positions instead of numeric type checks.
                        if _is_bool_like(step_out[1]):
                            reward_raw = step_out[0]
                            done = bool(step_out[1])
                        elif _is_bool_like(step_out[2]):
                            # Gym-style fallback: (obs, reward, done, ...)
                            reward_raw = step_out[1]
                            done = bool(step_out[2])
                        else:
                            reward_raw = step_out[0]
                            done = False

                        reward = float(reward_raw) if isinstance(reward_raw, Real) and not isinstance(reward_raw, bool) else 0.0
                        total_reward += reward
                    if done:
                        break

                if total_reward > best_reward:
                    best_reward = total_reward
                    best_actions = executed

                # In this env, `done` is not a reliable success signal for task completion.
                # Use reward threshold as the solve criterion.
                this_seed_solved = bool(total_reward > 10)
                seed_results[seed_key] = "success" if this_seed_solved else "failure"
                if this_seed_solved and not solved:
                    solved = True
                    solved_seed = int(seed)
            except Exception as e:
                last_runtime_error = str(e)
                seed_results[seed_key] = f"error: {e}"
                continue

        if solved:
            solved_count += 1

        if best_reward == float("-inf"):
            reason = f"runtime_error: {last_runtime_error}" if last_runtime_error else "runtime_error"
            results.append({
                "task": task_name,
                "function": safe_name,
                "solved": False,
                "reason": reason,
                "function_path": func_path,
                "seeds_tested": seeds_tested,
                "seed_results": seed_results,
            })
        else:
            results.append(
                {
                    "task": task_name,
                    "function": safe_name,
                    "solved": solved,
                    "executed_actions": best_actions,
                    "total_reward": best_reward,
                    "function_path": func_path,
                    "seeds_tested": seeds_tested,
                    "solved_seed": solved_seed,
                    "seed_results": seed_results,
                }
            )

    all_solved = solved_count == len(TASKS)
    out_dir = os.path.join(experiment_dir, "results_tracking")
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "all_solved": all_solved,
        "solved_count": solved_count,
        "total_tasks": len(TASKS),
        "func_evolution_round": int(func_evolution_round),
        "test_seeds": selected_seeds,
        "results": results,
    }

    # Per-round artifact to avoid overwriting previous round evaluations.
    out_path = os.path.join(
        out_dir,
        f"baseline_final_eval_func{int(func_evolution_round)}.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Keep a "latest" compatibility file for existing tooling.
    latest_out = os.path.join(out_dir, "baseline_final_eval.json")
    with open(latest_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[Baseline Final Eval] solved {solved_count}/{len(TASKS)} tasks")
    print(f"[Baseline Final Eval] wrote: {out_path}")
    print(f"[Baseline Final Eval] latest: {latest_out}")
    for r in results:
        status = "PASS" if r.get("solved") else "FAIL"
        print(f"  {status} {r.get('task')} ({r.get('function')})")

    return 0 if all_solved else 1


def main():
    parser = argparse.ArgumentParser(description="Baseline FunSearch runner for env_task_list tasks")
    parser.add_argument("--spec_file", type=str, required=True, help="Path to specification file")
    parser.add_argument("--experiment_dir", type=str, default="experiments/baseline", help="Experiment directory")
    parser.add_argument("--model_type", type=str, default="huggingface", choices=["huggingface", "ollama", "gemini", "openai_compat"])
    parser.add_argument("--openai_compat_key_file", type=str, default=None, help="File with OpenAI-compatible API key (first non-empty line). Default: <repo>/key.txt if OPENAI_COMPAT_API_KEY unset.")
    parser.add_argument("--dsl_round", type=int, default=0, help="Deprecated in baseline; ignored for round semantics")
    parser.add_argument("--func_evolution_round", type=int, default=0, help="Function round index (for naming)")
    parser.add_argument("--total_samples", type=int, default=1000, help="Total FunSearch samples per task")
    parser.add_argument("--num_explicit_feedback_iterations", type=int, default=30,
                        help="Number of explicit feedback iterations per task")
    parser.add_argument("--job_prefix", type=str, default=None, help="Optional job name prefix for SLURM")
    parser.add_argument("--cfg_text", type=str, default="", help="CFG text to inject into spec templates that contain <<CFG>>")
    parser.add_argument("--nld_path", type=str, default=None, help="Optional NLD file path for <<NLD>> replacement")
    parser.add_argument("--codebase_path", type=str, default=None, help="Optional codebase file path for <<CODEBASE>> replacement")
    parser.add_argument("--grid_prompt", type=str, default="prompt_specifications/grid_prompt.txt", help="Path to grid generation prompt template")
    parser.add_argument("--require_test_type", type=lambda x: x.lower() != "false", default=True, help="Whether to require test_type in generated grid specs")
    parser.add_argument("--skip_positive_grids", type=lambda x: x.lower() == "true", default=False, help="When true, only save negative/edge generated grids")
    parser.add_argument(
        "--baseline_variant",
        type=str,
        default=BASELINE_VARIANT_TESTCASE,
        choices=[BASELINE_VARIANT_TASK_ENV, BASELINE_VARIANT_TESTCASE, BASELINE_VARIANT_TWO_PHASE],
        help="Baseline mode: task_env, testcase, or two_phase_seeded_random",
    )
    parser.add_argument("--final_eval_only", action="store_true", help="Run only final baseline evaluation on saved final functions")
    parser.add_argument("--recipes_path", type=str, default="craft/resources/recipes.yaml", help="Path to recipes YAML")
    parser.add_argument("--hints_path", type=str, default="craft/resources/hints.yaml", help="Path to hints YAML")
    parser.add_argument("--max_steps", type=int, default=400, help="Max env steps for final evaluation")
    parser.add_argument(
        "--test_seeds",
        type=int,
        nargs='+',
        default=DEFAULT_TEST_SEEDS,
        help="Seeds used for baseline final evaluation (default: 0 5 10 15 20 25 30 35 40 45)",
    )
    parser.add_argument("--phase2_only", action="store_true", help="Skip phase-1 and run seeded task-env phase directly")
    parser.add_argument("--phase2_seed_round", type=int, default=None, help="Seed from this completed func round; phase2 runs at round+1")
    parser.add_argument("--task_env_rounds", type=int, default=1, help="Number of chained task_env rounds to run starting from current func_evolution_round")
    args = parser.parse_args()

    if int(args.task_env_rounds) < 1:
        args.task_env_rounds = 1

    if args.final_eval_only:
        return run_baseline_final_evaluation(
            experiment_dir=args.experiment_dir,
            dsl_round=0,
            func_evolution_round=int(args.func_evolution_round),
            recipes_path=args.recipes_path,
            hints_path=args.hints_path,
            max_steps=int(args.max_steps),
            test_seeds=args.test_seeds,
        )

    if not os.path.exists(args.spec_file):
        print(f" Specification file not found: {args.spec_file}", file=sys.stderr)
        return 1

    if args.phase2_only:
        args.baseline_variant = BASELINE_VARIANT_TASK_ENV
        if args.phase2_seed_round is not None:
            args.func_evolution_round = int(args.phase2_seed_round) + 1
            print(f"[Phase2-Only] Using explicit seed round func{int(args.phase2_seed_round)} -> running func{int(args.func_evolution_round)}")
        elif int(args.func_evolution_round) <= 0:
            latest = _latest_func_round(args.experiment_dir)
            if latest is None:
                print(
                    " ERROR: phase2_only requested but no prior final functions found for this dsl round. "
                    "Provide --phase2_seed_round or set --func_evolution_round explicitly.",
                    file=sys.stderr,
                )
                return 1
            args.func_evolution_round = int(latest) + 1
            print(f"[Phase2-Only] Auto-detected latest func round func{latest}; running func{int(args.func_evolution_round)}")
        else:
            print(f"[Phase2-Only] Using provided --func_evolution_round={int(args.func_evolution_round)}")

    requires_testcase_generation = args.baseline_variant in {BASELINE_VARIANT_TESTCASE, BASELINE_VARIANT_TWO_PHASE}

    experiment_dir = args.experiment_dir
    os.makedirs(experiment_dir, exist_ok=True)

    baseline_log_dir = os.environ.get("BASELINE_LOG_DIR", "").strip()
    if not baseline_log_dir:
        baseline_log_dir = os.path.join("scripts", "log", experiment_dir)
    os.makedirs(baseline_log_dir, exist_ok=True)
    print(f"[Logs] Baseline logs will be written to: {baseline_log_dir}")

    os.makedirs(os.path.join(experiment_dir, "function_specific_prompts"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "functions_generated"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "results", "funsearch"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "explicit_feedback"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "final_functions"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "cfg"), exist_ok=True)

    # Save a minimal cfg_output.json for record-keeping
    terminals = {task["title"]: task["description"] for task in TASKS}
    cfg_text = args.cfg_text or ""
    cfg_path = os.path.join(experiment_dir, "cfg", "cfg_output.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({"cfg": cfg_text, "terminals": terminals, "example": None}, f, indent=2)

    # Load and template the specification once for prompt generation.
    with open(args.spec_file, "r", encoding="utf-8") as f:
        specification = f.read()
    specification = apply_specification_template_placeholders(
        specification,
        cfg=cfg_text,
        nld_path=args.nld_path,
        codebase_path=args.codebase_path,
    )

    # Shared vLLM is only required when testcase grids are needed.
    shared_vllm = None
    if requires_testcase_generation and args.model_type == "huggingface" and vLLM is not None:
        try:
            print("\n[Setup] Initializing shared vLLM for baseline test-case generation...")
            shared_vllm = vLLM(model="/scratch/avani/gpt", tensor_parallel_size=4)
            print(" Shared vLLM instance created")
        except Exception as e:
            print(f" ERROR: Could not create shared vLLM instance: {e}", file=sys.stderr)
            return 1

    if requires_testcase_generation and shared_vllm is None and os.environ.get("USE_EXISTING_GRID_SPECS", "").lower() not in {"1", "true", "yes"}:
        print(
            " ERROR: Baseline now generates test cases before FunSearch, but no vLLM is available "
            "and USE_EXISTING_GRID_SPECS is not enabled.",
            file=sys.stderr,
        )
        print(
            "  Use --model_type huggingface on a GPU node, or set USE_EXISTING_GRID_SPECS=1 with pre-generated grids.",
            file=sys.stderr,
        )
        return 1

    config = load_config()
    positive_grids = int(config.get("positive_girds", config.get("positive_grids", 10)))
    negative_grids = int(config.get("negative_grids", 4))
    edge_grids = int(config.get("edge_grids", 1))

    print(f"\n[Mode] baseline_variant={args.baseline_variant}")
    if args.baseline_variant == BASELINE_VARIANT_TASK_ENV:
        print("  Using direct task envs without testcase grid generation.")
        print(f"  Chained task_env rounds requested: {int(args.task_env_rounds)}")
    elif args.baseline_variant == BASELINE_VARIANT_TWO_PHASE:
        print("  Using testcase-guided phase-1 then seeded task-env phase-2.")
    else:
        print("  Using testcase-guided baseline (current behavior).")

    # Generate function-specific prompt and init files (with grid/test-case generation first)
    func_files = {}
    func_init_files = {}
    func_signatures = {}
    for task in TASKS:
        display_name = task["title"]
        description = task["description"]
        task_name = task_name_from_title(display_name)
        impl_name = sanitize_function_name(display_name)

        func_file, func_signature = generate_baseline_function_prompt(
            func_name=impl_name,
            description=description,
            cfg=cfg_text,
            specification=specification,
            experiment_dir=experiment_dir,
            dsl_round=None,
            func_evolution_round=args.func_evolution_round,
            task_name=task_name,
            variant=args.baseline_variant,
            shared_vllm=shared_vllm,
            grid_prompt_path=args.grid_prompt,
            require_test_type=args.require_test_type,
            skip_positive_grids=args.skip_positive_grids,
            positive_grids=positive_grids,
            negative_grids=negative_grids,
            edge_grids=edge_grids,
        )
        func_signatures[display_name] = func_signature

        func_files[display_name] = func_file

        func_init_file = generate_func_init(
            func_name=impl_name,
            description=description,
            cfg=cfg_text,
            experiment_dir=experiment_dir,
            dsl_round=None,
            func_evolution_round=args.func_evolution_round,
        )
        func_init_files[display_name] = func_init_file

    file_gen_status = {
        "stage": "file_generation",
        "status": "completed",
        "func_evolution_round": args.func_evolution_round,
        "func_files": func_files,
        "func_init_files": func_init_files,
        "func_signatures": func_signatures,
    }
    write_status(experiment_dir, 0, "file_generation", file_gen_status)

    # Backward-compat artifact for older scripts that still inspect this path.
    with open(os.path.join(experiment_dir, "stage_file_generation_status.json"), "w", encoding="utf-8") as f:
        json.dump(file_gen_status, f, indent=2)

    # Initialize pipeline state for separate job execution
    update_state(
        experiment_dir,
        function_implementation_total=len(terminals),
        function_implementation_remaining=len(terminals),
        dsl_round=0,
        func_evolution_round=args.func_evolution_round,
        test_tasks_submitted=1,
        phase="baseline",
        tasks=[task["title"] for task in TASKS],
    )

    print(f"\n[Submit] Launching {len(terminals)} FunSearch+Explicit Feedback jobs...")
    job_prefix = args.job_prefix or os.path.basename(experiment_dir)[:20]
    scripts_dir = "scripts/stages"
    impl_job_time = os.environ.get("IMPLEMENT_CFG_SINGLE_TIME", "").strip()
    if impl_job_time:
        print(f"  Using per-function implement job time override: {impl_job_time}")

    # Resource profiles: API mode uses no GPUs and much less RAM.
    # Each per-function job is mostly I/O-bound (waiting on API responses) with
    # lightweight craft-env evaluation, so 4 CPUs / 8 GB is sufficient.
    using_api = args.model_type == "openai_compat"
    if using_api:
        impl_cpus = os.environ.get("IMPL_CPUS_API", "4")
        impl_mem  = os.environ.get("IMPL_MEM_API", "8G")
        impl_gres = "gpu:0"
        impl_time_default = "04:00:00"
        print(f"  API mode: per-function jobs will use {impl_cpus} CPUs, {impl_mem} RAM, no GPU")
    else:
        impl_cpus = impl_mem = impl_gres = impl_time_default = None  # use script defaults

    submitted = 0
    submitted_job_ids = []

    for func_name in terminals.keys():
        env_vars = {
            "EXPERIMENT_DIR": experiment_dir,
            "SPEC_FILE": args.spec_file,
            "MODEL_TYPE": args.model_type,
            "FUNC_EVOLUTION_ROUND": str(args.func_evolution_round),
            "FUNCTION_NAME": func_name,
            "TOTAL_SAMPLES": str(args.total_samples),
            "NUM_EXPLICIT_FEEDBACK_ITERATIONS": str(args.num_explicit_feedback_iterations),
            "BASELINE_MODE": "1",
            "SKIP_CHAINING": "1",
            "DISABLE_LLM_VERIFIER": "1",
        }
        if args.openai_compat_key_file:
            env_vars["OPENAI_COMPAT_KEY_FILE"] = args.openai_compat_key_file
        env_str = ",".join([f"{k}={v}" for k, v in env_vars.items()])
        job_name = f"{job_prefix}_impl_{sanitize_function_name(func_name)}"
        submit_cmd = [
            "sbatch", "--parsable", "--export", f"ALL,{env_str}",
            "--job-name", job_name,
            "--output", os.path.join(baseline_log_dir, f"stage_implement_cfg_baseline_impl_{sanitize_function_name(func_name)}_%j.out"),
            "--error", os.path.join(baseline_log_dir, f"stage_implement_cfg_baseline_impl_{sanitize_function_name(func_name)}_%j.err"),
        ]
        if using_api:
            submit_cmd.extend([
                "--cpus-per-task", impl_cpus,
                "--mem", impl_mem,
                "--gres", impl_gres,
                "--time", impl_job_time or impl_time_default,
            ])
        elif impl_job_time:
            submit_cmd.extend(["--time", impl_job_time])
        submit_cmd.append(f"{scripts_dir}/stage_implement_cfg_single.slurm")

        result = subprocess.run(
            submit_cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            submitted += 1
            raw_job = result.stdout.strip()
            print(f"  Submitted {func_name}: {raw_job}")
            m = re.search(r"\d+", raw_job)
            if m:
                submitted_job_ids.append(m.group(0))
        else:
            print(f"  Failed {func_name}: {result.stderr.strip()}", file=sys.stderr)

    # Submit one final evaluation job that runs after all baseline function jobs complete.
    # This provides a single solved/not-solved report using final functions from explicit feedback.
    if submitted_job_ids:
        dep = ":".join(submitted_job_ids)
        wrap_cmd = (
            ". new_dsl_env/bin/activate && "
            "cd /home/avani/projects/aip-lelis/avani/DSL_Generator && "
            "python src/baseline.py "
            f"--spec_file {shlex.quote(args.spec_file)} "
            f"--experiment_dir {shlex.quote(experiment_dir)} "
            "--final_eval_only "
            "--recipes_path craft/resources/recipes.yaml "
            "--hints_path craft/resources/hints.yaml "
            "--max_steps 400 "
            f"--test_seeds {' '.join(shlex.quote(str(s)) for s in args.test_seeds)} "
            f"--func_evolution_round {int(args.func_evolution_round)}"
        )

        # Final evaluation only runs the craft game environment — no LLM involved.
        # Use lightweight resources regardless of model_type.
        final_eval_cpus = os.environ.get("FINAL_EVAL_CPUS", "8")
        final_eval_mem  = os.environ.get("FINAL_EVAL_MEM", "16G")
        eval_submit = subprocess.run(
            [
                "sbatch",
                "--parsable",
                "--dependency",
                f"afterany:{dep}",
                "--job-name",
                f"{job_prefix}_final_test",
                "--output",
                os.path.join(baseline_log_dir, "stage_baseline_final_test_%j.out"),
                "--error",
                os.path.join(baseline_log_dir, "stage_baseline_final_test_%j.err"),
                "--time",
                os.environ.get("BASELINE_FINAL_EVAL_TIME", "01:00:00"),
                "--cpus-per-task",
                final_eval_cpus,
                "--mem",
                final_eval_mem,
                "--gres",
                "gpu:0",
                "--account",
                "aip-lelis",
                "--wrap",
                _bash_wrap(wrap_cmd),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if eval_submit.returncode == 0:
            print(f"\n[Submit] Final baseline test job submitted: {eval_submit.stdout.strip()}")
            print("  This will run once all baseline function jobs finish and report solved/not-solved tasks.")
        else:
            print(
                f"\n[Submit] Failed to submit final baseline test job: {eval_submit.stderr.strip()}",
                file=sys.stderr,
            )

        if args.baseline_variant == BASELINE_VARIANT_TWO_PHASE:
            test_results_path = os.path.join(experiment_dir, "results_tracking", "test_py_phase1.json")
            test_wrap_cmd = (
                ". new_dsl_env/bin/activate && "
                "cd /home/avani/projects/aip-lelis/avani/DSL_Generator && "
                "python test.py "
                f"--final_functions_dir {shlex.quote(os.path.join(experiment_dir, 'final_functions'))} "
                f"--output_json {shlex.quote(test_results_path)}"
            )
            test_submit = subprocess.run(
                [
                    "sbatch",
                    "--parsable",
                    "--dependency",
                    f"afterany:{dep}",
                    "--job-name",
                    f"{job_prefix}_phase1_testpy",
                    "--output",
                    os.path.join(baseline_log_dir, "stage_baseline_phase1_testpy_%j.out"),
                    "--error",
                    os.path.join(baseline_log_dir, "stage_baseline_phase1_testpy_%j.err"),
                    "--time",
                    "04:00:00",
                    "--cpus-per-task",
                    "16",
                    "--mem",
                    "64G",
                    "--gres",
                    "gpu:1",
                    "--account",
                    "aip-lelis",
                    "--wrap",
                    _bash_wrap(test_wrap_cmd),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            if test_submit.returncode == 0:
                print(f"\n[Submit] Phase-1 test.py job submitted: {test_submit.stdout.strip()}")
                m2 = re.search(r"\d+", test_submit.stdout.strip())
                if m2:
                    test_job_id = m2.group(0)
                    phase2_round = int(args.func_evolution_round) + 1
                    phase2_job_prefix = f"{job_prefix}_p2"
                    phase2_wrap_cmd = (
                        ". new_dsl_env/bin/activate && "
                        "cd /home/avani/projects/aip-lelis/avani/DSL_Generator && "
                        "python src/baseline.py "
                        f"--spec_file {shlex.quote(args.spec_file)} "
                        f"--experiment_dir {shlex.quote(experiment_dir)} "
                        f"--model_type {shlex.quote(args.model_type)} "
                        f"--func_evolution_round {phase2_round} "
                        f"--total_samples {int(args.total_samples)} "
                        f"--num_explicit_feedback_iterations {int(args.num_explicit_feedback_iterations)} "
                        f"--job_prefix {shlex.quote(phase2_job_prefix)} "
                        f"--baseline_variant {BASELINE_VARIANT_TASK_ENV}"
                    )
                    if args.grid_prompt:
                        phase2_wrap_cmd += f" --grid_prompt {shlex.quote(args.grid_prompt)}"
                    if args.cfg_text:
                        phase2_wrap_cmd += f" --cfg_text {shlex.quote(args.cfg_text)}"
                    if args.nld_path:
                        phase2_wrap_cmd += f" --nld_path {shlex.quote(args.nld_path)}"
                    if args.codebase_path:
                        phase2_wrap_cmd += f" --codebase_path {shlex.quote(args.codebase_path)}"
                    if args.openai_compat_key_file:
                        phase2_wrap_cmd += f" --openai_compat_key_file {shlex.quote(args.openai_compat_key_file)}"

                    phase2_submit = subprocess.run(
                        [
                            "sbatch",
                            "--parsable",
                            "--dependency",
                            f"afterany:{test_job_id}",
                            "--job-name",
                            f"{job_prefix}_phase2_seeded",
                            "--output",
                            os.path.join(baseline_log_dir, "stage_baseline_phase2_seeded_%j.out"),
                            "--error",
                            os.path.join(baseline_log_dir, "stage_baseline_phase2_seeded_%j.err"),
                            "--time",
                            "04:00:00",
                            "--cpus-per-task",
                            "32",
                            "--mem",
                            "256G",
                            "--gres",
                            "gpu:4",
                            "--account",
                            "aip-lelis",
                            "--wrap",
                            _bash_wrap(phase2_wrap_cmd),
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if phase2_submit.returncode == 0:
                        print(f"[Submit] Phase-2 seeded baseline orchestrator submitted: {phase2_submit.stdout.strip()}")
                    else:
                        print(f"[Submit] Failed to submit phase-2 seeded baseline orchestrator: {phase2_submit.stderr.strip()}", file=sys.stderr)
            else:
                print(f"\n[Submit] Failed to submit phase-1 test.py job: {test_submit.stderr.strip()}", file=sys.stderr)

        if args.baseline_variant == BASELINE_VARIANT_TASK_ENV and int(args.task_env_rounds) > 1:
            next_round = int(args.func_evolution_round) + 1
            remaining_rounds = int(args.task_env_rounds) - 1
            next_job_prefix = f"{job_prefix}_r{next_round}"
            next_wrap_cmd = (
                ". new_dsl_env/bin/activate && "
                "cd /home/avani/projects/aip-lelis/avani/DSL_Generator && "
                "python src/baseline.py "
                f"--spec_file {shlex.quote(args.spec_file)} "
                f"--experiment_dir {shlex.quote(experiment_dir)} "
                f"--model_type {shlex.quote(args.model_type)} "
                f"--func_evolution_round {next_round} "
                f"--total_samples {int(args.total_samples)} "
                f"--num_explicit_feedback_iterations {int(args.num_explicit_feedback_iterations)} "
                f"--job_prefix {shlex.quote(next_job_prefix)} "
                f"--baseline_variant {BASELINE_VARIANT_TASK_ENV} "
                f"--task_env_rounds {remaining_rounds}"
            )
            if args.grid_prompt:
                next_wrap_cmd += f" --grid_prompt {shlex.quote(args.grid_prompt)}"
            if args.cfg_text:
                next_wrap_cmd += f" --cfg_text {shlex.quote(args.cfg_text)}"
            if args.nld_path:
                next_wrap_cmd += f" --nld_path {shlex.quote(args.nld_path)}"
            if args.codebase_path:
                next_wrap_cmd += f" --codebase_path {shlex.quote(args.codebase_path)}"
            if args.phase2_only:
                next_wrap_cmd += " --phase2_only"
            if args.phase2_seed_round is not None:
                next_wrap_cmd += f" --phase2_seed_round {int(args.phase2_seed_round)}"
            if args.openai_compat_key_file:
                next_wrap_cmd += f" --openai_compat_key_file {shlex.quote(args.openai_compat_key_file)}"

            chain_submit = subprocess.run(
                [
                    "sbatch",
                    "--parsable",
                    "--dependency",
                    f"afterany:{dep}",
                    "--job-name",
                    f"{job_prefix}_next_round",
                    "--output",
                    os.path.join(baseline_log_dir, "stage_baseline_taskenv_chain_%j.out"),
                    "--error",
                    os.path.join(baseline_log_dir, "stage_baseline_taskenv_chain_%j.err"),
                    "--time",
                    "04:00:00",
                    "--cpus-per-task",
                    "32",
                    "--mem",
                    "256G",
                    "--gres",
                    "gpu:4",
                    "--account",
                    "aip-lelis",
                    "--wrap",
                    _bash_wrap(next_wrap_cmd),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if chain_submit.returncode == 0:
                print(
                    f"[Submit] Chained next task_env baseline round submitted: {chain_submit.stdout.strip()} "
                    f"(next func round={next_round}, remaining rounds={remaining_rounds})"
                )
            else:
                print(
                    f"[Submit] Failed to submit chained next task_env round: {chain_submit.stderr.strip()}",
                    file=sys.stderr,
                )

    print(f"\n Submitted {submitted}/{len(terminals)} jobs")
    return 0 if submitted else 1


if __name__ == "__main__":
    sys.exit(main())
