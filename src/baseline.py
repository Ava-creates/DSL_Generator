#!/usr/bin/env python3
"""
Baseline runner: build prompts for env_task_list tasks, run FunSearch,
and generate explicit feedback outputs without CFG generation.
"""

import argparse
import json
import os
import sys
import subprocess

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from src.pipeline.cfg_to_funsearch_pipeline import (
    sanitize_function_name,
)
from src.utils.pipeline_state import update_state
from src.pipeline.domain_templates import (
    craft_solve_template_basic,
    craft_evaluate_template,
)


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


def build_prompt_content(task, experiment_dir, dsl_round, func_round):
    display_name = task["title"]
    description = task["description"]
    safe_name = sanitize_function_name(display_name)

    func_params = "env"
    func_call_args = "env"

    solve_func = craft_solve_template_basic(
        func_name=safe_name,
        func_params=func_params,
        func_call_args=func_call_args,
    )

    task_name = task_name_from_title(display_name)
    env_setup = f"""
  from craft import env_factory
  recipes_path = "craft/resources/recipes.yaml"
  hints_path = "craft/resources/hints.yaml"
  env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 7, max_steps=300, reuse_environments=False,
            visualise=visualise)
  env = env_sampler.sample_environment(task_name="{task_name}")
  """

    eval_func = craft_evaluate_template(
        display_name=display_name,
        env_setup=env_setup,
        args_definitions="",
        func_call_args=func_call_args,
    )

    evolve_func = f'''@funsearch.evolve
def {safe_name}({func_params}):
  """
  {description}
  
  Args:
      env: The current environment instance.
  Returns:
      List[int]: A sequence of encoded actions the agent should execute.
  """
'''

    prompt_content = solve_func + "\n" + eval_func + "\n" + evolve_func
    func_signature = f"def {safe_name}({func_params})"
    return prompt_content, func_signature


def main():
    parser = argparse.ArgumentParser(description="Baseline FunSearch runner for env_task_list tasks")
    parser.add_argument("--spec_file", type=str, required=True, help="Path to specification file")
    parser.add_argument("--experiment_dir", type=str, default="experiments/baseline", help="Experiment directory")
    parser.add_argument("--model_type", type=str, default="huggingface", choices=["huggingface", "ollama", "gemini"])
    parser.add_argument("--dsl_round", type=int, default=0, help="DSL round index (for naming)")
    parser.add_argument("--func_evolution_round", type=int, default=0, help="Function round index (for naming)")
    parser.add_argument("--total_samples", type=int, default=2000, help="Total FunSearch samples per task")
    parser.add_argument("--num_explicit_feedback_iterations", type=int, default=30,
                        help="Number of explicit feedback iterations per task")
    parser.add_argument("--job_prefix", type=str, default=None, help="Optional job name prefix for SLURM")
    args = parser.parse_args()

    if not os.path.exists(args.spec_file):
        print(f" Specification file not found: {args.spec_file}", file=sys.stderr)
        return 1

    experiment_dir = args.experiment_dir
    os.makedirs(experiment_dir, exist_ok=True)

    os.makedirs(os.path.join(experiment_dir, "function_specific_prompts"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "functions_generated"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "results", "funsearch"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "explicit_feedback"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "final_functions"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "cfg"), exist_ok=True)

    # Save a minimal cfg_output.json for record-keeping
    terminals = {task["title"]: task["description"] for task in TASKS}
    cfg_path = os.path.join(experiment_dir, "cfg", "cfg_output.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({"cfg": "DUMMY ::= DUMMY", "terminals": terminals, "example": None}, f, indent=2)

    # Generate function-specific prompt and init files
    func_files = {}
    func_init_files = {}
    func_signatures = {}
    for task in TASKS:
        display_name = task["title"]
        safe_name = sanitize_function_name(display_name)

        prompt_content, func_signature = build_prompt_content(
            task, experiment_dir, args.dsl_round, args.func_evolution_round
        )
        func_signatures[display_name] = func_signature

        prompt_file = os.path.join(
            experiment_dir,
            "function_specific_prompts",
            f"{safe_name}_dsl{args.dsl_round}_func{args.func_evolution_round}.txt",
        )
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt_content)
        func_files[display_name] = prompt_file

        func_init_file = os.path.join(
            experiment_dir,
            "functions_generated",
            f"{safe_name}_dsl{args.dsl_round}_func{args.func_evolution_round}_func_init.py",
        )
        with open(func_init_file, "w", encoding="utf-8") as f:
            f.write(f"def {safe_name}(env):\n    return []\n")
        func_init_files[display_name] = func_init_file

    file_gen_status = {
        "stage": "file_generation",
        "status": "completed",
        "dsl_round": args.dsl_round,
        "func_evolution_round": args.func_evolution_round,
        "func_files": func_files,
        "func_init_files": func_init_files,
        "func_signatures": func_signatures,
    }
    with open(os.path.join(experiment_dir, "stage_file_generation_status.json"), "w", encoding="utf-8") as f:
        json.dump(file_gen_status, f, indent=2)

    # Initialize pipeline state for separate job execution
    update_state(
        experiment_dir,
        function_implementation_total=len(terminals),
        function_implementation_remaining=len(terminals),
        dsl_round=args.dsl_round,
        func_evolution_round=args.func_evolution_round,
        test_tasks_submitted=1,
        phase="baseline",
        tasks=[task["title"] for task in TASKS],
    )

    print(f"\n[Submit] Launching {len(terminals)} FunSearch+Explicit Feedback jobs...")
    job_prefix = args.job_prefix or os.path.basename(experiment_dir)[:20]
    scripts_dir = "scripts/stages"
    submitted = 0

    for func_name in terminals.keys():
        env_vars = {
            "EXPERIMENT_DIR": experiment_dir,
            "SPEC_FILE": args.spec_file,
            "MODEL_TYPE": args.model_type,
            "DSL_ROUND": str(args.dsl_round),
            "FUNC_EVOLUTION_ROUND": str(args.func_evolution_round),
            "FUNCTION_NAME": func_name,
            "TOTAL_SAMPLES": str(args.total_samples),
            "NUM_EXPLICIT_FEEDBACK_ITERATIONS": str(args.num_explicit_feedback_iterations),
            "SKIP_CHAINING": "1",
            "DISABLE_LLM_VERIFIER": "1",
        }
        env_str = ",".join([f"{k}={v}" for k, v in env_vars.items()])
        job_name = f"{job_prefix}_impl_{sanitize_function_name(func_name)}"
        result = subprocess.run(
            [
                "sbatch", "--parsable", "--export", f"ALL,{env_str}",
                "--job-name", job_name,
                f"{scripts_dir}/stage_implement_cfg_single.slurm",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            submitted += 1
            print(f"  Submitted {func_name}: {result.stdout.strip()}")
        else:
            print(f"  Failed {func_name}: {result.stderr.strip()}", file=sys.stderr)

    print(f"\n Submitted {submitted}/{len(terminals)} jobs")
    return 0 if submitted else 1


if __name__ == "__main__":
    sys.exit(main())
