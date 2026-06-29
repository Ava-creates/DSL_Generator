#!/usr/bin/env python3
"""Stage 1 (domain-aware): Get CFG for the selected domain."""

import argparse
import json
import os
import sys

_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from domains.registry import get_adapter, list_domains
from src.utils.pipeline_state import update_state
from src_agnostic.pipeline.cfg_to_funsearch_pipeline import get_cfg

try:
    from vllm import LLM as vLLM
except ImportError:  # pragma: no cover
    vLLM = None


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1: Get CFG (domain-agnostic)")
    parser.add_argument("--domain", type=str, required=True, choices=list_domains())
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument("--skip_cfg_generation", action="store_true")
    parser.add_argument("--cfg_output_file", type=str, default=None)
    parser.add_argument("--max_cfg_retries", type=int, default=int(os.environ.get("MAX_CFG_RETRIES", "10")))
    parser.add_argument("--cfg_generator_prompt_path", type=str,
                        default=os.environ.get("CFG_GENERATOR_PROMPT_PATH", "prompt_specifications/cfg_generator.txt"))
    parser.add_argument("--domain_context_template_path", type=str,
                        default=os.environ.get("DOMAIN_CONTEXT_TEMPLATE_PATH"))
    parser.add_argument("--nld_path", type=str, default=None,
                        help="Override NLD path used by the adapter")
    parser.add_argument("--recipes_path", type=str, default=None,
                        help="Craft only: recipes YAML path")
    parser.add_argument("--hints_path", type=str, default=None,
                        help="Craft only: hints YAML path")

    args = parser.parse_args()

    if not args.experiment_dir or args.experiment_dir.strip() == "":
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_root = "experiments"
        os.makedirs(base_root, exist_ok=True)
        args.experiment_dir = os.path.join(base_root, f"experiment_{timestamp}")
        print(f"Auto-generated experiment directory: {args.experiment_dir}", flush=True)

    adapter_kwargs = {}
    if args.domain == "craft":
        if args.recipes_path:
            adapter_kwargs["recipes_path"] = args.recipes_path
        if args.hints_path:
            adapter_kwargs["hints_path"] = args.hints_path
        if args.nld_path:
            adapter_kwargs["nld_path"] = args.nld_path
    elif args.domain == "crafter" and args.nld_path:
        adapter_kwargs["nld_path"] = args.nld_path

    adapter = get_adapter(args.domain, **adapter_kwargs)

    shared_vllm = None
    if not args.skip_cfg_generation:
        model_type = os.environ.get("MODEL_TYPE", "huggingface").strip().lower()
        if model_type == "openai_compat":
            from src.pipeline.explicit_feedback_generation import OpenAICompatLLMWrapper

            key_file = os.environ.get("OPENAI_COMPAT_KEY_FILE", "").strip() or None
            shared_vllm = OpenAICompatLLMWrapper(key_file)
        elif vLLM is not None:
            try:
                shared_vllm = vLLM(model="/scratch/avani/gpt", tensor_parallel_size=4)
            except Exception as e:
                print(f" Warning: Could not create shared vLLM instance: {e}")
                shared_vllm = None

    os.makedirs(args.experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(args.experiment_dir, "cfg"), exist_ok=True)

    cfg, terminals, example, success = get_cfg(
        adapter=adapter,
        experiment_dir=args.experiment_dir,
        skip_cfg_generation=args.skip_cfg_generation,
        cfg_output_file=args.cfg_output_file,
        max_cfg_retries=args.max_cfg_retries,
        cfg_generator_prompt_path=args.cfg_generator_prompt_path,
        domain_context_template_path=args.domain_context_template_path,
        shared_vllm=shared_vllm,
    )

    if not success or not cfg or not terminals:
        print(" Failed to get valid CFG", file=sys.stderr)
        return 1

    print(f"\n Got CFG with {len(terminals)} terminal functions")

    tasks_env = os.environ.get("TASKS", "")
    if tasks_env:
        try:
            tasks = json.loads(tasks_env)
        except Exception:
            tasks = tasks_env.split()
    else:
        tasks = list(adapter.spec.tasks[:1]) or [""]

    update_state(
        args.experiment_dir,
        function_implementation_total=len(terminals),
        function_implementation_remaining=len(terminals),
        test_tasks_total=len(tasks),
        tasks=json.dumps(tasks) if tasks else "[]",
        phase="initial",
        dsl_round=0,
        max_dsl_evolutions=int(os.environ.get("MAX_DSL_EVOLUTIONS", "2")),
        dsl_evolutions_remaining=int(os.environ.get("MAX_DSL_EVOLUTIONS", "2")),
        func_evolution_round=0,
        max_function_evolutions=int(os.environ.get("MAX_FUNCTION_EVOLUTIONS", "1")),
        domain_name=adapter.spec.name,
    )

    stage_status = {
        "stage": "get_cfg",
        "status": "completed",
        "cfg_path": os.path.join(args.experiment_dir, "cfg", "cfg_output.json"),
        "num_terminals": len(terminals),
        "domain": adapter.spec.name,
    }
    status_file = os.path.join(args.experiment_dir, "stage_get_cfg_status.json")
    with open(status_file, "w") as f:
        json.dump(stage_status, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
