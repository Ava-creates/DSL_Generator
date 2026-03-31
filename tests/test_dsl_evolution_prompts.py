#!/usr/bin/env python3

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from vllm import LLM, SamplingParams

from src.pipeline.cfg_parser import CFGParser

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FAILURE_PROMPT_PATH = _PROJECT_ROOT / "prompt_specifications" / "failure_analysis.txt"
DEFAULT_CFG_PROMPT_PATH = _PROJECT_ROOT / "prompt_specifications" / "cfg_evolution.txt"
DEFAULT_RECIPES_PATH = _PROJECT_ROOT / "craft" / "resources" / "recipes.yaml"
DEFAULT_FAILED_PROGRAMS_PATH = _PROJECT_ROOT / "scripts" / "data" / "dsl0_failed_programs.txt"
DEFAULT_NLD_PATH = _PROJECT_ROOT / "prompt_specifications" / "nld.txt"


class SafeFormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


@dataclass
class PromptContext:
    nld: str
    failing_tasks: List[str]
    cfg: str
    recipes: str
    terminal_descriptions: str
    final_function_descriptions: str
    failed_programs_per_task: str
    prompt_plugin_info: str


HARD_CODED_CONTEXT = PromptContext(
    nld="",
    failing_tasks=[
        "get[gem]",
        "get[grass]",
        "get[gold]",
        "make[cloth]",
        "make[bridge]",
        "make[bed]",
        "make[shears]",
        "make[ladder]",
        "make[goldarrow]",
    ],
    cfg="""program        ::= statement_seq

statement_seq  ::= statement
                |  statement SEMICOLON statement_seq

statement      ::= MOVE LPAR DIR RPAR
                |  TURN LPAR TURN_DIR RPAR
                |  PICKUP
                |  USE LPAR TOOL RPAR
                |  CRAFT LPAR ITEM RPAR

DIR            ::= NORTH | SOUTH | EAST | WEST
TURN_DIR       ::= LEFT | RIGHT
TOOL           ::= AXE | HAMMER | KNIFE | SHEARS | STICK
ITEM           ::= AXE | SLINGSHOT | ARROW | GOLDARROW | BRIDGE | BUNDLE | HAMMER | KNIFE | BED | SHEARS | LADDER | BOW | BENCH | FLAG | PLANK | CLOTH | ROPE | STICK

SEMICOLON      ::= ';'
LPAR           ::= '('
RPAR           ::= ')'
COMMA          ::= ','""",
        recipes="",
    terminal_descriptions="""MOVE: Moves one step in a cardinal direction.
TURN: Rotates facing direction left or right.
PICKUP: Attempts to collect object in front of the agent.
USE: Applies a tool to the relevant nearby obstacle.
CRAFT: Crafts an item using available inventory at required workstation.""",
    final_function_descriptions="""## move_dsl0_func0.py
def move(env, dir):
    \"\"\"Navigates toward direction target and emits primitive actions.\"\"\"

## turn_dsl0_func0.py
def turn(env, turn_dir):
    \"\"\"Turns agent orientation left or right and emits primitive actions.\"\"\"

## pickup_dsl0_func0.py
def pickup(env):
    \"\"\"Attempts context-aware pickup and returns primitive actions.\"\"\"

## use_dsl0_func0.py
def use(env, tool):
    \"\"\"Uses provided tool in environment context and returns primitive actions.\"\"\"

## craft_dsl0_func0.py
def craft(env, item):
    \"\"\"Handles travel/resource/workstation logic to craft target item.\"\"\"""",
    failed_programs_per_task="",
    prompt_plugin_info="",
)


def _safe_format(template: str, values: Dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        value = values.get(key)
        if value is None:
            return match.group(0)
        return str(value)

    return re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", replace, template)


def _variant_label(path: Path) -> str:
    return path.stem.replace(" ", "_")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_cfg_block(text: str) -> str:
    if not text:
        return ""

    fenced_blocks = re.findall(r"```(?:bnf|cfg|grammar)?\s*\n(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    for block in fenced_blocks:
        if "::=" in block:
            return block.strip()

    lines = text.splitlines()
    cfg_lines: List[str] = []
    in_cfg = False
    for line in lines:
        if "::=" in line:
            in_cfg = True
            cfg_lines.append(line)
            continue
        if in_cfg:
            stripped = line.strip()
            if stripped.startswith("|") or stripped == "" or stripped.startswith("#"):
                cfg_lines.append(line)
            else:
                break
    return "\n".join(cfg_lines).strip()


def _validate_cfg_text(cfg_output: str) -> Dict[str, str]:
    extracted_cfg = _extract_cfg_block(cfg_output)
    if not extracted_cfg:
        return {
            "is_valid": False,
            "reason": "No CFG block containing '::=' found in cfg_output.",
            "start_symbol": "",
            "num_rules": 0,
            "extracted_cfg": "",
        }

    try:
        parser = CFGParser(extracted_cfg)
        return {
            "is_valid": True,
            "reason": "",
            "start_symbol": parser.start(),
            "num_rules": len(parser.rules),
            "extracted_cfg": extracted_cfg,
        }
    except Exception as e:
        return {
            "is_valid": False,
            "reason": str(e),
            "start_symbol": "",
            "num_rules": 0,
            "extracted_cfg": extracted_cfg,
        }


def _run_prompt_test(
    llm: LLM,
    params: SamplingParams,
    failure_template: str,
    cfg_template: str,
    context: PromptContext,
) -> Dict[str, str]:
    fmt = {
        "nld": context.nld,
        "failing_tasks": context.failing_tasks,
        "cfg": context.cfg,
        "recipes": context.recipes,
        "terminal_descriptions": context.terminal_descriptions,
        "final_function_descriptions": context.final_function_descriptions,
        "failed_programs_per_task": context.failed_programs_per_task,
        "prompt_plugin_info": context.prompt_plugin_info,
    }

    failure_prompt = _safe_format(failure_template, fmt)
    if "{prompt_plugin_info}" not in failure_template:
        failure_prompt = failure_prompt.rstrip() + "\n\n" + context.prompt_plugin_info

    failure_output = llm.chat(
        [{"role": "user", "content": failure_prompt, "chat_template_kwargs": {"reasoning_effort": "high"}}],
        params,
    )[0].outputs[0].text

    cfg_fmt = dict(fmt)
    cfg_fmt["failure_analysis"] = failure_output
    cfg_prompt = _safe_format(cfg_template, cfg_fmt)
    if "{prompt_plugin_info}" not in cfg_template:
        cfg_prompt = cfg_prompt.rstrip() + "\n\n" + context.prompt_plugin_info

    cfg_output = llm.chat(
        [{"role": "user", "content": cfg_prompt, "chat_template_kwargs": {"reasoning_effort": "high"}}],
        params,
    )[0].outputs[0].text

    return {
        "failure_prompt": failure_prompt,
        "failure_output": failure_output,
        "cfg_prompt": cfg_prompt,
        "cfg_output": cfg_output,
    }


def _load_variants(paths: List[str], default_name: str, default_template: str) -> List[Dict[str, str]]:
    if not paths:
        return [{"name": default_name, "template": default_template}]
    variants: List[Dict[str, str]] = []
    for raw in paths:
        path = Path(raw).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Prompt variant not found: {path}")
        variants.append({"name": _variant_label(path), "template": _read_text(path)})
    return variants


def _load_default_templates() -> Dict[str, str]:
    return {
        "failure": _read_text(DEFAULT_FAILURE_PROMPT_PATH),
        "cfg": _read_text(DEFAULT_CFG_PROMPT_PATH),
    }


def _load_nld_variants(paths: List[str], default_path: str) -> List[Dict[str, str]]:
    if not paths:
        path = Path(default_path).resolve()
        if not path.exists():
            return [{"name": "no_nld", "nld": ""}]
        return [{"name": _variant_label(path), "nld": _read_text(path)}]

    variants: List[Dict[str, str]] = []
    for raw in paths:
        path = Path(raw).resolve()
        if not path.exists():
            raise FileNotFoundError(f"NLD variant not found: {path}")
        variants.append({"name": _variant_label(path), "nld": _read_text(path)})
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Test DSL evolution prompt variants with one shared LLM instance using hardcoded context.")
    parser.add_argument("--failure-prompts", nargs="+", default=[])
    parser.add_argument("--cfg-prompts", nargs="+", default=[])
    parser.add_argument("--recipes-path", default=str(DEFAULT_RECIPES_PATH))
    parser.add_argument("--failed-programs-path", default=str(DEFAULT_FAILED_PROGRAMS_PATH))
    parser.add_argument("--nld-path", default=str(DEFAULT_NLD_PATH))
    parser.add_argument("--nld-paths", nargs="+", default=[])
    parser.add_argument("--model-path", default="/scratch/avani/gpt")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=25000)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    templates = _load_default_templates()
    recipes_text = _read_text(Path(args.recipes_path).resolve())
    failed_programs_text = _read_text(Path(args.failed_programs_path).resolve())
    base_context = PromptContext(
        failing_tasks=HARD_CODED_CONTEXT.failing_tasks,
        nld=HARD_CODED_CONTEXT.nld,
        cfg=HARD_CODED_CONTEXT.cfg,
        recipes=recipes_text,
        terminal_descriptions=HARD_CODED_CONTEXT.terminal_descriptions,
        final_function_descriptions=HARD_CODED_CONTEXT.final_function_descriptions,
        failed_programs_per_task=failed_programs_text,
        prompt_plugin_info=HARD_CODED_CONTEXT.prompt_plugin_info,
    )

    failure_variants = _load_variants(args.failure_prompts, "default_failure_spec", templates["failure"])
    cfg_variants = _load_variants(args.cfg_prompts, "default_cfg_spec", templates["cfg"])
    nld_variants = _load_nld_variants(args.nld_paths, args.nld_path)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (_PROJECT_ROOT / "experiments" / "prompt_tests" / ts)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "prompt_plugin_info.txt").write_text(base_context.prompt_plugin_info, encoding="utf-8")
    (out_dir / "context_summary.json").write_text(
        json.dumps(
            {
                "context_source": "hardcoded",
                "failing_tasks": base_context.failing_tasks,
                "failure_variants": [v["name"] for v in failure_variants],
                "cfg_variants": [v["name"] for v in cfg_variants],
                "nld_variants": [v["name"] for v in nld_variants],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Loading vLLM model from: {args.model_path}")
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size)
    params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    pairs = []
    for fv in failure_variants:
        for cv in cfg_variants:
            for nv in nld_variants:
                pairs.append((fv, cv, nv))

    print(f"Running {len(pairs)} prompt test combination(s)...")

    for idx, (failure_variant, cfg_variant, nld_variant) in enumerate(pairs, start=1):
        print(
            f"[{idx}/{len(pairs)}] failure={failure_variant['name']} cfg={cfg_variant['name']} nld={nld_variant['name']}"
        )
        context = PromptContext(
            failing_tasks=base_context.failing_tasks,
            nld=nld_variant["nld"],
            cfg=base_context.cfg,
            recipes=base_context.recipes,
            terminal_descriptions=base_context.terminal_descriptions,
            final_function_descriptions=base_context.final_function_descriptions,
            failed_programs_per_task=base_context.failed_programs_per_task,
            prompt_plugin_info=base_context.prompt_plugin_info,
        )
        outputs = _run_prompt_test(
            llm=llm,
            params=params,
            failure_template=failure_variant["template"],
            cfg_template=cfg_variant["template"],
            context=context,
        )
        cfg_validation = _validate_cfg_text(outputs["cfg_output"])

        run_name = f"{idx:02d}_{failure_variant['name']}__{cfg_variant['name']}__{nld_variant['name']}"
        run_dir = out_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "failure_prompt.txt").write_text(outputs["failure_prompt"], encoding="utf-8")
        (run_dir / "failure_output.txt").write_text(outputs["failure_output"], encoding="utf-8")
        (run_dir / "cfg_prompt.txt").write_text(outputs["cfg_prompt"], encoding="utf-8")
        (run_dir / "cfg_output.txt").write_text(outputs["cfg_output"], encoding="utf-8")
        (run_dir / "cfg_validation.json").write_text(json.dumps(cfg_validation, indent=2), encoding="utf-8")
        if cfg_validation["is_valid"]:
            print(f"    CFG validation: PASS (start={cfg_validation['start_symbol']}, rules={cfg_validation['num_rules']})")
        else:
            print(f"    CFG validation: FAIL ({cfg_validation['reason']})")

    print(f"Done. Results written to: {out_dir}")


if __name__ == "__main__":
    main()
