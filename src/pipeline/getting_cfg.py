from google import genai
from typing import Tuple, Dict, Optional as Opt
import sys
import re
import json
import ast
import os
from pydantic import BaseModel



class Output(BaseModel):
    CFG: str
    terminal_functions_and_their_description: Dict[str, str]
    example_program: list[str]

json_schema = Output.model_json_schema()
# print(json_schema)

from vllm import SamplingParams
from vllm import LLM as vLLM



def CFGPromptBuilder(
    nld_path: str = "prompt_specifications/nld.txt",
    recipes_path: Opt[str] = None,
    prompt_template_path: str = "prompt_specifications/cfg_generator.txt",
    domain_context_template_path: Opt[str] = None,
):
    """Builds the prompt string given domain description .

    This isolates prompt formatting so changes to wording don't affect other code.
    """
    nld_text = ""
    if nld_path and os.path.exists(nld_path):
        with open(nld_path, 'r', encoding='utf-8') as f:
            nld_text = f.read().strip()

    recipes_text = ""
    if recipes_path and os.path.exists(recipes_path):
        with open(recipes_path, 'r', encoding='utf-8') as f:
            recipes_text = f.read().strip()

    domain_context = ""
    if domain_context_template_path:
        if not os.path.exists(domain_context_template_path):
            raise FileNotFoundError(f"Domain context template not found: {domain_context_template_path}")
        with open(domain_context_template_path, 'r', encoding='utf-8') as f:
            domain_context_template = f.read()
        domain_context = domain_context_template.format(
            nld=nld_text,
            recipes=recipes_text,
        ).strip()

    if not domain_context:
        domain_context = "No additional domain information provided."

    if not os.path.exists(prompt_template_path):
        raise FileNotFoundError(f"CFG prompt template not found: {prompt_template_path}")
    with open(prompt_template_path, 'r', encoding='utf-8') as f:
        template = f.read()

    return template.format(
        nld=nld_text,
        domain_context=domain_context,
    )


class GenAIWrapper:
    """Wrapper for the `genai.Client` to allow easier testing and dependency injection.

    Keeps a very small interface: generate(prompt) -> str
    """

    def __init__(self, client: Opt[genai.Client] = None, vllm_instance: Opt[vLLM] = None) -> None:

        self._client = client or genai.Client()

        if vllm_instance is not None:
            # Use provided vLLM instance
            self.llm = vllm_instance
        elif client == "vllm":
            # Create new vLLM instance if not provided
            self.llm = vLLM(model="/scratch/avani/gpt",    tensor_parallel_size=4 )

    def generate(self, prompt: str, model: str = "vllm") -> str:
        # Keep the original call semantics but guard against errors.
        if model == "vllm":
            # Ensure llm is set
            if not hasattr(self, 'llm') or self.llm is None:
                raise ValueError("vLLM instance not available. Either provide vllm_instance or set client='vllm'")
            
            self.params = SamplingParams(temperature=0.7, max_tokens=35000)
            output = self.llm.generate([prompt], sampling_params=self.params)
            response = output[0].outputs[0].text
            print(response)
            return response
        else:
            response = self._client.models.generate_content(model=model, contents=prompt)
        # Some versions may use .text or .response; favor .text if present.
        return getattr(response, "text", getattr(response, "response", str(response)))



class CFGGenerator:
    """Orchestrates building prompt, calling the LLM, and returning text."""

    def __init__(
        self,
        llm: GenAIWrapper,
        nld_path: str = "prompt_specifications/nld.txt",
        recipes_path: Opt[str] = None,
        prompt_template_path: str = "prompt_specifications/cfg_generator.txt",
        domain_context_template_path: Opt[str] = None,
    ) -> None:
        self.llm = llm
        self.nld_path = nld_path
        self.recipes_path = recipes_path
        self.prompt_template_path = prompt_template_path
        self.domain_context_template_path = domain_context_template_path

    def generate_cfg(self) -> str:
        prompt = CFGPromptBuilder(
            nld_path=self.nld_path,
            recipes_path=self.recipes_path,
            prompt_template_path=self.prompt_template_path,
            domain_context_template_path=self.domain_context_template_path,
        )
        return self.llm.generate(prompt)


# ---------------------- Parsing Utilities ----------------------

FENCE_RE = re.compile(r"```\s*([a-zA-Z0-9_]*)\s*\n(.*?)\n```", re.DOTALL | re.MULTILINE)


def _extract_fenced_blocks(text: str):
    """Return list of tuples: (lang, content) for each fenced code block."""
    blocks = []
    for m in FENCE_RE.finditer(text):
        lang = m.group(1).strip().lower() if m.group(1) else ""
        content = m.group(2).strip()
        blocks.append((lang, content))
    return blocks


def _pick_cfg_block(blocks, full_text: str) -> Opt[str]:
    # Prefer blocks marked as bnf/ebnf/grammar that contain '::='
    preferred_langs = {"bnf", "ebnf", "grammar"}
    for lang, content in blocks:
        if lang in preferred_langs and "::=" in content:
            return content
    # Fallback: first block that looks like grammar (contains ::=)
    for _, content in blocks:
        if "::=" in content:
            return content
    # Last resort: scan plain text lines
    lines = []
    started = False
    for line in full_text.splitlines():
        if "::=" in line:
            started = True
        if started:
            if line.strip() == "" and lines:
                break
            lines.append(line)
    return "\n".join(lines).strip() if lines else None


def _parse_json_or_python_dict(text: str) -> Opt[Dict[str, str]]:
    # Try pure JSON first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            # Ensure string values
            return {str(k): (str(v) if not isinstance(v, str) else v) for k, v in obj.items()}
    except Exception:
        pass
    # Try extracting the largest {...} region and parse as JSON
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            cand = text[start : end + 1]
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict):
                    return {str(k): (str(v) if not isinstance(v, str) else v) for k, v in obj.items()}
            except Exception:
                pass
            # Fallback: Python literal eval
            obj = ast.literal_eval(cand)
            if isinstance(obj, dict):
                return {str(k): (str(v) if not isinstance(v, str) else v) for k, v in obj.items()}
    except Exception:
        pass
    return None


def _pick_terminals_dict(blocks, full_text: str) -> Dict[str, str]:
    # Prefer blocks labelled json or python / dict-like content
    for lang, content in blocks:
        if lang in {"json", "javascript"} or (content.strip().startswith("{") and content.strip().endswith("}")):
            parsed = _parse_json_or_python_dict(content)
            if parsed is not None:
                return parsed
    # Try any block with braces
    for _, content in blocks:
        if "{" in content and "}" in content:
            parsed = _parse_json_or_python_dict(content)
            if parsed is not None:
                return parsed
    # Line-based fallback: detect TERMINAL: description pairs
    pairs: Dict[str, str] = {}
    line_re = re.compile(r"^\s*([A-Z_][A-Z0-9_()]*)\s*[:\-–]\s*(.+?)\s*$")
    for line in full_text.splitlines():
        m = line_re.match(line)
        if m:
            key = m.group(1).strip()
            val = m.group(2).strip().rstrip(";.")
            pairs[key] = val
    return pairs


def _pick_example_program(blocks) -> Opt[str]:
    # Choose the last block that isn't the CFG or dict based on simple heuristics
    for lang, content in reversed(blocks):
        if "::=" in content:
            continue
        if content.strip().startswith("{") and content.strip().endswith("}"):
            continue
        # Looks like a program
        return content.strip()
    return None


def parse_generated_output(text: str) -> Tuple[str, Dict[str, str], Opt[str]]:
    """
    Parse LLM output and return (cfg_string, terminals_dict, example_program_or_None).
    Robust to multiple formats; prefers fenced blocks when present.
    """
    blocks = _extract_fenced_blocks(text)
    cfg = _pick_cfg_block(blocks, text) or ""
    terminals = _pick_terminals_dict(blocks, text)
    example = _pick_example_program(blocks)
    return cfg, terminals, example


def generate_and_parse_cfg(
    vllm_instance: Opt[vLLM] = None,
    nld_path: str = "prompt_specifications/nld.txt",
    recipes_path: Opt[str] = None,
    prompt_template_path: str = "prompt_specifications/cfg_generator.txt",
    domain_context_template_path: Opt[str] = None,
) -> Tuple[str, Dict[str, str], Opt[str]]:
    """
    Generate CFG and return parsed results as a tuple.
    
    Args:
        vllm_instance: Optional vLLM instance to reuse (if None, creates a new one)
    
    Returns: (cfg_string, terminals_dict, example_program_or_None)
    """
    if not os.path.exists(nld_path):
        raise FileNotFoundError(f"NLD file not found: {nld_path}")
    if recipes_path and not os.path.exists(recipes_path):
        raise FileNotFoundError(f"Recipes file not found: {recipes_path}")
    if not os.path.exists(prompt_template_path):
        raise FileNotFoundError(f"CFG prompt template not found: {prompt_template_path}")
    if domain_context_template_path and not os.path.exists(domain_context_template_path):
        raise FileNotFoundError(f"Domain context template not found: {domain_context_template_path}")

    llm_wrapper = GenAIWrapper("vllm", vllm_instance=vllm_instance)
    generator = CFGGenerator(
        llm_wrapper,
        nld_path=nld_path,
        recipes_path=recipes_path,
        prompt_template_path=prompt_template_path,
        domain_context_template_path=domain_context_template_path,
    )
    
    output = generator.generate_cfg()
    cfg, terminals, example = parse_generated_output(output)
    return cfg, terminals, example


def main() -> int:
    try:
        cfg, terminals, example = generate_and_parse_cfg()

        # Print parsed results
        print("CFG (BNF):")
        print(cfg)
        print()
        print("Terminal Functions (JSON):")
        print(json.dumps(terminals, indent=2, ensure_ascii=False))
        if example:
            print()
            print("Example Program:")
            print(example)
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise_code = main()
    sys.exit(raise_code)