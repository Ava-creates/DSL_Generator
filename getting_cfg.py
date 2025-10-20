from google import genai
from typing import Tuple, Dict, Optional as Opt
import os
import sys
import re
import json
import ast
from pydantic import BaseModel
from enum import Enum
from vllm.sampling_params import StructuredOutputsParams
from vllm.guided_decoding.settings import GuidedDecodingParams



class Output(BaseModel):
    CFG: str
    terminal_functions_and_their_description: Dict[str, str]
    example_program: list[str]

json_schema = Output.model_json_schema()

from vllm import SamplingParams
from vllm import LLM as vLLM
class RecipeLoader:
    """Single responsibility: load recipe file contents as a string."""

    def __init__(self, path: str = "craft/resources/recipes.yaml") -> None:
        self._path = path

    def load(self) -> str:
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"Recipes file not found: {self._path}")
        with open(self._path, "r") as f:
            return f.read()


class CFGPromptBuilder:
    """Builds the prompt string given domain description and recipes.

    This isolates prompt formatting so changes to wording don't affect other code.
    """

    TEMPLATE = '''
You are a context-free grammar (CFG) designer. Given the domain specified below in natural language, return a CFG in BNF format that models valid actions in this domain. Use UPPERCASE for all terminal function names and terminal symbols. Do not include assignments like X ::= "x" — only include grammar rules. Terminal functions can take arguments in LPAR and RPAR.

Output MUST consist of exactly three fenced code blocks in this order:
1) CFG in a fenced block labelled bnf
2) Terminal functions dictionary as pure JSON in a fenced block labelled json
3) One example program in a fenced block (label can be dsl or text)

For example:
```bnf
expr   ::= term PLUS expr | term
term   ::= factor TIMES term | factor
factor ::= NUMBER | LPAR expr RPAR
```
```json
{{"PLUS": "Adds two terms", "TIMES": "Multiplies two terms"}}
```
```dsl
PLUS(TIMES(NUMBER, NUMBER), NUMBER)
```


## Natural Language Description of Domain
Craft is a single-agent game in a pre-specified environment. 
The environment of craft is a grid world of size n * n. Each cell can be empty, contain an item, or part of natural terrain or functional structures. When the cell is nonempty, it is considered as blocked. An agent can move around the environment freely through empty cells. At each step, the agent can either move or perform specific actions, such as collect or craft, towards the immediate cell that it is facing towards. 
At the beginning of each episode, the agent is placed at a starting cell and a distribution of items across the grid is initialized. The agent’s tasks involve either collecting primitives (raw resources) or crafting items. An item can only be crafted at the specific workshop mentioned in the recipes. 
The items to be crafted are produced from primitives (or other crafted items) by following recipes. Each recipe specifies which items are required and at which workshop the crafting must occur. A primitive item might not need to be crafted but just collected. More complex items, such as axe, or flag, require intermediate items along with primitives. This all is specified in the recipe file of the environment. Please note an item can only be crafted at the specific workshop mentioned in the recipes. 
We can move in up, down, left, and right directions.

Sometimes, primitives can be blocked by obstacles (environment items that cannot be used as primitives for crafting and are not boundary or workshop) and need the player to use a tool to pass the obstacle in order to reach and collect the primitive. The tool can be any crafted item.

### You can use the recipes file below to come up with the grammar.

{recipes}

## Task
Return:
- a CFG in BNF format that models valid actions in this domain
- a dictionary mapping TERMINAL_FUNCTION -> natural language description
- an example program written in the grammar

Strictly adhere to the fenced block output format above with no extra commentary outside code fences.

'''

    def build(self, recipes: str) -> str:
        return self.TEMPLATE.format(recipes=recipes)


class GenAIWrapper:
    """Wrapper for the `genai.Client` to allow easier testing and dependency injection.

    Keeps a very small interface: generate(prompt) -> str
    """

    def __init__(self, client: Opt[genai.Client] = None) -> None:

        self._client = client or genai.Client()

        if client == "vllm":
            self.llm = vLLM(model="/scratch/avani/gpt",    tensor_parallel_size=4 )

    def generate(self, prompt: str, model: str = "gemini-2.5-pro") -> str:
        # Keep the original call semantics but guard against errors.
        if model == "vllm":
            structured_outputs_params_json = StructuredOutputsParams(json=json_schema)
            guided_decoding_params = GuidedDecodingParams(json=json_schema)

            self.params = SamplingParams(temperature=0.7, max_tokens=15000, structured_outputs=structured_outputs_params_json, guided_decoding=guided_decoding_params)
            output = self.llm.generate([prompt], sampling_params=self.params)
            response = output[0].outputs[0].text
            print(response)
            return response
        else:
            response = self._client.models.generate_content(model=model, contents=prompt)
        # Some versions may use .text or .response; favor .text if present.
        return getattr(response, "text", getattr(response, "response", str(response)))



class CFGGenerator:
    """Orchestrates loading recipes, building prompt, calling the LLM, and returning text."""

    def __init__(self, loader: RecipeLoader, builder: CFGPromptBuilder, llm: GenAIWrapper) -> None:
        self.loader = loader
        self.builder = builder
        self.llm = llm

    def generate_cfg(self) -> str:
        recipes = self.loader.load()
        prompt = self.builder.build(recipes)
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


def main(recipes_path: str = "craft/resources/recipes.yaml") -> int:
    try:
        loader = RecipeLoader(recipes_path)
        builder = CFGPromptBuilder()
        llm_wrapper = GenAIWrapper()
        generator = CFGGenerator(loader, builder, llm_wrapper)

        output = generator.generate_cfg()
        cfg, terminals, example = parse_generated_output(output)

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
    # Allow passing a custom recipes path as the first CLI argument
    path = sys.argv[1] if len(sys.argv) > 1 else "craft/resources/recipes.yaml"
    raise_code = main(path)
    sys.exit(raise_code)