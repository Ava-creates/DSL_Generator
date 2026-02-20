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
from vllm.sampling_params import GuidedDecodingParams



class Output(BaseModel):
    CFG: str
    terminal_functions_and_their_description: Dict[str, str]
    example_program: list[str]

json_schema = Output.model_json_schema()
# print(json_schema)

from vllm import SamplingParams
from vllm import LLM as vLLM



def CFGPromptBuilder():
    """Builds the prompt string given domain description .

    This isolates prompt formatting so changes to wording don't affect other code.
    """
    with open("prompt_specifications/nld.txt", 'r') as f:
        domain_description = f.read()

    with open("craft/resources/recipes.yaml", 'r') as f:
        recipes = f.read()

    TEMPLATE = '''
You are a context-free grammar (CFG) designer. Given the domain specified below in natural language, return a CFG in BNF format that can be used to solve this domain.

Only include terminal functions that can be implemented as a pure action-sequence generator: a function that returns a list of primitive actions (0-4). Do NOT include terminals that require external memory, cross-function state sharing, or knowledge accumulation . If a function cannot be expressed as a list of primitive actions without side effects, omit it from the CFG.

## CRITICAL RULES - Follow Exactly:

1. **ALL SYMBOLS MUST BE UPPERCASE**: Use UPPERCASE for ALL terminal functions, terminal symbols, and non-terminals. NEVER use lowercase or mixed case.

2. **Terminal Functions**:
   - Terminal functions are actions that appear directly in productions (e.g., ACTION1, ACTION2, ACTION3)
   - Use UPPERCASE names for all terminal functions
   - If a function takes no arguments, use it directly: `statement ::= ACTION1`
   - If a function takes arguments, use: `statement ::= ACTION1 LPAR PARAM RPAR`
   - NEVER create rules like `ACTION1 ::= 'action1'` or `ACTION1 ::= 'ACTION1'` - terminal functions appear directly in productions, not as separate rules

3. **Function Arguments Format**:
   - Use space-separated format: `FUNC LPAR ARG RPAR`
   - Example: `ACTION1 LPAR PARAM RPAR` (correct)
   - Example: `ACTION2 LPAR PARAM1 COMMA PARAM2 RPAR` (correct for multiple args)
   - NEVER use literal parentheses like `ACTION1(PARAM)` - always use `ACTION1 LPAR PARAM RPAR`

4. **Special Symbols** (single characters only):
   - Define punctuation as: `SYMBOL ::= 'char'` (single character in single quotes)
   - Examples: `SEMICOLON ::= ';'`, `LPAR ::= '('`, `RPAR ::= ')'`, `COMMA ::= ','`
   - These are the ONLY terminals that should have quoted character definitions

5. **Enumeration Rules** (for parameter values):
   - Use: `PARAM ::= VALUE1 | VALUE2 | VALUE3`
   - Example: `PARAM ::= OPTION1 | OPTION2 | OPTION3 | OPTION4`
   - DO NOT create individual rules like `VALUE1 ::= 'VALUE1'` - the enumeration is sufficient
   - All values in enumerations must be UPPERCASE

6. **Start Symbol**: Use lowercase `program` as the top-level non-terminal

7. **Rule Format**: One rule per line, use `|` for alternatives:
```
program        ::= statement_seq

statement_seq  ::= statement
                |  statement SEMICOLON statement_seq

statement      ::= ACTION1 LPAR PARAM RPAR
                |  ACTION2 LPAR PARAM RPAR
                |  ACTION3
                |  ACTION4 LPAR PARAM1 COMMA PARAM2 RPAR

PARAM          ::= VALUE1 | VALUE2 | VALUE3 | VALUE4
PARAM1         ::= OPTION1 | OPTION2
PARAM2         ::= CHOICE1 | CHOICE2

SEMICOLON      ::= ';'
LPAR           ::= '('
RPAR           ::= ')'
COMMA          ::= ','
```

8. **What NOT to do**:
   - ❌ NEVER create `ACTION1 ::= 'action1'` or any lowercase definitions
   - ❌ NEVER create `ACTION1 ::= 'ACTION1'` - terminal functions appear directly in productions
   - ❌ NEVER use lowercase terminal function names
   - ❌ NEVER use literal parentheses in productions (always use LPAR/RPAR)
   - ❌ NEVER create circular rules like `X ::= X`
   - ❌ NEVER create redundant rules for enumeration values

## Output Format:

Return three fenced code blocks:

1. **BNF block** (labeled `bnf` or `grammar`):
```bnf
[Your CFG here]
```

2. **JSON block** (labeled `json`):
```json
{{"ACTION1": "Description of what ACTION1 does", "ACTION2": "Description of what ACTION2 does", ...}}
```
**CRITICAL**: You MUST include ALL terminal functions that appear in your CFG productions. Every function name that appears at the start of a statement production (e.g., ACTION1, ACTION2, ACTION3 in `statement ::= ACTION1 LPAR PARAM RPAR | ACTION2 LPAR PARAM RPAR | ACTION3`) MUST have an entry in this JSON dictionary. Do not omit any terminal functions - completeness is essential.

Each description MUST be detailed and include ALL of the following:
- **What the function does** (high-level purpose)
- **Behavioral details**: how the function should work step by step — e.g., how it navigates, what state it reads, what conditions it checks

3. **DSL example block** (labeled `dsl` or `example`):
```dsl
ACTION1(VALUE1); ACTION2(VALUE2); ACTION3
```
Use actual terminal function names and values from your CFG.

## Natural Language Description of Domain:
{domain_description}

## Recipes for the domain:
{recipes}

## Domain semantics guidance (soft, not rigid):
- Use the recipe file to infer meaningful value categories.
- It is fine to keep the grammar compact, but preserve semantic clarity in symbol naming and function argument choices.

Return the CFG, terminal functions dictionary, and example program as specified above. Use ONLY UPPERCASE for all terminal functions and symbols. Choose meaningful UPPERCASE names for terminal functions based on the domain.
'''

    return TEMPLATE.format(domain_description=domain_description, recipes=recipes)


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
            
            structured_outputs_params_json = StructuredOutputsParams(json=json_schema)
            guided_decoding_params = GuidedDecodingParams(json=json_schema)

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

    def __init__(self,  llm: GenAIWrapper) -> None:
        self.llm = llm

    def generate_cfg(self) -> str:
        prompt = CFGPromptBuilder()
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


def generate_and_parse_cfg(vllm_instance: Opt[vLLM] = None) -> Tuple[str, Dict[str, str], Opt[str]]:
    """
    Generate CFG and return parsed results as a tuple.
    
    Args:
        vllm_instance: Optional vLLM instance to reuse (if None, creates a new one)
    
    Returns: (cfg_string, terminals_dict, example_program_or_None)
    """
    builder = CFGPromptBuilder()
    llm_wrapper = GenAIWrapper("vllm", vllm_instance=vllm_instance)
    generator = CFGGenerator(llm_wrapper)
    
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