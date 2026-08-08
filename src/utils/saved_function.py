"""Normalize generated function code before writing final_functions/*.py."""

from __future__ import annotations

import ast
import os
import re
from typing import Optional


def _is_import_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    if not (stripped.startswith("import ") or stripped.startswith("from ")):
        return False
    try:
        ast.parse(stripped)
    except SyntaxError:
        return False
    return True


def repair_saved_function_source(source: str) -> str:
    """Drop hoisted prose lines that were mistaken for imports in older saves."""
    lines = source.splitlines()
    kept_imports: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            if kept_imports:
                break
            index += 1
            continue
        if stripped.startswith("import ") or stripped.startswith("from "):
            if _is_import_line(line):
                kept_imports.append(line.rstrip())
            index += 1
            continue
        break

    body_lines = lines[index:]
    while body_lines and not body_lines[0].strip():
        body_lines = body_lines[1:]

    body = "\n".join(body_lines).strip("\n")
    if kept_imports:
        return "\n".join(kept_imports) + "\n\n" + body + "\n"
    return body + "\n"


def _target_func_name(func_signature: str) -> Optional[str]:
    match = re.search(r"^\s*def\s+(\w+)\s*\(", func_signature or "", re.MULTILINE)
    return match.group(1) if match else None


def _has_top_level_def(code: str, func_name: str) -> bool:
    """True only when *func_name* is defined at module scope (column 0)."""
    pattern = rf"^def\s+{re.escape(func_name)}\s*\("
    return re.search(pattern, code, re.MULTILINE) is not None


def _dedent_block(text: str) -> str:
    lines = text.splitlines()
    indents = [len(ln) - len(ln.lstrip()) for ln in lines if ln.strip()]
    if not indents:
        return text
    min_indent = min(indents)
    dedented = []
    for ln in lines:
        if ln.strip():
            dedented.append(ln[min_indent:])
        else:
            dedented.append("")
    return "\n".join(dedented)


def _normalize_mixed_body(code: str) -> str:
    """Merge column-0 statements with an indented FunSearch body block."""
    lines = code.splitlines()
    top_level: list[str] = []
    indented: list[str] = []

    for ln in lines:
        if not ln.strip():
            if indented:
                indented.append(ln)
            elif top_level:
                top_level.append(ln)
            else:
                top_level.append(ln)
        elif ln[0] in (" ", "\t"):
            indented.append(ln)
        else:
            top_level.append(ln)

    if not indented:
        return _dedent_block(code)

    min_indent = min(len(ln) - len(ln.lstrip()) for ln in indented if ln.strip())
    dedented_indented = []
    for ln in indented:
        if ln.strip():
            dedented_indented.append(ln[min_indent:])
        else:
            dedented_indented.append("")

    while top_level and not top_level[-1].strip():
        top_level.pop()

    parts: list[str] = []
    if top_level:
        parts.extend(top_level)
    if top_level and dedented_indented:
        parts.append("")
    parts.extend(dedented_indented)
    return "\n".join(parts)


def _indent_block(text: str, spaces: int = 4) -> str:
    prefix = " " * spaces
    return "\n".join(f"{prefix}{ln}" if ln.strip() else "" for ln in text.splitlines())


def ensure_function_def(code: str, func_signature: str) -> str:
    """Ensure *code* contains a module-level ``def`` using *func_signature*."""
    if not code or not code.strip():
        return code

    func_name = _target_func_name(func_signature)
    if func_name and _has_top_level_def(code, func_name):
        return code

    sig_clean = re.sub(r"\s*->\s*[^:]+", "", (func_signature or "").strip())
    if not sig_clean and func_name:
        sig_clean = f"def {func_name}(env)"
    if not sig_clean:
        return code
    if not sig_clean.endswith(":"):
        sig_clean += ":"

    body = _normalize_mixed_body(code.strip("\n"))
    return f"{sig_clean}\n{_indent_block(body)}\n"


def _extract_evolve_function(code: str, func_name: Optional[str] = None) -> Optional[str]:
    """If *code* is a FunSearch program, return only the evolved function source."""
    if "@funsearch" not in code and "funsearch." not in code:
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    evolve_fn: Optional[ast.FunctionDef] = None
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        decorators = []
        for d in node.decorator_list:
            if isinstance(d, ast.Attribute) and isinstance(d.value, ast.Name) and d.value.id == "funsearch":
                decorators.append(d.attr)
            elif isinstance(d, ast.Name):
                decorators.append(d.id)
        if "evolve" in decorators or (func_name and node.name == func_name and not decorators):
            evolve_fn = node
            break
        if func_name and node.name == func_name:
            evolve_fn = node

    if evolve_fn is None:
        return None

    lines = code.splitlines()
    # Drop decorator lines immediately above the def.
    start = evolve_fn.lineno - 1
    while start > 0 and lines[start - 1].lstrip().startswith("@"):
        start -= 1
    end = evolve_fn.end_lineno
    return "\n".join(lines[start:end]).strip("\n")


# Imports that LLM-generated programs sometimes emit but that do not exist on
# craft.env (or are not needed for final-eval loading). Drop them so import
# succeeds; callers that referenced the names usually have a fallback path.
_BAD_CRAFT_ENV_IMPORTS = re.compile(
    r"^from\s+craft\.env\s+import\s+(Action|env_factory)\b"
)


def _meaningful_body_stmts(func_def: ast.FunctionDef) -> list:
    """Return function body statements excluding docstrings."""
    meaningful = []
    for stmt in func_def.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            continue
        meaningful.append(stmt)
    return meaningful


def function_impl_is_trivial(code: str, func_name: Optional[str] = None) -> bool:
    """True when the target function has no real implementation (empty / pass / return [])."""
    if not code or not code.strip():
        return True
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True

    candidates = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if func_name is None or node.name == func_name:
            candidates.append(node)

    if not candidates:
        return True

    for func_def in candidates:
        meaningful = _meaningful_body_stmts(func_def)
        if not meaningful:
            return True
        if len(meaningful) == 1 and isinstance(meaningful[0], ast.Pass):
            return True
        if len(meaningful) == 1 and isinstance(meaningful[0], ast.Return):
            value = meaningful[0].value
            if value is None:
                return True
            if isinstance(value, ast.List) and not value.elts:
                return True
            if isinstance(value, ast.Constant) and value.value in (None, []):
                return True
        return False
    return True


def _extract_plain_function(code: str, func_name: Optional[str] = None) -> Optional[str]:
    """Return a module-level function definition that is not a FunSearch harness."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    skip_names = {"solve", "evaluate"}
    lines = code.splitlines()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name in skip_names:
            continue
        decorators = []
        for d in node.decorator_list:
            if isinstance(d, ast.Attribute) and isinstance(d.value, ast.Name) and d.value.id == "funsearch":
                decorators.append(d.attr)
            elif isinstance(d, ast.Name):
                decorators.append(d.id)
        if "run" in decorators or "evolve" in decorators:
            continue
        if func_name and node.name != func_name:
            continue
        start = node.lineno - 1
        while start > 0 and lines[start - 1].lstrip().startswith("@"):
            start -= 1
        end = node.end_lineno
        return "\n".join(lines[start:end]).strip("\n")
    return None


def prepare_function_module_source(
    code: str,
    func_signature: str = "",
    *,
    allow_trivial: bool = False,
) -> Optional[str]:
    """Normalize FunSearch artifacts into import-safe module source for evaluation."""
    if not code or not code.strip():
        return None

    func_name = _target_func_name(func_signature)
    normalized = normalize_saved_function(code, func_signature)
    if func_name and function_impl_is_trivial(normalized, func_name) and not allow_trivial:
        return None
    return normalized


def normalize_saved_function(code: str, func_signature: str = "") -> str:
    """Return import-safe module text with a valid top-level function definition."""
    if not code or not code.strip():
        return code

    func_name = _target_func_name(func_signature)
    extracted = _extract_evolve_function(code, func_name)
    if extracted is not None and not function_impl_is_trivial(extracted, func_name):
        code = extracted
    elif extracted is not None and func_name:
        plain = _extract_plain_function(code, func_name)
        if plain and not function_impl_is_trivial(plain, func_name):
            code = plain

    # Strip leftover FunSearch decorators from a lone function.
    cleaned_lines: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("@funsearch"):
            continue
        cleaned_lines.append(line)
    code = "\n".join(cleaned_lines)

    import_lines: list[str] = []
    body_lines: list[str] = []
    seen_imports: set[str] = set()

    for line in code.splitlines():
        if _is_import_line(line):
            normalized = " ".join(line.strip().split())
            if _BAD_CRAFT_ENV_IMPORTS.match(normalized):
                continue
            if normalized not in seen_imports:
                import_lines.append(normalized)
                seen_imports.add(normalized)
        else:
            body_lines.append(line)

    body = "\n".join(body_lines).strip("\n")
    body = ensure_function_def(body, func_signature)

    if import_lines:
        return "\n".join(import_lines) + "\n\n" + body.strip() + "\n"
    return body.strip() + "\n"


def resolve_func_signature(
    func_name: str,
    func_signatures: Optional[dict] = None,
) -> str:
    """Look up a function signature or synthesize ``def <safe_name>(env):``."""
    if func_signatures:
        for key in (func_name, func_name.strip()):
            sig = (func_signatures.get(key) or "").strip()
            if sig:
                return sig

    name = func_name.strip().lower()
    name = re.sub(r"\W|^(?=\d)", "_", name)
    return f"def {name}(env):"


def best_function_from_funsearch_log(log_path: str, func_signature: str) -> Optional[str]:
    """Return the top-scoring FunSearch program as a normalized module."""
    if not log_path or not os.path.isfile(log_path):
        return None

    from src.pipeline.explicit_feedback_generation import parse_log_file

    top = parse_log_file(log_path, k=1)
    if not top:
        return None

    _score, body = top[0]
    if not body or not body.strip():
        return None

    prepared = prepare_function_module_source(body, func_signature, allow_trivial=False)
    return prepared
