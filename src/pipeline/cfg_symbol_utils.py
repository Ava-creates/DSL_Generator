from __future__ import annotations

import re
from typing import Optional


DEFAULT_EXCLUDED_SYMBOLS = {
    "LPAR",
    "RPAR",
    "COMMA",
    "SEMI",
    "SEMICOLON",
    "LBRACKET",
    "RBRACKET",
}


def _clean_token(token: str) -> str:
    value = " ".join((token or "").split()).strip()
    value = value.strip('"').strip("'")
    return value


def _split_rhs(rhs: str) -> list[str]:
    values: list[str] = []
    for part in (rhs or "").split("|"):
        value = _clean_token(part)
        if value and "::=" not in value:
            values.append(value)
    return values


def build_cfg_rule_map(cfg_text: str) -> dict[str, list[str]]:
    """Parse CFG text into an ordered mapping of symbol -> RHS alternatives."""
    rule_map: dict[str, list[str]] = {}
    current_symbol: str | None = None
    rule_pattern = re.compile(r"^([A-Z_][A-Z0-9_]*)\s*::=\s*(.*)$")

    for raw_line in (cfg_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            current_symbol = None
            continue

        match = rule_pattern.match(line)
        if match:
            current_symbol = match.group(1).strip()
            rhs_values = _split_rhs(match.group(2).strip())
            existing = rule_map.setdefault(current_symbol, [])
            existing.extend(rhs_values)
            continue

        if current_symbol and line.startswith("|"):
            rhs_values = _split_rhs(re.sub(r"^\|\s*", "", line))
            rule_map[current_symbol].extend(rhs_values)
            continue

        current_symbol = None

    return rule_map


def resolve_symbol_to_terminal(
    symbol: str,
    rule_map: dict[str, list[str]],
    visited: Optional[set[str]] = None,
    excluded_symbols: Optional[set[str]] = None,
) -> Optional[str]:
    """Resolve a symbol to the first reachable terminal value."""
    if visited is None:
        visited = set()
    if excluded_symbols is None:
        excluded_symbols = DEFAULT_EXCLUDED_SYMBOLS

    clean_symbol = _clean_token(symbol).upper()
    if clean_symbol in visited:
        return None
    visited.add(clean_symbol)

    if clean_symbol not in rule_map:
        return None

    for value in rule_map[clean_symbol]:
        candidate = _clean_token(value)
        if not candidate:
            continue
        candidate_upper = candidate.upper()
        if candidate_upper in excluded_symbols:
            continue

        if candidate_upper in rule_map:
            resolved = resolve_symbol_to_terminal(
                candidate_upper, rule_map, visited, excluded_symbols
            )
            if resolved is not None:
                return resolved
            continue

        return candidate

    return None


def expand_symbol_to_terminals(
    symbol: str,
    rule_map: dict[str, list[str]],
    visiting: Optional[set[str]] = None,
    excluded_symbols: Optional[set[str]] = None,
) -> set[str]:
    """Expand a symbol into all reachable terminal leaf values."""
    if visiting is None:
        visiting = set()
    if excluded_symbols is None:
        excluded_symbols = DEFAULT_EXCLUDED_SYMBOLS

    clean_symbol = _clean_token(symbol).upper()
    if clean_symbol in visiting:
        return set()

    if clean_symbol not in rule_map:
        if clean_symbol in excluded_symbols:
            return set()
        return {clean_symbol}

    next_visiting = set(visiting)
    next_visiting.add(clean_symbol)

    expanded: set[str] = set()
    for value in rule_map[clean_symbol]:
        candidate = _clean_token(value)
        if not candidate:
            continue
        candidate_upper = candidate.upper()
        if candidate_upper in excluded_symbols:
            continue
        if candidate_upper in rule_map:
            expanded.update(
                expand_symbol_to_terminals(
                    candidate_upper, rule_map, next_visiting, excluded_symbols
                )
            )
        else:
            expanded.add(candidate_upper)
    return expanded
