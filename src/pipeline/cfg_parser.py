from lark import Lark
from typing import Dict, List, Set, Tuple, Optional, Union
import re
import os

class CFGParser:
    def __init__(self, cfg_input: Union[str, os.PathLike], start_symbol: Optional[str] = None):
        """Initialize CFG parser.
        
        Args:
            cfg_input: Either a file path (string or PathLike) or CFG string in BNF format
            start_symbol: Optional start symbol name. If None, auto-detects from CFG
                         (prefers 'program', otherwise uses first non-terminal)
        """
        self.cfg_input = cfg_input
        # print("in the cfg parser", self.cfg_input)
        self.start_symbol = start_symbol
        self.terminals: Set[str] = set()
        self.non_terminals: Set[str] = set()
        self.rules: Dict[str, List[str]] = {}
        self.terminal_functions: List[Tuple[str, List[str]]] = []
        self.special_symbols: Dict[str, str] = {}
        self.keywords: Dict[str, str] = {}
        
        self._read_cfg()
        self._detect_special_symbols()
        self._detect_keywords()
        self._detect_start_symbol()
        self._build_grammar()
        self._extract_terminal_functions()

    def _read_cfg(self):
        """Read the BNF file or string and store rules."""
        # Check if input is a file path or string
        if isinstance(self.cfg_input, (str, os.PathLike)) and os.path.exists(self.cfg_input):
            # It's a file path
            with open(self.cfg_input, 'r') as f:
                cfg_text = f.read()
        else:
            # It's a CFG string
            cfg_text = str(self.cfg_input)
        
        # Parse the CFG text - handle multi-line rules
        current_lhs = None
        for line in cfg_text.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Check if this is a continuation line (starts with |)
            if line.startswith('|'):
                # This is a continuation of the previous rule
                if current_lhs is None:
                    continue  # Skip if no current rule
                # Extract the alternative (remove leading |)
                alt = line[1:].strip()
                if alt or alt == '':
                    # Split on | to handle multiple alternatives on the same line
                    # e.g., "|  PLANK | ROPE | BUNDLE" should become 3 separate alternatives
                    alternatives = [a.strip() for a in alt.split('|')]
                    for alternative in alternatives:
                        # Check for epsilon productions (/* ε */ or just ε)
                        # Remove C-style comments and check if what remains is just epsilon or empty
                        alt_clean = re.sub(r'/\*.*?\*/', '', alternative).strip()
                        if not alt_clean or alt_clean == 'ε' or 'ε' in alternative:
                            # Replace epsilon with empty string
                            alternative = ''
                        if alternative or alternative == '':  # Allow empty alternatives (epsilon)
                            self.rules[current_lhs].append(alternative)
                            # Extract terminals from each alternative (skip if empty)
                            if alternative:
                                self._extract_terminals_from_production(alternative)
                continue

            # Split on ::=
            parts = line.split('::=')
            if len(parts) != 2:
                # Reset current_lhs if we hit a line without ::=
                current_lhs = None
                continue

            # Get left and right hand sides
            lhs = parts[0].strip()
            rhs = parts[1].strip().rstrip(';')  # Remove trailing semicolon

            # Add non-terminal
            self.non_terminals.add(lhs)
            current_lhs = lhs  # Track current rule for continuation lines

            # Handle rules
            if lhs not in self.rules:
                self.rules[lhs] = []

            # Split on alternation operator
            alternatives = [alt.strip() for alt in rhs.split('|')]
            for alt in alternatives:
                # Check for epsilon productions (/* ε */ or just ε)
                # Remove C-style comments and check if what remains is just epsilon or empty
                alt_clean = re.sub(r'/\*.*?\*/', '', alt).strip()
                if not alt_clean or alt_clean == 'ε' or 'ε' in alt:
                    # Replace epsilon with empty string
                    alt = ''
                if alt or alt == '':  # Allow empty alternatives (epsilon)
                    self.rules[lhs].append(alt)
                    # Extract terminals from this alternative (skip if empty)
                    if alt:
                        self._extract_terminals_from_production(alt)

    def _extract_terminals_from_production(self, production: str):
        """Extract terminals and terminal functions from a production."""
        # First, find all terminal functions (words in uppercase)
        terminal_funcs = re.findall(r'\b[A-Z_][A-Z0-9_]*\b', production)
        for func in terminal_funcs:
            if func not in self.non_terminals:
                self.terminals.add(func)

        # Then find all quoted strings and bare words
        # But skip single characters that will be handled as special symbols
        tokens = re.findall(r'\b[a-z][a-z0-9_]*\b|"(.*?)"|\'(.*?)\'', production)
        for t in tokens:
            if isinstance(t, tuple):
                for tok in t:
                    # Skip single characters - they're handled by special_symbols
                    if tok and len(tok) > 1 and tok not in self.non_terminals:
                        self.terminals.add(tok)
            elif t and t not in self.non_terminals:
                # Skip single characters
                if len(t) > 1:
                    self.terminals.add(t)

    def _detect_special_symbols(self):
        """Auto-detect special symbols from CFG rules (like SEMICOLON, LPAR, RPAR)."""
        # Look for rules that define special symbols with quoted strings
        # Pattern: SYMBOL ::= 'char' or SYMBOL ::= "char"
        for non_terminal, productions in self.rules.items():
            for production in productions:
                # Check if production is just a quoted character
                quoted_match = re.match(r'^[\'"](\S)[\'"]$', production.strip())
                if quoted_match:
                    char = quoted_match.group(1)
                    self.special_symbols[non_terminal] = char
                    self.terminals.add(non_terminal)

    def _detect_keywords(self):
        """Auto-detect keywords from CFG rules (lowercase words in quotes)."""
        # Look for rules that define keywords
        # Pattern: KEYWORD ::= "word" or KEYWORD ::= 'word'
        for non_terminal, productions in self.rules.items():
            for production in productions:
                # Check if production is a quoted word (lowercase)
                quoted_match = re.match(r'^[\'"]([a-z][a-z0-9_]*)[\'"]$', production.strip())
                if quoted_match:
                    keyword_value = quoted_match.group(1)
                    self.keywords[non_terminal] = keyword_value
                    self.terminals.add(non_terminal)

    def _detect_start_symbol(self):
        """Auto-detect the start symbol from the CFG."""
        if self.start_symbol:
            # Use provided start symbol
            if self.start_symbol not in self.non_terminals:
                raise ValueError(f"Start symbol '{self.start_symbol}' not found in non-terminals")
            return
        
        # Auto-detect: prefer 'program', otherwise use first non-terminal
        if 'program' in self.non_terminals:
            self.start_symbol = 'program'
        elif self.non_terminals:
            # Use the first non-terminal (typically the first rule)
            self.start_symbol = next(iter(self.non_terminals))
        else:
            raise ValueError("No non-terminals found in CFG")

    def _extract_terminal_functions(self):
        """Extract terminal functions from productions."""
        for rule, productions in self.rules.items():
            for production in productions:
                # Pattern 1: FUNC_NAME(ARG1, ARG2, ...) - literal parentheses
                matches = re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*([^)]*)\s*\)', production)
                for match in matches:
                    func_name = match.group(1)
                    args_str = match.group(2).strip()
                    args = [arg.strip() for arg in args_str.split(',') if arg.strip()] if args_str else []
                    if (func_name, args) not in self.terminal_functions:
                        self.terminal_functions.append((func_name, args))
                        self.terminals.add(func_name)
                
                # Pattern 2: FUNC_NAME LPAR ARG RPAR - terminal symbols
                # Match FUNC_NAME followed by LPAR, then capture everything until RPAR
                matches2 = re.finditer(r'\b([A-Z_][A-Z0-9_]*)\s+LPAR\s+(.*?)\s+RPAR\b', production)
                for match in matches2:
                    func_name = match.group(1)
                    args_str = match.group(2).strip()
                    # Split by COMMA if present, otherwise treat as single arg
                    if 'COMMA' in args_str:
                        args = [arg.strip() for arg in args_str.split('COMMA') if arg.strip()]
                    else:
                        args = [args_str] if args_str.strip() else []
                    # Filter out terminal symbols like LPAR, RPAR, COMMA
                    args = [arg for arg in args if arg and arg not in ['LPAR', 'RPAR', 'COMMA', 'SEMICOLON', 'SEMI']]
                    if (func_name, args) not in self.terminal_functions:
                        self.terminal_functions.append((func_name, args))
                        self.terminals.add(func_name)

    def get_terminal_functions(self) -> List[Tuple[str, List[str]]]:
        return self.terminal_functions

    def get_functions_with_args(self) -> List[Tuple[str, List[str]]]:
        """Extract functions that take arguments (have parentheses) from the grammar."""
        # Return terminal functions that have arguments
        return [(name, args) for name, args in self.terminal_functions if args]

    def _is_enumeration_rule(self, non_terminal: str, productions: List[str]) -> bool:
        """Check if a rule is just an enumeration of terminal values."""
        # A rule is an enumeration if all productions are single uppercase words
        # and none of them are non-terminals
        if len(productions) < 2:
            return False
        
        for production in productions:
            production = production.strip()
            # Check if production is just a single uppercase word/identifier
            if not re.match(r'^[A-Z_][A-Z0-9_]*$', production):
                return False
            # Check if it's not a non-terminal
            if production in self.non_terminals:
                return False
        
        return True

    def _build_grammar(self):
        """
        Build a complete Lark grammar from the CFG:
        - Regex terminals → emitted as token definitions
        - Literal terminals (like "(" ")" ";") → NOT emitted as token rules,
        but inlined directly inside productions as string literals.

        This preserves all original behavior while applying terminal normalization.
        """
        
        # Build a dictionary mapping terminal names to their patterns
        terminal_patterns: Dict[str, str] = {}
        
        # Extract patterns from rules (for terminals defined in rules)
        for non_terminal, productions in self.rules.items():
            if non_terminal in self.terminals:
                # This terminal is defined by a rule
                for production in productions:
                    production = production.strip()
                    # Check if it's a quoted literal (single char or word)
                    quoted_match = re.match(r'^[\'"](\S+)[\'"]$', production)
                    if quoted_match:
                        literal_value = quoted_match.group(1)
                        terminal_patterns[non_terminal] = f"'{literal_value}'"
                        break
                    # Otherwise, treat as regex pattern (use the production as-is)
                    elif production and non_terminal not in terminal_patterns:
                        terminal_patterns[non_terminal] = production
        
        # Add special symbols (they're already terminals)
        for symbol, char in self.special_symbols.items():
            terminal_patterns[symbol] = f"'{char}'"
        
        # Add keywords (they're already terminals)
        for keyword, word in self.keywords.items():
            terminal_patterns[keyword] = f"'{word}'"
        
        # For terminals that don't have explicit patterns, create default regex patterns
        # (typically uppercase identifiers that should match themselves)
        for term in self.terminals:
            if term not in terminal_patterns:
                # Default: match the uppercase identifier as a literal string
                terminal_patterns[term] = f'"{term}"'

        def inline_terminal(symbol):
            """Return the inline grammar form of a terminal symbol."""
            if symbol not in self.terminals:
                return symbol  # Nonterminal

            pattern = terminal_patterns.get(symbol, f'"{symbol}"')

            # Literal pattern: "'('", "';'", etc.
            if (
                isinstance(pattern, str)
                and len(pattern) >= 2
                and pattern[0] == "'" and pattern[-1] == "'"
            ):
                literal = pattern[1:-1]           # strip quotes
                literal = literal.replace('"', '\\"')  # escape inner quotes
                return f"\"{literal}\""

            # Regex token: keep symbolic name
            return symbol

        def tokenize_production(production: str) -> List[str]:
            """Tokenize a production string into symbols."""
            # Handle function calls like MOVE(LPAR DIR RPAR) or MOVE(DIR)
            # and regular sequences like "action SEMICOLON action_seq"
            tokens = []
            i = 0
            while i < len(production):
                # Skip whitespace
                if production[i].isspace():
                    i += 1
                    continue
                
                # Check for opening parenthesis (function call)
                if production[i] == '(':
                    tokens.append('(')
                    i += 1
                    continue
                
                # Check for closing parenthesis
                if production[i] == ')':
                    tokens.append(')')
                    i += 1
                    continue
                
                # Extract identifier (alphanumeric + underscore)
                if production[i].isalnum() or production[i] == '_':
                    start = i
                    while i < len(production) and (production[i].isalnum() or production[i] == '_'):
                        i += 1
                    tokens.append(production[start:i])
                    continue
                
                # Unknown character, skip it
                i += 1
            
            return tokens

        parts = []

        # ---------------------------------------------------------
        # 1. Emit START rule
        # ---------------------------------------------------------
        if "start" in self.non_terminals:
            # If they already have a "start" nonterminal, reuse it
            parts.append("start: start\n")
        else:
            # Otherwise automatically use the detected start symbol
            parts.append(f"start: {self.start_symbol}\n")

        # ---------------------------------------------------------
        # 2. Emit token rules for regex terminals ONLY
        # ---------------------------------------------------------
        for term in self.terminals:
            # Skip if this is actually a non-terminal (has a rule definition)
            if term in self.non_terminals:
                continue
                
            pattern = terminal_patterns.get(term, f'"{term}"')
            
            is_literal = (
                isinstance(pattern, str)
                and len(pattern) >= 2
                and pattern[0] == "'" and pattern[-1] == "'"
            )
            if is_literal:
                # Skip literal terminals; they will be inlined
                continue

            # Emit token rule normally
            parts.append(f"{term}: {pattern}")

        parts.append("")  # spacing

        # ---------------------------------------------------------
        # 3. Emit all production rules with literals inlined
        # ---------------------------------------------------------
        for lhs, rhs_list in self.rules.items():
            # Skip terminals that are special symbols or keywords - they're handled as inline literals
            if lhs in self.special_symbols or lhs in self.keywords:
                continue
            
            rule_variants = []
            for rhs in rhs_list:
                # Handle epsilon (empty) productions
                if not rhs or rhs.strip() == '':
                    # In Lark, empty production is represented by having nothing after the colon
                    # We'll mark it as empty
                    rule_variants.append(None)  # Use None to mark empty
                else:
                    # Tokenize the production string
                    symbols = tokenize_production(rhs)
                    if not symbols:
                        # If tokenization resulted in empty, treat as epsilon
                        rule_variants.append(None)
                    else:
                        rewritten = [inline_terminal(symbol) for symbol in symbols]
                        rule_variants.append(" ".join(rewritten))

            # If all variants are empty, ensure at least one empty production
            if not rule_variants:
                rule_variants.append(None)
            
            # Build the body, separating empty and non-empty productions
            non_empty = [v for v in rule_variants if v is not None]
            has_empty = any(v is None for v in rule_variants)
            
            if non_empty and has_empty:
                # Mix of empty and non-empty: non-empty first, then empty
                body = "\n    | ".join(non_empty) + "\n    | "  # Empty production (nothing after |)
            elif non_empty:
                # Only non-empty productions
                body = "\n    | ".join(non_empty)
            else:
                # Only empty production(s)
                body = ""  # Nothing after colon
            
            parts.append(f"{lhs}: {body}\n")

        # ---------------------------------------------------------
        # 4. Add whitespace handling
        # ---------------------------------------------------------
        parts.append("%import common.WS")
        parts.append("%ignore WS")

        # Store the grammar and create the parser
        self.grammar = "\n".join(parts)
        self.parser = Lark(self.grammar, start='start', parser='lalr')




    def parse(self, program: str):
        """Parse a program string using the CFG."""
        return self.parser.parse(program)

    def get_grammar(self) -> str:
        """Get the generated Lark grammar string."""
        return self.grammar
    
    def start(self) -> str:
        """Return the start symbol of the grammar."""
        return self.start_symbol

# Example usage
if __name__ == "__main__":
    # Test with file path
    if os.path.exists("cfg/cfg.txt"):
        cfg_parser = CFGParser("cfg/cfg.txt")
    else:
        # Test with CFG string
        cfg_string = """program        ::= action_seq
action_seq     ::= action
                |  action SEMICOLON action_seq
action         ::= MOVE(LPAR DIR RPAR)
                |  TURN(LPAR DIR RPAR)
                |  COLLECT(LPAR ITEM RPAR)
DIR            ::= UP | DOWN | LEFT | RIGHT
ITEM           ::= PRIMITIVE | CRAFTED
PRIMITIVE      ::= IRON | GRASS | WOOD
SEMICOLON      ::= ';'
LPAR           ::= '('
RPAR           ::= ')'"""
        cfg_parser = CFGParser(cfg_string)

    print("\nFunctions with arguments:")
    for name, args in cfg_parser.get_functions_with_args():
        print(f"{name}({', '.join(args)})")

    print("\nGenerated Grammar:")
    print(cfg_parser.get_grammar())
    print("\n" + "="*50)
    print("Rules:", cfg_parser.rules)
    print("Terminals:", cfg_parser.terminals)
    print("Non-terminals:", cfg_parser.non_terminals)
    print("Start symbol:", cfg_parser.start())
    print("Special symbols:", cfg_parser.special_symbols)
    print("Keywords:", cfg_parser.keywords)

    test_program = "MOVE(RIGHT); COLLECT(WOOD); TURN(UP)"

    try:
        tree = cfg_parser.parse(test_program)
        print("\nParse successful!")
        print(tree.pretty())
    except Exception as e:
        print(f"\nError parsing program: {e}")
