from lark import Lark
from typing import Dict, List, Set, Tuple
import re

class CFGParser:
    def __init__(self, cfg_file_path: str):
        self.cfg_file_path = cfg_file_path
        self.terminals: Set[str] = set()
        self.non_terminals: Set[str] = set()
        self.rules: Dict[str, List[str]] = {}
        self.terminal_functions: List[Tuple[str, List[str]]] = []
        self._read_cfg()
        self._build_grammar()
        self._extract_terminal_functions()

    def _read_cfg(self):
        """Read the BNF file and store rules."""
        with open(self.cfg_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Split on ::=
                parts = line.split('::=')
                if len(parts) != 2:
                    continue

                # Get left and right hand sides
                lhs = parts[0].strip()
                rhs = parts[1].strip().rstrip(';')  # Remove trailing semicolon

                # Add non-terminal
                self.non_terminals.add(lhs)

                # Handle rules
                if lhs not in self.rules:
                    self.rules[lhs] = []

                # Split on alternation operator
                alternatives = [alt.strip() for alt in rhs.split('|')]
                for alt in alternatives:
                    self.rules[lhs].append(alt)

                # Extract terminals and terminal functions
                # First, find all terminal functions (words in uppercase)
                terminal_funcs = re.findall(r'\b[A-Z_][A-Z0-9_]*\b', rhs)
                for func in terminal_funcs:
                    if func not in self.non_terminals:
                        self.terminals.add(func)

                # Then find all quoted strings and bare words
                tokens = re.findall(r'\b[a-z][a-z0-9_]*\b|"(.*?)"|\'(.*?)\'', rhs)
                for t in tokens:
                    if isinstance(t, tuple):
                        for tok in t:
                            if tok and tok not in self.non_terminals:
                                self.terminals.add(tok)
                    elif t and t not in self.non_terminals:
                        self.terminals.add(t)

    def _extract_terminal_functions(self):
        for rule, productions in self.rules.items():
            for production in productions:
                matches = re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*([^)]*)\s*\)', production)
                for match in matches:
                    func_name = match.group(1)
                    args_str = match.group(2).strip()
                    args = [arg.strip() for arg in args_str.split(',')] if args_str else []
                    if (func_name, args) not in self.terminal_functions:
                        self.terminal_functions.append((func_name, args))
                        self.terminals.add(func_name)

    def get_terminal_functions(self) -> List[Tuple[str, List[str]]]:
        return self.terminal_functions

    def get_functions_with_args(self) -> List[Tuple[str, List[str]]]:
        """Extract functions that take arguments (have parentheses) from the grammar."""
        functions = []
        # print("\nDebug - Rules:", self.rules)  # Debug print
        for rule, productions in self.rules.items():
            # print(f"\nDebug - Processing rule: {rule}")  # Debug print
            for production in productions:
                # print(f"Debug - Production: {production}")  # Debug print
                # Find patterns like FUNCTION LPAR ARG RPAR
                # Look for any word followed by LPAR
                matches = re.finditer(r'\b([A-Z_][A-Z0-9_]*)\s+LPAR\s+([^R]*)\s*RPAR', production)
                for match in matches:
                    func_name = match.group(1)
                    args_str = match.group(2).strip()
                    print(f"Debug - Found function: {func_name} with args: {args_str}")  # Debug print
                    # Split arguments and clean them
                    args = [arg.strip() for arg in args_str.split(',')] if args_str else []
                    # Only add if not already in the list
                    if (func_name, args) not in functions:
                        functions.append((func_name, args))
                        # Also add the function name to terminals if it's not already there
                        if func_name not in self.terminals:
                            self.terminals.add(func_name)
        return functions

    def _build_grammar(self) -> str:
        """Build Lark grammar from the BNF rules."""
        grammar_lines = []

        # First, collect all terminals that need to be defined
        all_terminals = set()
        
        # Add special symbol terminals
        special_symbols = {
            "SEMI": ";",
            "LPAR": "(",
            "RPAR": ")",
        }
        all_terminals.update(special_symbols.keys())
        
        # Add keywords
        keywords = {
            "IF": "if",
            "HAS": "has",
            "THEN": "then"
        }
        all_terminals.update(keywords.keys())
        
        # Collect terminals from rules
        for non_terminal, productions in self.rules.items():
            for production in productions:
                # Find all bare words that should be terminals
                if non_terminal.lower() in ['item', 'dir', 'kind']:
                    matches = re.finditer(r'\b([A-Z][A-Z0-9_]*)\b', production)
                    for match in matches:
                        all_terminals.add(match.group(1))
                # Add terminal functions
                terminal_funcs = re.findall(r'\b[A-Z_][A-Z0-9_]*\b', production)
                all_terminals.update(terminal_funcs)

        # Add terminal definitions (Lark expects uppercase for terminals)
        for terminal in sorted(all_terminals):
            if terminal in special_symbols:
                grammar_lines.append(f"{terminal}: \"{special_symbols[terminal]}\"")
            elif terminal in keywords:
                grammar_lines.append(f"{terminal}: \"{keywords[terminal]}\"")
            elif terminal.isupper():
                # For terminal functions, use exact matching
                if terminal.endswith('_FUNC'):
                    grammar_lines.append(f"{terminal}: \"{terminal}\"")
                else:
                    # For other uppercase terminals (like items), use case-insensitive matching
                    grammar_lines.append(f"{terminal}: /(?i){terminal}/")
            else:
                grammar_lines.append(f"{terminal.upper()}: /(?i){terminal}/")

        # Add start rule
        grammar_lines.append("start: s")

        # Add all rules from the BNF file
        for non_terminal, productions in self.rules.items():
            rule_def = f"{non_terminal.lower()}: "
            alternatives = []

            for production in productions:
                # Handle special symbols
                lark_prod = production
                lark_prod = lark_prod.replace(';', ' SEMI ')
                lark_prod = re.sub(r'\s+', ' ', lark_prod).strip()

                # Replace all non-terminals (case-insensitive, word-boundary)
                for nt in sorted(self.non_terminals, key=len, reverse=True):
                    lark_prod = re.sub(r'\b' + re.escape(nt) + r'\b', nt.lower(), lark_prod, flags=re.IGNORECASE)

                # Replace keywords
                for keyword, value in keywords.items():
                    lark_prod = re.sub(r'\b' + re.escape(value) + r'\b', keyword, lark_prod, flags=re.IGNORECASE)

                # Replace terminal functions (uppercase) with their exact names
                for term in sorted(all_terminals, key=len, reverse=True):
                    if term.isupper() and term not in keywords:
                        lark_prod = re.sub(r'\b' + re.escape(term) + r'\b', term, lark_prod)
                    elif term not in self.non_terminals and term not in keywords.values():
                        # For bare word terminals, use their uppercase version
                        lark_prod = re.sub(r'\b' + re.escape(term) + r'\b', term.upper(), lark_prod, flags=re.IGNORECASE)

                alternatives.append(lark_prod)

            # Special handling for rules that are just alternations of bare words
            if non_terminal.lower() in ['item', 'dir', 'kind']:
                # Convert the alternatives to a list of terminals
                term_list = []
                for alt in alternatives:
                    # Extract the bare word from each alternative
                    match = re.match(r'\b([A-Z][A-Z0-9_]*)\b', alt)
                    if match:
                        term_list.append(match.group(1))
                if term_list:
                    # Create a rule that matches any of the terminals
                    rule_def = f"{non_terminal.lower()}: " + " | ".join(term_list)
                else:
                    rule_def += " | ".join(alternatives)
            else:
                rule_def += " | ".join(alternatives)

            grammar_lines.append(rule_def)

        # Add whitespace handling
        grammar_lines.append("%import common.WS")
        grammar_lines.append("%ignore WS")

        self.grammar = "\n".join(grammar_lines)
        try:
            self.parser = Lark(self.grammar, parser='lalr')
        except Exception as e:
            print("Error in generated grammar:")
            print(self.grammar)
            raise e

    def parse(self, program: str):
        return self.parser.parse(program)

    def get_grammar(self) -> str:
        return self.grammar
    
    def start(self):
    # Return the start symbol of the grammar, assume the first non-terminal as start
    # If you want a fixed start symbol, you can hardcode it here, e.g. "S"
        return next(iter(self.non_terminals))

# Example usage
if __name__ == "__main__":
    cfg_parser = CFGParser("cfg/cfg.txt")

    print("\nFunctions with arguments:")
    for name, args in cfg_parser.get_functions_with_args():
        print(f"{name}({', '.join(args)})")

    print("\nGenerated Grammar:")
    print(cfg_parser.get_grammar())
    print("\n" + "="*50)
    print("Rules:", cfg_parser.rules)
    print("Terminals:", cfg_parser.terminals)
    print("Non-terminals:", cfg_parser.non_terminals)

    test_program = '''
    MOVE_FUNC(UP);
    CRAFT_FUNC(PLANK);
    if has(FLAG) then CRAFT_FUNC(AXE); MOVE_FUNC(DOWN);
    '''

    try:
        tree = cfg_parser.parse(test_program)
        print("\nParse successful!")
        print(tree.pretty())
    except Exception as e:
        print(f"\nError parsing program: {e}")

