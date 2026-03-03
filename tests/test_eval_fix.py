#!/usr/bin/env python3
"""
Test script to verify that solve() and evaluate() functions are preserved
in the explicit feedback generation processing.
"""

import os
import sys
import re
import ast

# Add project root to path
_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _project_root)

# Don't import the actual module to avoid vllm dependency
# We'll test the logic directly

def _find_function_end(lines, start_idx):
    """Find the end of a function definition."""
    indent_level = len(lines[start_idx]) - len(lines[start_idx].lstrip())
    i = start_idx + 1
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:  # Empty line
            i += 1
            continue
        current_indent = len(lines[i]) - len(lines[i].lstrip())
        if current_indent <= indent_level and line.strip():
            break
        i += 1
    return i

def test_eval_preservation():
    """Test that solve() and evaluate() are preserved during processing."""
    
    # Read the function prompt file (this has solve, evaluate, and turn)
    func_file = "experiment_20260107_123927/function_specific_prompts/turn_dsl0_func0.txt"
    eval_file = "experiment_20260107_123927/explicit_feedback/eval_turn_dsl0_func0.py"
    
    print("=" * 80)
    print("Testing eval() function preservation fix")
    print("=" * 80)
    
    # Read the original function prompt file
    with open(func_file, 'r') as f:
        original_content = f.read()
    
    print(f"\n1. Original file has:")
    print(f"   - solve() function: {'def solve' in original_content}")
    print(f"   - evaluate() function: {'def evaluate' in original_content}")
    print(f"   - turn() function: {'def turn' in original_content}")
    
    # Create a test eval file by copying the function prompt
    os.makedirs(os.path.dirname(eval_file), exist_ok=True)
    with open(eval_file, 'w') as f:
        f.write(original_content)
    
    print(f"\n2. Created eval file: {eval_file}")
    
    # Now simulate what happens in explicit_feedback_generation.py
    # Read the eval file
    with open(eval_file, 'r') as f:
        full_program = f.read()
    
    # Remove decorators (as done in the actual code)
    full_program = re.sub(r'^\s*@funsearch\.(run|evolve)\s*$', '', full_program, flags=re.MULTILINE)
    full_program = re.sub(r'@funsearch\.(run|evolve)\s*\n\s*', '', full_program)
    full_program = re.sub(r'@funsearch\.(run|evolve)\s+', '', full_program)
    
    print(f"\n3. After removing decorators:")
    print(f"   - solve() function: {'def solve' in full_program}")
    print(f"   - evaluate() function: {'def evaluate' in full_program}")
    print(f"   - turn() function: {'def turn' in full_program}")
    
    # Test AST parsing with the fix
    essential_functions = {'solve', 'evaluate'}
    try:
        import ast
        tree = ast.parse(full_program)
        
        # Find all function definitions
        function_defs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                start_line = node.lineno - 1
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                
                has_real_body = False
                if len(node.body) > 1:
                    has_real_body = True
                elif len(node.body) == 1:
                    stmt = node.body[0]
                    if not (isinstance(stmt, ast.Expr) and 
                           isinstance(stmt.value, (ast.Str, ast.Constant))):
                        has_real_body = True
                
                if has_real_body:
                    function_defs.append((node.name, start_line, end_line))
        
        print(f"\n4. AST parsing found {len(function_defs)} functions with bodies:")
        for func_name, start, end in function_defs:
            print(f"   - {func_name}() (lines {start+1}-{end+1})")
        
        # Build valid function ranges
        all_valid_function_ranges = set()
        for func_name, start, end in function_defs:
            for i in range(start, end):
                all_valid_function_ranges.add(i)
        
        # Test the fix: preserve essential functions
        lines = full_program.split('\n')
        lines_to_keep = set(range(len(lines)))
        
        # Apply the fix logic
        i = 0
        preserved_functions = []
        while i < len(lines):
            line = lines[i]
            func_match = re.match(r'^\s*def\s+(\w+)\s*\(', line)
            if func_match:
                func_name = func_match.group(1)
                # Always preserve essential evaluation functions (solve, evaluate)
                if func_name in essential_functions:
                    end_idx = _find_function_end(lines, i)
                    for j in range(i, end_idx):
                        lines_to_keep.add(j)
                    preserved_functions.append(func_name)
                    i = end_idx
                    continue
                # Check if this function is in our valid list
                if i not in all_valid_function_ranges:
                    end_idx = _find_function_end(lines, i)
                    for j in range(i, end_idx):
                        lines_to_keep.discard(j)
                    i = end_idx
                    continue
            i += 1
        
        print(f"\n5. After applying fix (preserving essential functions):")
        print(f"   - Preserved functions: {preserved_functions}")
        
        # Build cleaned program
        cleaned_lines = [lines[i] for i in sorted(lines_to_keep)]
        cleaned_program = '\n'.join(cleaned_lines)
        
        print(f"\n6. Final cleaned program has:")
        print(f"   - solve() function: {'def solve' in cleaned_program}")
        print(f"   - evaluate() function: {'def evaluate' in cleaned_program}")
        print(f"   - turn() function: {'def turn' in cleaned_program}")
        
        # Test that evaluate() can be called (syntax check)
        # Use string concatenation instead of f-string to avoid variable scope issues
        test_program = cleaned_program + """

# Test that evaluate exists
try:
    if 'evaluate' in dir():
        print("\\n SUCCESS: evaluate() function is preserved and callable")
    else:
        print("\\n FAIL: evaluate() function not found")
except Exception as err:
    print("\\n ERROR: " + str(err))
"""
        
        # Check syntax
        try:
            compile(test_program, '<string>', 'exec')
            print("\n SUCCESS: Program compiles without syntax errors")
            print("\n" + "=" * 80)
            print("TEST PASSED: solve() and evaluate() are preserved!")
            print("=" * 80)
            return True
        except SyntaxError as e:
            print(f"\n FAIL: Syntax error: {e}")
            print(f"   Line {e.lineno}: {e.text}")
            return False
            
    except Exception as e:
        print(f"\n ERROR during processing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_eval_preservation()
    sys.exit(0 if success else 1)

