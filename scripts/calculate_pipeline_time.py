#!/usr/bin/env python3
"""
Calculate time estimates for each pipeline round.

This script helps determine appropriate SLURM time limits based on:
- Number of functions to implement
- Number of tasks to test
- Number of function evolution rounds
- Number of DSL evolution rounds
"""

import argparse
from typing import Dict

# Time estimates (in minutes) - adjust based on your observations
# Updated with parallelization improvements:
# - FunSearch: 500 sequential iterations per function, but with 4 samplers + 4 evaluators working in parallel
#   Each iteration: sampler gets prompt from database -> LLM generates samples -> evaluators test samples -> update database
#   With 4 samplers: 4x parallelization of the evolutionary process (4 samplers all contributing to same database)
#   With 4 evaluators: 4x parallelization of program evaluation
#   Total speedup: ~16x (4 samplers * 4 evaluators) for the evolutionary work
# - Explicit feedback: 8 functions in parallel
# - Program synthesis: Sequential per task (30 attempts per task)

TIME_ESTIMATES = {
    # CFG Generation
    "cfg_generation": 5,  # Initial CFG generation from LLM
    
    # CFG Implementation (per DSL round)
    # FunSearch: 500 sequential iterations, but parallelized with 4 samplers + 4 evaluators
    # Time per iteration: ~0.5 min (LLM call + evaluation)
    # With 4 samplers working in parallel: 500 iterations * 0.5 min / 4 = ~62.5 min
    # With 4 evaluators: further speedup on evaluation side
    # Total: ~60 min per function (500 * 0.5 / 4 samplers, with evaluator parallelization)
    "funsearch_per_function": 60,  # 500 iterations * 0.5 min / 4 samplers (with evaluator parallelization)
    "funsearch_parallel_factor": 16,  # 16 functions run in parallel (different functions)
    
    # Explicit feedback: With 8 functions in parallel
    "explicit_feedback_per_function": 5,  # Single iteration, ~5 min per function
    "explicit_feedback_parallel_factor": 8,  # 8 functions run in parallel
    
    # Program synthesis: 30 attempts per task, sequential (can parallelize tasks)
    "program_synthesis_per_task": 30,  # Testing each task: 30 attempts * 1 min per attempt
    
    # Function Evolution (per round)
    # Similar parallelization as initial implementation
    "function_evolution_funsearch": 60,  # FunSearch for evolved functions: 500 iterations with parallelization
    "function_evolution_explicit_feedback": 5,  # Explicit feedback: single iteration
    "function_evolution_testing": 30,  # Re-testing failing tasks: 30 attempts per task
    
    # DSL Evolution
    "dsl_evolution": 10,  # LLM call to evolve DSL
    
    # Overhead
    "overhead_per_round": 5,  # File I/O, checkpointing, plotting, etc.
    "plotting": 2,  # Generating plots
}

def calculate_time_for_dsl_round(
    num_functions: int,
    num_tasks: int,
    num_function_evolutions: int = 3,
    include_initial_implementation: bool = True
) -> Dict[str, float]:
    """Calculate time for one DSL round.
    
    Args:
        num_functions: Number of functions to implement
        num_tasks: Number of tasks to test
        num_function_evolutions: Number of function evolution rounds
        include_initial_implementation: Whether to include initial CFG implementation
    
    Returns:
        Dictionary with time breakdown
    """
    times = {}
    total = 0
    
    if include_initial_implementation:
        # Initial CFG implementation
        # FunSearch: parallelized across functions (16 at a time) and within each (4 samplers + 4 evaluators)
        funsearch_time_per_batch = TIME_ESTIMATES["funsearch_per_function"]
        num_funsearch_batches = (num_functions + TIME_ESTIMATES["funsearch_parallel_factor"] - 1) // TIME_ESTIMATES["funsearch_parallel_factor"]
        funsearch_time = num_funsearch_batches * funsearch_time_per_batch
        
        # Explicit feedback: parallelized across functions (8 at a time)
        explicit_feedback_time_per_batch = TIME_ESTIMATES["explicit_feedback_per_function"]
        num_ef_batches = (num_functions + TIME_ESTIMATES["explicit_feedback_parallel_factor"] - 1) // TIME_ESTIMATES["explicit_feedback_parallel_factor"]
        explicit_feedback_time = num_ef_batches * explicit_feedback_time_per_batch
        
        # Program synthesis: sequential per task
        initial_testing_time = num_tasks * TIME_ESTIMATES["program_synthesis_per_task"]
        
        times["initial_funsearch"] = funsearch_time
        times["initial_explicit_feedback"] = explicit_feedback_time
        times["initial_testing"] = initial_testing_time
        total += funsearch_time + explicit_feedback_time + initial_testing_time
    
    # Function evolution rounds
    function_evolution_time = 0
    for round_num in range(num_function_evolutions):
        # FunSearch for evolved functions (typically fewer functions than initial)
        # Assume ~50% of functions need evolution on average
        avg_functions_to_evolve = max(1, num_functions // 2)
        
        # FunSearch: parallelized
        num_funsearch_batches_evo = (avg_functions_to_evolve + TIME_ESTIMATES["funsearch_parallel_factor"] - 1) // TIME_ESTIMATES["funsearch_parallel_factor"]
        funsearch_evo_time = num_funsearch_batches_evo * TIME_ESTIMATES["function_evolution_funsearch"]
        
        # Explicit feedback: parallelized
        num_ef_batches_evo = (avg_functions_to_evolve + TIME_ESTIMATES["explicit_feedback_parallel_factor"] - 1) // TIME_ESTIMATES["explicit_feedback_parallel_factor"]
        explicit_feedback_evo_time = num_ef_batches_evo * TIME_ESTIMATES["function_evolution_explicit_feedback"]
        
        # Testing: sequential
        testing_evo_time = num_tasks * TIME_ESTIMATES["function_evolution_testing"]
        function_evolution_time += funsearch_evo_time + explicit_feedback_evo_time + testing_evo_time
    
    times["function_evolution"] = function_evolution_time
    total += function_evolution_time
    
    # DSL evolution (happens at end if tasks still failing)
    times["dsl_evolution"] = TIME_ESTIMATES["dsl_evolution"]
    total += TIME_ESTIMATES["dsl_evolution"]
    
    # Overhead
    overhead = TIME_ESTIMATES["overhead_per_round"] + TIME_ESTIMATES["plotting"]
    times["overhead"] = overhead
    total += overhead
    
    times["total"] = total
    return times


def calculate_first_iteration_time(
    num_functions: int,
    num_tasks: int,
    num_function_evolutions: int = 3,
    num_dsl_evolutions: int = 3
) -> Dict[str, float]:
    """Calculate time for first iteration (includes CFG generation).
    
    Args:
        num_functions: Number of functions to implement
        num_tasks: Number of tasks to test
        num_function_evolutions: Number of function evolution rounds per DSL round
        num_dsl_evolutions: Number of DSL evolution rounds (but first iteration only does 1)
    
    Returns:
        Dictionary with time breakdown
    """
    times = {}
    
    # CFG generation (only in first iteration)
    times["cfg_generation"] = TIME_ESTIMATES["cfg_generation"]
    
    # First DSL round (includes initial implementation)
    dsl_round_time = calculate_time_for_dsl_round(
        num_functions=num_functions,
        num_tasks=num_tasks,
        num_function_evolutions=num_function_evolutions,
        include_initial_implementation=True
    )
    
    times["dsl_round_0"] = dsl_round_time["total"]
    times["dsl_round_0_breakdown"] = dsl_round_time
    
    # Total
    times["total"] = times["cfg_generation"] + times["dsl_round_0"]
    
    return times


def calculate_cfg_version_time(
    num_functions: int,
    num_tasks: int,
    num_function_evolutions: int = 3,
    num_dsl_evolutions: int = 3
) -> Dict[str, float]:
    """Calculate time for a CFG version job (resumes from checkpoint).
    
    Args:
        num_functions: Number of functions to implement
        num_tasks: Number of tasks to test
        num_function_evolutions: Number of function evolution rounds per DSL round
        num_dsl_evolutions: Number of DSL evolution rounds (but each job only does 1)
    
    Returns:
        Dictionary with time breakdown
    """
    times = {}
    
    # Each CFG version job does one DSL round
    # It may skip initial implementation if resuming from checkpoint
    # But worst case, it does full implementation
    dsl_round_time = calculate_time_for_dsl_round(
        num_functions=num_functions,
        num_tasks=num_tasks,
        num_function_evolutions=num_function_evolutions,
        include_initial_implementation=True  # Worst case
    )
    
    times["dsl_round"] = dsl_round_time["total"]
    times["dsl_round_breakdown"] = dsl_round_time
    times["total"] = times["dsl_round"]
    
    return times


def format_time(minutes: float) -> str:
    """Format time in minutes to SLURM format (DD-HH:MM) or hours."""
    if minutes < 60:
        return f"{int(minutes)} minutes"
    elif minutes < 1440:  # Less than 1 day
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        return f"{hours}:{mins:02d} hours"
    else:
        days = int(minutes // 1440)
        hours = int((minutes % 1440) // 60)
        mins = int(minutes % 60)
        return f"{days}-{hours:02d}:{mins:02d} (DD-HH:MM)"


def main():
    parser = argparse.ArgumentParser(
        description="Calculate time estimates for pipeline rounds"
    )
    parser.add_argument(
        "--num-functions",
        type=int,
        default=8,
        help="Number of functions to implement (default: 8)"
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=19,
        help="Number of tasks to test (default: 19)"
    )
    parser.add_argument(
        "--num-function-evolutions",
        type=int,
        default=3,
        help="Number of function evolution rounds per DSL round (default: 3)"
    )
    parser.add_argument(
        "--num-dsl-evolutions",
        type=int,
        default=3,
        help="Number of DSL evolution rounds (default: 3)"
    )
    parser.add_argument(
        "--job-type",
        type=str,
        choices=["first", "cfg_version", "both"],
        default="both",
        help="Type of job to calculate time for (default: both)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PIPELINE TIME ESTIMATION")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Number of functions: {args.num_functions}")
    print(f"  Number of tasks: {args.num_tasks}")
    print(f"  Function evolution rounds per DSL round: {args.num_function_evolutions}")
    print(f"  DSL evolution rounds: {args.num_dsl_evolutions}")
    print()
    
    if args.job_type in ["first", "both"]:
        print("FIRST ITERATION JOB:")
        print("-" * 80)
        first_times = calculate_first_iteration_time(
            num_functions=args.num_functions,
            num_tasks=args.num_tasks,
            num_function_evolutions=args.num_function_evolutions,
            num_dsl_evolutions=args.num_dsl_evolutions
        )
        
        print(f"  CFG Generation: {format_time(first_times['cfg_generation'])}")
        print(f"  DSL Round 0: {format_time(first_times['dsl_round_0'])}")
        print(f"    - Initial FunSearch: {format_time(first_times['dsl_round_0_breakdown'].get('initial_funsearch', 0))}")
        print(f"    - Initial Explicit Feedback: {format_time(first_times['dsl_round_0_breakdown'].get('initial_explicit_feedback', 0))}")
        print(f"    - Initial Testing: {format_time(first_times['dsl_round_0_breakdown'].get('initial_testing', 0))}")
        print(f"    - Function Evolution (FunSearch + Explicit Feedback + Testing): {format_time(first_times['dsl_round_0_breakdown'].get('function_evolution', 0))}")
        print(f"    - DSL Evolution: {format_time(first_times['dsl_round_0_breakdown'].get('dsl_evolution', 0))}")
        print(f"    - Overhead: {format_time(first_times['dsl_round_0_breakdown'].get('overhead', 0))}")
        print()
        print(f"  TOTAL: {format_time(first_times['total'])}")
        print()
        print(f"  Recommended SLURM time: {format_time(first_times['total'] * 1.2)} (with 20% buffer)")
        print()
    
    if args.job_type in ["cfg_version", "both"]:
        print("CFG VERSION JOB (per round):")
        print("-" * 80)
        cfg_times = calculate_cfg_version_time(
            num_functions=args.num_functions,
            num_tasks=args.num_tasks,
            num_function_evolutions=args.num_function_evolutions,
            num_dsl_evolutions=args.num_dsl_evolutions
        )
        
        print(f"  DSL Round: {format_time(cfg_times['dsl_round'])}")
        print(f"    - Initial FunSearch: {format_time(cfg_times['dsl_round_breakdown'].get('initial_funsearch', 0))}")
        print(f"    - Initial Explicit Feedback: {format_time(cfg_times['dsl_round_breakdown'].get('initial_explicit_feedback', 0))}")
        print(f"    - Initial Testing: {format_time(cfg_times['dsl_round_breakdown'].get('initial_testing', 0))}")
        print(f"    - Function Evolution: {format_time(cfg_times['dsl_round_breakdown'].get('function_evolution', 0))}")
        print(f"    - DSL Evolution: {format_time(cfg_times['dsl_round_breakdown'].get('dsl_evolution', 0))}")
        print(f"    - Overhead: {format_time(cfg_times['dsl_round_breakdown'].get('overhead', 0))}")
        print()
        print(f"  TOTAL: {format_time(cfg_times['total'])}")
        print()
        print(f"  Recommended SLURM time: {format_time(cfg_times['total'] * 1.2)} (with 20% buffer)")
        print()
    
    print("=" * 80)
    print("NOTE: These are estimates. Adjust based on actual runtime observations.")
    print("      Factors that affect time:")
    print("      - LLM response time (varies with load)")
    print("      - Number of functions that actually need evolution")
    print("      - Whether tasks are solved early (can exit early)")
    print("      - Network latency for LLM calls")
    print("=" * 80)


if __name__ == "__main__":
    main()

