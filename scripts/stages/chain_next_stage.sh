#!/bin/bash
# Helper script to chain to next stage based on pipeline state
# Source this script in SLURM files: source scripts/stages/chain_next_stage.sh

# Load configuration from YAML file if available
# This ensures config values are available for chaining decisions
if [ -n "${EXPERIMENT_CONFIG:-}" ] || [ -f "config/experiment_config.yaml" ]; then
    # Only source if we're in the project root (load_config.sh expects to be there)
    if [ -f "scripts/load_config.sh" ]; then
        source scripts/load_config.sh
    fi
fi

# Return log directory: scripts/log/<experiment_name> when EXPERIMENT_DIR is set, else scripts/log
get_log_dir() {
    if [ -n "${EXPERIMENT_DIR:-}" ]; then
        echo "scripts/log/$(basename "$EXPERIMENT_DIR")"
    else
        echo "scripts/log"
    fi
}

# Ensure log directory exists (call before sbatch)
ensure_log_dir() {
    mkdir -p "$(get_log_dir)"
}

# Function to read state value
get_state_value() {
    local key="$1"
    local state_file="$EXPERIMENT_DIR/pipeline_state.txt"
    if [ -f "$state_file" ]; then
        grep "^${key}=" "$state_file" | cut -d= -f2
    else
        echo "0"
    fi
}

# Resolve stage-level status file path.
# Layout: status/dsl{N}/{stage}/status (preferred), fallback: status.json
resolve_stage_status_file() {
    local stage_name="$1"
    local dsl_round="$2"
    local preferred="$EXPERIMENT_DIR/status/dsl${dsl_round}/${stage_name}/status"
    local fallback="$EXPERIMENT_DIR/status/dsl${dsl_round}/${stage_name}/status.json"

    if [ -f "$preferred" ]; then
        echo "$preferred"
    elif [ -f "$fallback" ]; then
        echo "$fallback"
    else
        # Default to preferred path for future writes/checks
        echo "$preferred"
    fi
}

# Helper function to verify all status files are complete
# Usage: verify_all_status_complete "status_type" "dsl_round" "func_evolution_round"
# Returns: 0 if all complete, 1 otherwise
verify_all_status_complete() {
    local status_type="$1"  # "explicit_feedback" or "evolve_function"
    local dsl_round="$2"
    local func_evol_round="${3:-0}"
    local max_retries="${4:-30}"
    local retry_delay="${5:-2}"
    
    local cfg_file="$EXPERIMENT_DIR/cfg/cfg_output.json"
    if [ ! -f "$cfg_file" ]; then
        return 1
    fi
    
    local retry_count=0
    while [ $retry_count -lt $max_retries ]; do
        DSL_ROUND=$dsl_round FUNC_EVOL_ROUND=$func_evol_round STATUS_TYPE=$status_type python3 << EOF
import sys
import json
import os

experiment_dir = os.environ.get("EXPERIMENT_DIR", "")
dsl_round = int(os.environ.get("DSL_ROUND", "0"))
func_evol_round = int(os.environ.get("FUNC_EVOL_ROUND", "0"))
status_type = os.environ.get("STATUS_TYPE", "")
if not experiment_dir or not status_type:
    sys.exit(1)

cfg_file = f"{experiment_dir}/cfg/cfg_output.json"
try:
    with open(cfg_file, 'r') as f:
        cfg_data = json.load(f)
    terminals = cfg_data.get("terminals", {})
    
    all_complete = True
    for func_name in terminals.keys():
        # Determine status file paths based on type
        if status_type == "explicit_feedback":
            status_file = f"{experiment_dir}/status/dsl{dsl_round}/explicit_feedback/{func_name}.json"
            check_dsl_only = True
        elif status_type == "evolve_function":
            status_file = f"{experiment_dir}/status/dsl{dsl_round}/evolve_function/{func_name}.json"
            check_dsl_only = False
        else:
            sys.exit(1)
        
        if not os.path.exists(status_file):
            all_complete = False
            continue
        
        try:
            with open(status_file, 'r') as f:
                status = json.load(f)
            # Check if job has finished (either "completed" or "failed")
            # We proceed with test_tasks even if some functions failed, as long as all jobs have finished
            job_status = status.get("status", "")
            if job_status not in ["completed", "failed"]:
                all_complete = False
                continue
            
            # Check DSL round
            status_dsl_round = status.get("dsl_round", 0)
            if status_dsl_round is None:
                status_dsl_round = 0
            else:
                status_dsl_round = int(status_dsl_round)
            if status_dsl_round != dsl_round:
                all_complete = False
                continue
            
            # Check func evolution round if needed
            if not check_dsl_only:
                status_func_round = status.get("func_evolution_round", 0)
                if status_func_round is None:
                    status_func_round = 0
                else:
                    status_func_round = int(status_func_round)
                if status_func_round != func_evol_round:
                    all_complete = False
                    continue
        except Exception:
            all_complete = False
            continue
    
    sys.exit(0 if all_complete else 1)
except Exception:
    sys.exit(1)
EOF
        if [ $? -eq 0 ]; then
            return 0
        fi
        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $max_retries ]; then
            sleep $retry_delay
        fi
    done
    return 1
}

# Usage: submit_test_tasks_job "func_evolution_round"
submit_test_tasks_job() {
    local func_evolution_round="$1"
    local tasks_str=$(get_state_value "tasks")
    
    if [ -z "$tasks_str" ] || [ "$tasks_str" = "[]" ]; then
        echo "   No tasks found in state, cannot submit test tasks"
        return 1
    fi
    
                    ensure_log_dir
    LOG_DIR="$(get_log_dir)"
    TASKS_STR="$tasks_str" JOB_PREFIX=$job_prefix FUNC_EVOLUTION_ROUND=$func_evolution_round LOG_DIR="$LOG_DIR" python3 << EOF
import sys
import os
import subprocess
sys.path.insert(0, '/home/avani/projects/aip-lelis/avani/DSL_Generator')
from src.utils.pipeline_state import mark_test_tasks_submitted, read_state, update_state
import json

if mark_test_tasks_submitted("$EXPERIMENT_DIR"):
    state = read_state("$EXPERIMENT_DIR")
    dsl_round = str(state.get("dsl_round", 0))
    func_evolution_round = os.environ.get("FUNC_EVOLUTION_ROUND")
    if func_evolution_round is None or func_evolution_round == "":
        func_evolution_round = str(state.get("func_evolution_round", 0))
    
    # Guard: Check final functions exist for this round before submitting
    func_round_int = int(func_evolution_round) if func_evolution_round else 0
    dsl_round_int = int(dsl_round)
    if func_round_int > 0:
        cfg_path = os.path.join("$EXPERIMENT_DIR", "cfg", "cfg_output.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r') as f:
                cfg_check = json.load(f)
            terminals_check = cfg_check.get("terminals", {})
            sys.path.insert(0, '/home/avani/projects/aip-lelis/avani/DSL_Generator')
            from src.pipeline.cfg_to_funsearch_pipeline import sanitize_function_name
            final_functions_dir = os.path.join("$EXPERIMENT_DIR", "final_functions")
            missing = []
            for fn in terminals_check:
                safe = sanitize_function_name(fn)
                expected = os.path.join(final_functions_dir, f"{safe}_dsl{dsl_round_int}_func{func_round_int}.py")
                if not os.path.exists(expected):
                    missing.append(os.path.basename(expected))
            if missing:
                print(f"ERROR: Cannot submit test_tasks for func_evolution_round={func_round_int}: "
                      f"missing final function files: {missing}")
                print("Skipping test_tasks submission.")
                # Reset the mark so it can be submitted later
                update_state("$EXPERIMENT_DIR", test_tasks_submitted=0)
                sys.exit(0)
    
    # Check if already completed for this round
    status_file = f"$EXPERIMENT_DIR/status/dsl{dsl_round}/test_tasks/status"
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                status = json.load(f)
            status_func_round = status.get("func_evolution_round", 0)
            if status_func_round is None:
                status_func_round = 0
            else:
                status_func_round = int(status_func_round)
            current_func_round = int(func_evolution_round) if func_evolution_round else 0
            if status_func_round == current_func_round and status.get("status") == "completed":
                print(f"Test tasks already completed for func_evolution_round={current_func_round}. Skipping submission.")
                sys.exit(0)
        except:
            pass
    
    tasks_str = os.environ.get("TASKS_STR", "[]")
    try:
        tasks = json.loads(tasks_str)
    except:
        tasks = []
    
    tasks = list(dict.fromkeys(tasks))
    num_unique_tasks = len(tasks)
    # Note: test_tasks runs in a single job, so we don't need counters
    # Just update test_tasks_total for informational purposes
    update_state("$EXPERIMENT_DIR", 
                 test_tasks_total=num_unique_tasks)
    print(f"Updated test tasks count: {num_unique_tasks} unique tasks (after deduplication)")

    scripts_dir = "$scripts_dir"
    job_prefix = os.environ.get("JOB_PREFIX", "exp")
    tasks_space_sep = " ".join(tasks)
    
    env_vars = {
        "EXPERIMENT_DIR": "$EXPERIMENT_DIR",
        "TASKS": tasks_space_sep,
        "RECIPES_PATH": os.environ.get("RECIPES_PATH", "craft/resources/recipes.yaml"),
        "HINTS_PATH": os.environ.get("HINTS_PATH", "craft/resources/hints.yaml"),
        "MAX_ATTEMPTS": os.environ.get("MAX_ATTEMPTS", "30"),
        "DSL_ROUND": dsl_round
    }
    if func_evolution_round and func_evolution_round != "0":
        env_vars["FUNC_EVOLUTION_ROUND"] = func_evolution_round
    
    env_str = ",".join([f"{k}={v}" for k, v in env_vars.items()])
    job_name = f"{job_prefix}_test_tasks"
    # Include EXPERIMENT_CONFIG if set
    experiment_config = os.environ.get("EXPERIMENT_CONFIG", "")
    if experiment_config:
        env_str = f"{env_str},EXPERIMENT_CONFIG={experiment_config}"
    
    log_dir = os.environ.get("LOG_DIR", "scripts/log")
    subprocess.run([
        "sbatch", "--parsable", "--export", f"ALL,{env_str}",
        "--job-name", job_name,
        "--output", f"{log_dir}/stage_test_tasks_%j.out",
        "--error", f"{log_dir}/stage_test_tasks_%j.err",
        f"{scripts_dir}/stage_test_tasks.slurm"
    ], check=False)
    print(f"Submitted single test tasks job for {num_unique_tasks} tasks")
else:
    print("Test task jobs already submitted by another process")
EOF
}

# Helper function to check if test_tasks already completed for a round
# Usage: test_tasks_already_done "func_evolution_round"
# Returns: 0 if done, 1 if not done
test_tasks_already_done() {
    local func_evolution_round="$1"
    local current_dsl_round=$(get_state_value "dsl_round")
    current_dsl_round="${current_dsl_round:-0}"
    local test_tasks_status_file
    test_tasks_status_file=$(resolve_stage_status_file "test_tasks" "$current_dsl_round")
    if [ -f "$test_tasks_status_file" ]; then
        local status_func_round=$(python3 -c "import json; f=open('$test_tasks_status_file'); d=json.load(f); print(d.get('func_evolution_round') if d.get('func_evolution_round') is not None else 0)" 2>/dev/null || echo "0")
        local status_dsl_round=$(python3 -c "import json; f=open('$test_tasks_status_file'); d=json.load(f); print(d.get('dsl_round', 0))" 2>/dev/null || echo "0")
        if [ "$status_func_round" = "$func_evolution_round" ] && [ "$status_dsl_round" = "$current_dsl_round" ]; then
            return 0
        fi
    fi
    return 1
}

# Helper function to verify test_tasks actually completed
# Returns: 0 if complete, 1 if not
verify_test_tasks_complete() {
    local current_dsl_round=$(get_state_value "dsl_round")
    current_dsl_round="${current_dsl_round:-0}"
    local test_tasks_status_file
    test_tasks_status_file=$(resolve_stage_status_file "test_tasks" "$current_dsl_round")
    if [ -f "$test_tasks_status_file" ]; then
        local test_status=$(python3 -c "import json; f=open('$test_tasks_status_file'); d=json.load(f); print(d.get('status', ''))" 2>/dev/null || echo "")
        if [ "$test_status" = "completed" ]; then
            return 0
        fi
    else
        # Wait a bit for status file to be created
        local max_wait=10
        local wait_count=0
        while [ $wait_count -lt $max_wait ] && [ ! -f "$test_tasks_status_file" ]; do
            sleep 1
            wait_count=$((wait_count + 1))
        done
        if [ -f "$test_tasks_status_file" ]; then
            local test_status=$(python3 -c "import json; f=open('$test_tasks_status_file'); d=json.load(f); print(d.get('status', ''))" 2>/dev/null || echo "")
            if [ "$test_status" = "completed" ]; then
                return 0
            fi
        fi
    fi
    return 1
}

# Function to submit next stage based on state
chain_based_on_state() {
    local scripts_dir="scripts/stages"
    
    if [ -z "${EXPERIMENT_DIR:-}" ]; then
        echo "ERROR: EXPERIMENT_DIR not set; cannot chain stages safely"
        return 1
    fi
    
    local state_file="$EXPERIMENT_DIR/pipeline_state.txt"
    if [ ! -f "$state_file" ]; then
        echo "No state file found, skipping chaining"
        return
    fi
    
    # Read all state values
    local phase=$(get_state_value "phase")
    local function_impl_remaining=$(get_state_value "function_implementation_remaining")
    local function_impl_total=$(get_state_value "function_implementation_total")
    local func_evolution_remaining=$(get_state_value "function_evolution_remaining")
    local test_tasks_submitted=$(get_state_value "test_tasks_submitted")
    local func_evolution_submitted=$(get_state_value "function_evolution_submitted")
    local file_generation_submitted=$(get_state_value "file_generation_submitted")
    local dsl_evolutions_remaining=$(get_state_value "dsl_evolutions_remaining")
    local dsl_round=$(get_state_value "dsl_round")
    local func_evolution_round=$(get_state_value "func_evolution_round")
    local max_function_evolutions=$(get_state_value "max_function_evolutions")
    local max_dsl_evolutions=$(get_state_value "max_dsl_evolutions")
    local phase_print="Phase: $phase"
    local function_impl_remaining_print="function_implementation_remaining: $function_impl_remaining"
    local function_impl_total_print="function_implementation_total: $function_impl_total"
    local func_evolution_remaining_print="function_evolution_remaining: $func_evolution_remaining"
    local test_tasks_submitted_print="test_tasks_submitted: $test_tasks_submitted"
    local func_evolution_submitted_print="function_evolution_submitted: $func_evolution_submitted"
    local file_generation_submitted_print="file_generation_submitted: $file_generation_submitted"
    local dsl_evolutions_remaining_print="dsl_evolutions_remaining: $dsl_evolutions_remaining"
    local dsl_round_print="dsl_round: $dsl_round"
    local func_evolution_round_print="func_evolution_round: $func_evolution_round"
    local max_function_evolutions_print="max_function_evolutions: $max_function_evolutions"
    local max_dsl_evolutions_print="max_dsl_evolutions: $max_dsl_evolutions"
    echo "[DEBUG] Pipeline state variables:"
    echo "  $phase_print"
    echo "  $function_impl_remaining_print"
    echo "  $function_impl_total_print"
    echo "  $func_evolution_remaining_print"
    echo "  $test_tasks_submitted_print"
    echo "  $func_evolution_submitted_print"
    echo "  $file_generation_submitted_print"
    echo "  $dsl_evolutions_remaining_print"
    echo "  $dsl_round_print"
    echo "  $func_evolution_round_print"
    echo "  $max_function_evolutions_print"
    echo "  $max_dsl_evolutions_print"
    # Generate job prefix
    local exp_name=$(basename "$EXPERIMENT_DIR")
    local job_prefix="${exp_name:0:20}"
    echo "Checking pipeline state for chaining..."
    echo "  Phase: $phase, DSL round: $dsl_round, Func evolution round: $func_evolution_round"
    echo "  Function implementation remaining: $function_impl_remaining"
    echo "  Test tasks submitted: $test_tasks_submitted"
    
    # ========================================================================
    # STAGE 1: After Get CFG -> Submit File Generation
    # ========================================================================
    # Only chain to file generation if test_tasks hasn't been submitted yet
    # If test_tasks has completed, we should go to evolution instead (checked later)
    if [ "$phase" = "initial" ] && [ "$function_impl_remaining" -eq "$function_impl_total" ] && [ "$function_impl_total" -gt 0 ] && [ "$file_generation_submitted" -eq 0 ] && [ "$test_tasks_submitted" -eq 0 ]; then
        echo "Chaining to file generation..."
        ensure_log_dir
        LOG_DIR="$(get_log_dir)"
        JOB_PREFIX=$job_prefix LOG_DIR="$LOG_DIR" python3 << EOF
import sys
import os
sys.path.insert(0, '/home/avani/projects/aip-lelis/avani/DSL_Generator')
from src.utils.pipeline_state import mark_file_generation_submitted
import subprocess

if mark_file_generation_submitted("$EXPERIMENT_DIR"):
    spec_file = "${SPEC_FILE:-prompt_specifications/specification_with_updated_nld.txt}"
    job_prefix = os.environ.get("JOB_PREFIX", "exp")
    job_name = f"{job_prefix}_file_gen"
    # Include EXPERIMENT_CONFIG if set
    experiment_config = os.environ.get("EXPERIMENT_CONFIG", "")
    export_str = f"ALL,SPEC_FILE={spec_file}"
    if experiment_config:
        export_str = f"{export_str},EXPERIMENT_CONFIG={experiment_config}"
    
    log_dir = os.environ.get("LOG_DIR", "scripts/log")
    subprocess.run([
        "sbatch", "--parsable", "--export", export_str,
        "--job-name", job_name,
        "--output", f"{log_dir}/stage_file_generation_%j.out",
        "--error", f"{log_dir}/stage_file_generation_%j.err",
        "$scripts_dir/stage_file_generation.slurm"
    ], check=False)
    print("Submitted file generation job")
else:
    print("File generation already submitted by another process")
EOF
        return
    fi
    
    # ========================================================================
    # STAGE 2: After File Generation -> Submit implement_cfg jobs (one per function)
    # ========================================================================
    local file_gen_status=""
    local file_gen_status_file
    file_gen_status_file=$(resolve_stage_status_file "file_generation" "${dsl_round:-0}")
    if [ -f "$file_gen_status_file" ]; then
        file_gen_status=$(python3 -c "import json; f=open('$file_gen_status_file'); d=json.load(f); print(d.get('status', ''))" 2>/dev/null || echo "")
    fi
    
    # Check if implement_cfg jobs have already been submitted for THIS round.
    # Ignore stale status files from previous DSL/function rounds.
    local implement_cfg_already_submitted=false
    local has_status_files=$(python3 -c "
import os
import glob
import json

exp_dir = '$EXPERIMENT_DIR'
dsl_round = int('${dsl_round:-0}')
func_round = int('${func_evolution_round:-0}')

status_files = []
status_files.extend(glob.glob(os.path.join(exp_dir, 'status', f'dsl{dsl_round}', 'explicit_feedback', '*.json')))

found_for_current_round = False
for path in status_files:
    try:
        with open(path, 'r') as f:
            st = json.load(f)
        st_dsl = st.get('dsl_round', 0)
        st_func = st.get('func_evolution_round', 0)
        st_dsl = 0 if st_dsl is None else int(st_dsl)
        st_func = 0 if st_func is None else int(st_func)
        if st_dsl == dsl_round and st_func == func_round:
            found_for_current_round = True
            break
    except Exception:
        continue

print('true' if found_for_current_round else 'false')
" 2>/dev/null || echo "false")
    if [ "$has_status_files" = "true" ]; then
        implement_cfg_already_submitted=true
    fi
    
    # Trigger implement_cfg submission when file_generation completes and implement_cfg hasn't been submitted yet
    # Don't rely on SLURM_JOB_NAME matching since SLURM truncates job names
    # PATCH: Always allow implement_cfg jobs after file_generation, even if max_function_evolutions=0
    if [ "$phase" = "initial" ] && [ "$file_gen_status" = "completed" ] && [ "$implement_cfg_already_submitted" = false ] && [ "$function_impl_remaining" -eq "$function_impl_total" ]; then
        echo "File generation complete. Submitting implement_cfg package jobs (one per function, in parallel)..."
        ensure_log_dir
        LOG_DIR="$(get_log_dir)"
        JOB_PREFIX=$job_prefix DSL_ROUND=$dsl_round FUNC_EVOLUTION_ROUND=$func_evolution_round LOG_DIR="$LOG_DIR" python3 << EOF
import sys
import os
import json
sys.path.insert(0, '/home/avani/projects/aip-lelis/avani/DSL_Generator')
from src.utils.pipeline_state import read_state
import subprocess

scripts_dir = "$scripts_dir"
job_prefix = os.environ.get("JOB_PREFIX", "$job_prefix")
dsl_round = os.environ.get("DSL_ROUND", "0")
# Read func_evolution_round from state file to ensure it's correct (should be 0 after DSL evolution)
state = read_state("$EXPERIMENT_DIR")
func_evolution_round = state.get("func_evolution_round", 0)
if func_evolution_round is None:
    func_evolution_round = 0

cfg_path = os.path.join("$EXPERIMENT_DIR", "cfg", "cfg_output.json")
with open(cfg_path, 'r') as f:
    cfg_data = json.load(f)
terminals = cfg_data.get("terminals", {})

if not terminals:
    print("No terminal functions found in CFG")
    sys.exit(1)

print(f"Submitting implement_cfg jobs with dsl_round={dsl_round}, func_evolution_round={func_evolution_round}")

submitted_count = 0
for func_name in terminals.keys():
    env_vars = {
        "EXPERIMENT_DIR": "$EXPERIMENT_DIR",
        "SPEC_FILE": os.environ.get("SPEC_FILE", "prompt_specifications/specification_with_updated_nld.txt"),
        "MODEL_TYPE": os.environ.get("MODEL_TYPE", "huggingface"),
        "DSL_ROUND": dsl_round,
        "FUNC_EVOLUTION_ROUND": str(func_evolution_round),
        "FUNCTION_NAME": func_name,
        "TOTAL_SAMPLES": os.environ.get("TOTAL_SAMPLES", "1000"),
        "NUM_EXPLICIT_FEEDBACK_ITERATIONS": os.environ.get("NUM_EXPLICIT_FEEDBACK_ITERATIONS", "30")
    }
    env_str = ",".join([f"{k}={v}" for k, v in env_vars.items()])
    # Include EXPERIMENT_CONFIG if set
    experiment_config = os.environ.get("EXPERIMENT_CONFIG", "")
    if experiment_config:
        env_str = f"{env_str},EXPERIMENT_CONFIG={experiment_config}"
    
    job_name = f"{job_prefix}_impl_{func_name}"
    log_dir = os.environ.get("LOG_DIR", "scripts/log")
    result = subprocess.run([
        "sbatch", "--parsable", "--export", f"ALL,{env_str}",
        "--job-name", job_name,
        "--output", f"{log_dir}/stage_implement_cfg_%x_%j.out",
        "--error", f"{log_dir}/stage_implement_cfg_%x_%j.err",
        f"{scripts_dir}/stage_implement_cfg_single.slurm"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        submitted_count += 1
        print(f"Submitted implement_cfg job for {func_name}: {result.stdout.strip()}")
    else:
        print(f"Failed to submit job for {func_name}: {result.stderr}", file=sys.stderr)

print(f"Submitted {submitted_count}/{len(terminals)} implement_cfg package jobs")
EOF
                return
    fi
    
    # ========================================================================
    # STAGE 3: After implement_cfg (FunSearch + Explicit Feedback) -> Submit test_tasks
    # Note: implement_cfg includes both FunSearch and Explicit Feedback for each function
    # This handles both initial phase (func_round=0) and any DSL round
    # ========================================================================
    # Trigger test_tasks when: function_impl_remaining == 0 AND test_tasks_submitted == 0
    # After implement_cfg completes, we set both to 0 to trigger test_tasks
    if [ "$function_impl_remaining" -eq 0 ] && [ "$test_tasks_submitted" -eq 0 ]; then
        echo "Implement CFG package complete (FunSearch + Explicit Feedback). Checking if test tasks should be submitted..."
        
        # CRITICAL: Verify all implement_cfg jobs actually completed before submitting test_tasks
        # This checks that all explicit_feedback status files exist and are marked "completed"
        # Will retry up to 30 times (60 seconds) to handle race conditions
        if ! verify_all_status_complete "explicit_feedback" "$dsl_round" "$func_evolution_round"; then
            echo "   Not all implement_cfg jobs have completed yet. Skipping test_tasks submission."
            return
        fi
        
        # All status files are complete, so all implement_cfg jobs are done
        # Set both to 0 to trigger test_tasks (initial stage after implement_cfg)
        python3 << EOF
import sys
sys.path.insert(0, '/home/avani/projects/aip-lelis/avani/DSL_Generator')
from src.utils.pipeline_state import update_state
update_state("$EXPERIMENT_DIR", 
             function_implementation_remaining=0,
             test_tasks_submitted=0)
EOF
        function_impl_remaining=0
        test_tasks_submitted=0
        
        if test_tasks_already_done "$func_evolution_round"; then
            echo "  Test tasks already completed for func_evolution_round=$func_evolution_round"
        else
            echo "Submitting test task jobs after implement_cfg package..."
            submit_test_tasks_job "$func_evolution_round"
            return
        fi
    fi
    
    # ========================================================================
    # STAGE 4: After Test Tasks -> Submit Function Evolution or DSL Evolution
    # ========================================================================
    # Function evolution is triggered when:
    # - function_evolution_remaining > 0 (there are still function evolution jobs to do)
    # - test_tasks_submitted == 1 (test_tasks has been completed)
    # - function_implementation_remaining == max number of terminal functions (all functions need to be evolved)
    # Note: function_impl_total already read earlier in the function
    
    if [ "$test_tasks_submitted" -eq 1 ] && [ "$function_impl_remaining" -eq "$function_impl_total" ] && [ "$function_impl_total" -gt 0 ]; then
        echo "Test tasks completed. Checking if function evolution should be triggered..."
        
        # Verify test_tasks actually completed (this is the source of truth)
        if ! verify_test_tasks_complete; then
            echo "   Test tasks not actually complete yet. Skipping function evolution submission."
            return
        fi
        
        # Check if all tasks solved
        local tasks_str=$(get_state_value "tasks")
        if [ -n "$tasks_str" ] && [ "$tasks_str" != "[]" ]; then
            local all_solved=$(python3 -c "
import json
import os
tasks_str = os.environ.get('TASKS_STR', '[]')
try:
    tasks = json.loads(tasks_str)
except:
    tasks = []
status_candidates = [
    f"$EXPERIMENT_DIR/status/dsl${dsl_round:-0}/test_tasks/status",
]
status_file = next((p for p in status_candidates if os.path.exists(p)), status_candidates[0])
all_solved = False
if os.path.exists(status_file):
    with open(status_file, 'r') as f:
        status = json.load(f)
    all_solved = status.get('all_solved', False)
print('1' if all_solved else '0')
" 2>/dev/null || echo "0")
            if [ "$all_solved" = "1" ]; then
                    echo " All tasks solved! Pipeline complete."
                    return
            fi
        fi
        
        # Get evolution limits
        local max_func_evolutions=$(get_state_value "max_function_evolutions")
        if [ -z "$max_func_evolutions" ]; then
            max_func_evolutions="${MAX_FUNCTION_EVOLUTIONS:-1}"
        fi
        # Only set default for dsl_evolutions_remaining if it's truly unset (empty)
        # If it's 0, that's a valid value meaning no evolutions remaining - don't reset it!
        if [ -z "$dsl_evolutions_remaining" ]; then
            local max_dsl_evolutions=$(get_state_value "max_dsl_evolutions")
            if [ -z "$max_dsl_evolutions" ]; then
                max_dsl_evolutions="${MAX_DSL_EVOLUTIONS:-2}"
            fi
            dsl_evolutions_remaining="$max_dsl_evolutions"
        fi
        
        echo "Checking evolution conditions:"
        echo "  func_evolution_round=$func_evolution_round, max=$max_func_evolutions"
        echo "  dsl_evolutions_remaining=$dsl_evolutions_remaining"
        
        # SAFEGUARD: Ensure function evolution is attempted at least once
        # (only applies when max_function_evolutions > 0)
        local func_evolution_ever_attempted=$(get_state_value "function_evolution_submitted")
        func_evolution_ever_attempted="${func_evolution_ever_attempted:-0}"
        
        if [ "$max_func_evolutions" -gt 0 ] && [ "$func_evolution_ever_attempted" -eq 0 ]; then
            echo "   SAFEGUARD: Function evolution has never been attempted!"
            echo "  Forcing function evolution attempt..."
            if [ "$func_evolution_round" -ge "$max_func_evolutions" ]; then
                python3 << EOF
import sys
sys.path.insert(0, '/home/avani/projects/aip-lelis/avani/DSL_Generator')
from src.utils.pipeline_state import update_state, read_state
state = read_state("$EXPERIMENT_DIR")
current_round = state.get("func_evolution_round", 0)
if current_round >= ${max_func_evolutions}:
    update_state("$EXPERIMENT_DIR", func_evolution_round=0)
    print(f"Reset func_evolution_round from {current_round} to 0")
EOF
                func_evolution_round=$(get_state_value "func_evolution_round")
                func_evolution_round="${func_evolution_round:-0}"
            fi
        fi
        
        # Submit Function Evolution
        # Function evolution is triggered when:
        # - test_tasks_submitted == 1 (test_tasks has been completed)
        # - function_implementation_remaining == function_implementation_total (all functions need to be evolved)
        # - func_evolution_round < max_func_evolutions (function evolution rounds left)
        # Note: function_evolution_remaining will be initialized when jobs are submitted
        if [ "$func_evolution_round" -lt "$max_func_evolutions" ]; then
            local func_evolution_submitted=$(get_state_value "function_evolution_submitted")
            if [ "$func_evolution_submitted" -eq 0 ]; then
                echo "Submitting function evolution jobs..."
                local cfg_file="$EXPERIMENT_DIR/cfg/cfg_output.json"
                if [ -f "$cfg_file" ]; then
                    ensure_log_dir
                    LOG_DIR="$(get_log_dir)"
                    TASKS_STR="$tasks_str" JOB_PREFIX=$job_prefix DSL_ROUND=$dsl_round LOG_DIR="$LOG_DIR" python3 << EOF
import sys
import os
import json
sys.path.insert(0, '/home/avani/projects/aip-lelis/avani/DSL_Generator')
from src.utils.pipeline_state import mark_function_evolution_submitted
import subprocess

if mark_function_evolution_submitted("$EXPERIMENT_DIR"):
    with open("$cfg_file", 'r') as f:
        cfg_data = json.load(f)
    terminals = cfg_data.get("terminals", {})

    # Set function_evolution_total and function_evolution_remaining when first submitting
    from src.utils.pipeline_state import update_state
    num_terminals = len(terminals)
    update_state("$EXPERIMENT_DIR",
                 function_evolution_total=num_terminals,
                 function_evolution_remaining=num_terminals)
    
    tasks_str = os.environ.get("TASKS_STR", "[]")
    try:
        tasks = json.loads(tasks_str)
    except:
        tasks = []

    status_candidates = [
        f"$EXPERIMENT_DIR/status/dsl${dsl_round:-0}/test_tasks/status",
    ]
    status_file = next((p for p in status_candidates if os.path.exists(p)), status_candidates[0])
    failing_tasks = []
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                status = json.load(f)
            failing_tasks = status.get("failing_tasks", [])
        except:
            failing_tasks = tasks
    else:
        failing_tasks = tasks

    if not failing_tasks:
        print("No failing tasks detected; skipping function evolution submission.")
        sys.exit(0)

    failing_tasks_str = " ".join(failing_tasks)
    scripts_dir = "$scripts_dir"
    job_prefix = os.environ.get("JOB_PREFIX", "exp")
    dsl_round = os.environ.get("DSL_ROUND", "0")
    
    func_evolution_round = int(os.environ.get("FUNC_EVOLUTION_ROUND", "0"))
    for func_name in terminals.keys():
        env_vars = {
            "EXPERIMENT_DIR": "$EXPERIMENT_DIR",
            "SPEC_FILE": os.environ.get("SPEC_FILE", "prompt_specifications/specification_with_updated_nld.txt"),
            "FUNCTION_NAME": func_name,
            "FAILING_TASKS": failing_tasks_str,
            "MODEL_TYPE": os.environ.get("MODEL_TYPE", "huggingface"),
            "DSL_ROUND": dsl_round,
            "FUNC_EVOLUTION_ROUND": str(func_evolution_round + 1),
            "TOTAL_SAMPLES": os.environ.get("TOTAL_SAMPLES", "1000")
        }
        env_str = ",".join([f"{k}={v}" for k, v in env_vars.items()])
        # Include EXPERIMENT_CONFIG if set
        experiment_config = os.environ.get("EXPERIMENT_CONFIG", "")
        if experiment_config:
            env_str = f"{env_str},EXPERIMENT_CONFIG={experiment_config}"
        
        job_name = f"{job_prefix}_evf_{func_name}"
        log_dir = os.environ.get("LOG_DIR", "scripts/log")
        subprocess.run([
            "sbatch", "--parsable", "--export", f"ALL,{env_str}",
            "--job-name", job_name,
            "--output", f"{log_dir}/stage_evolve_function_%x_%j.out",
            "--error", f"{log_dir}/stage_evolve_function_%x_%j.err",
            f"{scripts_dir}/stage_evolve_function_single.slurm"
        ], check=False)
    print("Submitted function evolution jobs")
else:
    print("Function evolution jobs already submitted by another process")
EOF
                fi
            fi
            return
        fi
        
        # Submit DSL Evolution (only after function evolution exhausted)
        # DSL evolution is triggered when:
        # - test_tasks_submitted == 1 (test_tasks has been completed)
        # - function_implementation_remaining == function_implementation_total (all functions need to be evolved)
        # - func_evolution_round >= max_func_evolutions (function evolution exhausted)
        # - dsl_evolutions_remaining > 0 (DSL evolution rounds remaining)
        if [ "$func_evolution_round" -ge "$max_func_evolutions" ] && [ "$dsl_evolutions_remaining" -gt 0 ]; then
            if [ "$func_evolution_ever_attempted" -eq 0 ] && [ "$max_func_evolutions" -gt 0 ]; then
                echo "   ERROR: DSL evolution requested but function evolution never attempted!"
                return
            fi
            
            echo "Function evolution exhausted. Checking DSL evolution..."
            local dsl_evolution_submitted=$(get_state_value "dsl_evolution_submitted")
            if [ "$dsl_evolution_submitted" -eq 0 ]; then
                echo "Submitting DSL evolution job..."
                ensure_log_dir
                LOG_DIR="$(get_log_dir)"
                TASKS_STR="$tasks_str" JOB_PREFIX=$job_prefix LOG_DIR="$LOG_DIR" python3 << EOF
import sys
import os
import json
sys.path.insert(0, '/home/avani/projects/aip-lelis/avani/DSL_Generator')
from src.utils.pipeline_state import mark_dsl_evolution_submitted
import subprocess

if mark_dsl_evolution_submitted("$EXPERIMENT_DIR"):
    tasks_str = os.environ.get("TASKS_STR", "[]")
    try:
        tasks = json.loads(tasks_str)
    except:
        tasks = []

    status_candidates = [
        f"$EXPERIMENT_DIR/status/dsl${dsl_round:-0}/test_tasks/status",
    ]
    status_file = next((p for p in status_candidates if os.path.exists(p)), status_candidates[0])
    failing_tasks = []
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                status = json.load(f)
            failing_tasks = status.get("failing_tasks", [])
        except:
            failing_tasks = tasks
    else:
        failing_tasks = tasks

    failing_tasks_str = " ".join(failing_tasks)
    env_vars = {
        "EXPERIMENT_DIR": "$EXPERIMENT_DIR",
        "FAILING_TASKS": failing_tasks_str,
        "RECIPES_PATH": "${RECIPES_PATH:-craft/resources/recipes.yaml}",
        "MAX_DSL_RETRIES": "10",
        "DSL_VERSION": "$dsl_round"
    }
    env_str = ",".join([f"{k}={v}" for k, v in env_vars.items()])
    # Include EXPERIMENT_CONFIG if set
    experiment_config = os.environ.get("EXPERIMENT_CONFIG", "")
    if experiment_config:
        env_str = f"{env_str},EXPERIMENT_CONFIG={experiment_config}"
    
    job_prefix = os.environ.get("JOB_PREFIX", "exp")
    job_name = f"{job_prefix}_evdsl"
    log_dir = os.environ.get("LOG_DIR", "scripts/log")
    subprocess.run([
        "sbatch", "--parsable", "--export", f"ALL,{env_str}",
        "--job-name", job_name,
        "--output", f"{log_dir}/stage_evolve_dsl_%j.out",
        "--error", f"{log_dir}/stage_evolve_dsl_%j.err",
        "$scripts_dir/stage_evolve_dsl.slurm"
    ], check=False)
    print("Submitted DSL evolution job")
else:
    print("DSL evolution job already submitted by another process")
EOF
            fi
            return
        else
            echo "All evolution rounds exhausted. Pipeline stopping."
            return
        fi
    fi
    
    # ========================================================================
    # STAGE 5: After Function Evolution -> Submit test_tasks
    # ========================================================================
    # Trigger test_tasks when: function_impl_remaining == 0 AND test_tasks_submitted == 0
    # After function evolution completes, stage_evolve_function_single.py sets both to 0
    if [ "$func_evolution_remaining" -eq 0 ] && [ "$function_impl_remaining" -eq 0 ] && [ "$test_tasks_submitted" -eq 0 ]; then
        # Check if function evolution actually ran
        local cfg_file="$EXPERIMENT_DIR/cfg/cfg_output.json"
        local func_evolution_actually_ran=false
        if [ -f "$cfg_file" ]; then
            local has_evf_status=$(python3 -c "
import sys
import os
import json
sys.path.insert(0, '/home/avani/projects/aip-lelis/avani/DSL_Generator')
with open('$cfg_file', 'r') as f:
    cfg_data = json.load(f)
terminals = cfg_data.get('terminals', {})
for func_name in terminals.keys():
    status_file = os.path.join('$EXPERIMENT_DIR', f"dsl${dsl_round:-0}", 'evolve_function', f"{func_name}.json")
    if os.path.exists(status_file):
        print('true')
        sys.exit(0)
print('false')
" 2>/dev/null || echo "false")
            if [ "$has_evf_status" = "true" ]; then
                func_evolution_actually_ran=true
            fi
        fi
        
        if [ "$func_evolution_actually_ran" = true ]; then
            echo "All function evolution jobs complete. Checking if test tasks should be submitted..."
            
            # Verify all function evolution jobs completed
            if ! verify_all_status_complete "evolve_function" "$dsl_round" "$func_evolution_round"; then
                echo "   Not all function evolution jobs have completed yet. Skipping test_tasks submission."
                return
            fi
            
            # After function evolution, test_tasks_submitted should be reset to 0 by stage_evolve_function_single.py
            # We can submit test_tasks directly (function evolution already includes FunSearch + Explicit Feedback)
            test_tasks_submitted=$(get_state_value "test_tasks_submitted")
            test_tasks_submitted="${test_tasks_submitted:-0}"
            
            if [ "$test_tasks_submitted" -eq 0 ]; then
                if test_tasks_already_done "$func_evolution_round"; then
                        echo "  Test tasks already completed for func_evolution_round=$func_evolution_round"
                else
                    echo "Submitting test task jobs after function evolution..."
                    submit_test_tasks_job "$func_evolution_round"
                        return
                fi
            else
                echo "   test_tasks already submitted, waiting for completion..."
            fi
        fi
    fi
    
    # ========================================================================
    # STAGE 6: After DSL Evolution -> Submit file generation
    # ========================================================================
    local dsl_status_file_to_check="$EXPERIMENT_DIR/status/dsl${dsl_round:-0}/evolve_dsl/status"
    
    if [ -n "$dsl_status_file_to_check" ]; then
        local dsl_status=$(python3 -c "import json; f=open('$dsl_status_file_to_check'); d=json.load(f); print(d.get('evolved', False))" 2>/dev/null || echo "False")
        if [ "$dsl_status" = "True" ] && [ "$file_generation_submitted" -eq 0 ]; then
            echo "DSL evolution complete. Submitting file generation job..."
            ensure_log_dir
            LOG_DIR="$(get_log_dir)"
            JOB_PREFIX=$job_prefix LOG_DIR="$LOG_DIR" python3 << EOF
import sys
import os
sys.path.insert(0, '/home/avani/projects/aip-lelis/avani/DSL_Generator')
from src.utils.pipeline_state import mark_file_generation_submitted, read_state
import subprocess

if mark_file_generation_submitted("$EXPERIMENT_DIR"):
    state = read_state("$EXPERIMENT_DIR")
    dsl_round = state.get("dsl_round", 0)
    func_evolution_round = 0  # After DSL evolution, always start with func0 (initial functions for new DSL)
    spec_file = "${SPEC_FILE:-prompt_specifications/specification_with_updated_nld.txt}"
    job_prefix = os.environ.get("JOB_PREFIX", "exp")
    # Include EXPERIMENT_CONFIG if set
    experiment_config = os.environ.get("EXPERIMENT_CONFIG", "")
    export_str = f"ALL,SPEC_FILE={spec_file},DSL_ROUND={dsl_round},FUNC_EVOLUTION_ROUND={func_evolution_round}"
    if experiment_config:
        export_str = f"{export_str},EXPERIMENT_CONFIG={experiment_config}"
    
    job_name = f"{job_prefix}_file_gen"
    log_dir = os.environ.get("LOG_DIR", "scripts/log")
    subprocess.run([
        "sbatch", "--parsable", "--export", export_str,
        "--job-name", job_name,
        "--output", f"{log_dir}/stage_file_generation_%j.out",
        "--error", f"{log_dir}/stage_file_generation_%j.err",
        "$scripts_dir/stage_file_generation.slurm"
    ], check=False)
    print(f"Submitted file generation job after DSL evolution (dsl_round={dsl_round}, func_evolution_round={func_evolution_round})")
else:
    print("File generation already submitted by another process")
EOF
            return
        fi
    fi
    
    echo "No chaining conditions met."
}
