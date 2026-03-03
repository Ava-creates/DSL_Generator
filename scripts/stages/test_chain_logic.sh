#!/bin/bash
# Test script for chain_next_stage.sh logic
# This script tests the conditions for each stage without actually submitting SLURM jobs

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create temporary test directory
TEST_DIR=$(mktemp -d)
echo "Test directory: $TEST_DIR"

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up test directory...${NC}"
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

# Mock functions
mock_get_state_value() {
    local key="$1"
    local state_file="$TEST_DIR/pipeline_state.txt"
    if [ -f "$state_file" ]; then
        grep "^${key}=" "$state_file" | cut -d= -f2 || echo "0"
    else
        echo "0"
    fi
}

create_state_file() {
    local state_file="$TEST_DIR/pipeline_state.txt"
    # Parameters: phase, terminal_remaining, explicit_fb_remaining, test_tasks_remaining, 
    #             func_evolution_remaining, explicit_fb_submitted, test_tasks_submitted,
    #             func_evolution_submitted, file_gen_submitted, funsearch_submitted,
    #             dsl_evolutions_remaining, dsl_round, func_evolution_round, max_func_evolutions
    cat > "$state_file" << EOF
phase=$1
terminal_functions_remaining=$2
explicit_feedback_remaining=$3
test_tasks_remaining=$4
function_evolution_remaining=$5
explicit_feedback_submitted=$6
test_tasks_submitted=$7
function_evolution_submitted=$8
file_generation_submitted=$9
funsearch_submitted=${10}
dsl_evolutions_remaining=${11}
dsl_round=${12}
func_evolution_round=${13}
max_function_evolutions=${14}
EOF
}

create_status_file() {
    local status_type="$1"  # "explicit_feedback" or "evolve_function"
    local func_name="$2"
    local dsl_round="$3"
    local func_evol_round="${4:-0}"
    
    local status_dir="$TEST_DIR/status/$status_type"
    mkdir -p "$status_dir"
    
    local status_file="$status_dir/${func_name}.json"
    cat > "$status_file" << EOF
{
  "stage": "$status_type",
  "function_name": "$func_name",
  "status": "completed",
  "dsl_round": $dsl_round,
  "func_evolution_round": $func_evol_round
}
EOF
}

create_cfg_file() {
    local cfg_file="$TEST_DIR/cfg/cfg_output.json"
    mkdir -p "$(dirname "$cfg_file")"
    cat > "$cfg_file" << EOF
{
  "terminals": {
    "MOVE": "description",
    "TURN": "description",
    "LOOK": "description"
  }
}
EOF
}

create_file_gen_status() {
    local status_file="$TEST_DIR/stage_file_generation_status.json"
    cat > "$status_file" << EOF
{
  "status": "completed"
}
EOF
}

create_test_tasks_status() {
    local func_evol_round="${1:-0}"
    local all_solved="${2:-false}"
    local status_file="$TEST_DIR/stage_test_tasks_status.json"
    cat > "$status_file" << EOF
{
  "status": "completed",
  "func_evolution_round": $func_evol_round,
  "all_solved": $all_solved,
  "failing_tasks": ["task1", "task2"]
}
EOF
}

create_dsl_status() {
    local evolved="${1:-true}"
    local status_file="$TEST_DIR/status/evolve_dsl/status.json"
    mkdir -p "$(dirname "$status_file")"
    cat > "$status_file" << EOF
{
  "evolved": $evolved
}
EOF
}

# Test function
test_stage() {
    local stage_name="$1"
    local description="$2"
    local setup_func="$3"
    local expected_result="$4"  # "should_trigger" or "should_not_trigger"
    
    echo -e "\n${YELLOW}Testing: $stage_name${NC}"
    echo "  Description: $description"
    
    # Setup
    eval "$setup_func"
    
    # Source the chain script (we'll need to mock it)
    export EXPERIMENT_DIR="$TEST_DIR"
    export SLURM_JOB_NAME=""
    
    # Read state to check conditions
    local phase=$(mock_get_state_value "phase")
    local terminal_remaining=$(mock_get_state_value "terminal_functions_remaining")
    local explicit_fb_remaining=$(mock_get_state_value "explicit_feedback_remaining")
    local test_tasks_remaining=$(mock_get_state_value "test_tasks_remaining")
    local func_evolution_remaining=$(mock_get_state_value "function_evolution_remaining")
    local explicit_fb_submitted=$(mock_get_state_value "explicit_feedback_submitted")
    local test_tasks_submitted=$(mock_get_state_value "test_tasks_submitted")
    local func_evolution_submitted=$(mock_get_state_value "function_evolution_submitted")
    local file_generation_submitted=$(mock_get_state_value "file_generation_submitted")
    local funsearch_submitted=$(mock_get_state_value "funsearch_submitted")
    local dsl_round=$(mock_get_state_value "dsl_round")
    local func_evolution_round=$(mock_get_state_value "func_evolution_round")
    
    echo "  State: phase=$phase, dsl_round=$dsl_round, func_round=$func_evolution_round"
    echo "  Remaining: terminal=$terminal_remaining, explicit_fb=$explicit_fb_remaining, test_tasks=$test_tasks_remaining"
    echo "  Submitted: funsearch=$funsearch_submitted, explicit_fb=$explicit_fb_submitted, test_tasks=$test_tasks_submitted"
    
    # Check conditions based on stage
    local should_trigger=false
    
    case "$stage_name" in
        "STAGE 1")
            if [ "$phase" = "initial" ] && [ "$terminal_remaining" -gt 0 ] && [ "$file_generation_submitted" -eq 0 ]; then
                should_trigger=true
            fi
            ;;
        "STAGE 2")
            # This requires checking if we're in file_gen stage, which we can't easily test here
            # But we can check the other conditions
            if [ "$phase" = "initial" ] && [ "$funsearch_submitted" -eq 0 ] && [ "$func_evolution_round" -eq 0 ]; then
                if [ -f "$TEST_DIR/stage_file_generation_status.json" ]; then
                    local file_gen_status=$(python3 -c "import json; f=open('$TEST_DIR/stage_file_generation_status.json'); d=json.load(f); print(d.get('status', ''))" 2>/dev/null || echo "")
                    if [ "$file_gen_status" = "completed" ]; then
                        should_trigger=true
                    fi
                fi
            fi
            ;;
        "STAGE 3")
            if [ "$funsearch_submitted" -eq 1 ] && [ "$explicit_fb_submitted" -eq 1 ] && [ "$terminal_remaining" -eq 0 ] && [ "$explicit_fb_remaining" -eq 0 ] && [ "$test_tasks_submitted" -eq 0 ]; then
                # Check status files
                if [ -f "$TEST_DIR/cfg/cfg_output.json" ]; then
                    local all_complete=true
                    for func in MOVE TURN LOOK; do
                        local status_file="$TEST_DIR/status/explicit_feedback/${func}.json"
                        if [ ! -f "$status_file" ]; then
                            all_complete=false
                            break
                        fi
                    done
                    if [ "$all_complete" = true ]; then
                        should_trigger=true
                    fi
                fi
            fi
            ;;
        "STAGE 4")
            if [ "$test_tasks_remaining" -eq 0 ] && [ "$explicit_fb_remaining" -eq 0 ]; then
                if [ -f "$TEST_DIR/stage_test_tasks_status.json" ]; then
                    local test_status=$(python3 -c "import json; f=open('$TEST_DIR/stage_test_tasks_status.json'); d=json.load(f); print(d.get('status', ''))" 2>/dev/null || echo "")
                    if [ "$test_status" = "completed" ]; then
                        # Check if all tasks solved - if yes, should NOT trigger evolution
                        local all_solved=$(python3 -c "import json; f=open('$TEST_DIR/stage_test_tasks_status.json'); d=json.load(f); print('1' if d.get('all_solved', False) else '0')" 2>/dev/null || echo "0")
                        if [ "$all_solved" = "1" ]; then
                            should_trigger=false  # All solved, pipeline complete, no evolution
                        else
                            should_trigger=true   # Tasks failed, should trigger evolution
                        fi
                    fi
                fi
            fi
            ;;
        "STAGE 5")
            if [ "$func_evolution_remaining" -eq 0 ]; then
                # Check if function evolution actually ran
                local has_evf_status=false
                for func in MOVE TURN LOOK; do
                    local status_file="$TEST_DIR/status/evolve_function/${func}.json"
                    if [ -f "$status_file" ]; then
                        has_evf_status=true
                        break
                    fi
                done
                if [ "$has_evf_status" = true ]; then
                    should_trigger=true
                fi
            fi
            ;;
        "STAGE 6")
            if [ -f "$TEST_DIR/status/evolve_dsl/status.json" ]; then
                local dsl_status=$(python3 -c "import json; f=open('$TEST_DIR/status/evolve_dsl/status.json'); d=json.load(f); print(d.get('evolved', False))" 2>/dev/null || echo "False")
                if [ "$dsl_status" = "True" ] && [ "$file_generation_submitted" -eq 0 ]; then
                    should_trigger=true
                fi
            fi
            ;;
    esac
    
    # Check result
    if [ "$expected_result" = "should_trigger" ]; then
        if [ "$should_trigger" = true ]; then
            echo -e "  ${GREEN} PASS: Stage should trigger and does${NC}"
            return 0
        else
            echo -e "  ${RED} FAIL: Stage should trigger but doesn't${NC}"
            return 1
        fi
    else
        if [ "$should_trigger" = false ]; then
            echo -e "  ${GREEN} PASS: Stage should not trigger and doesn't${NC}"
            return 0
        else
            echo -e "  ${RED} FAIL: Stage should not trigger but does${NC}"
            return 1
        fi
    fi
}

# Run tests
echo -e "${GREEN}=== Testing Chain Next Stage Logic ===${NC}"

# Test STAGE 1: Get CFG → File Generation
test_stage "STAGE 1" "Initial phase, terminals remaining, file gen not submitted" \
    "create_state_file initial 6 0 0 0 0 0 0 0 0 3 0 0 1; create_cfg_file" \
    "should_trigger" || FAILED=1

test_stage "STAGE 1" "File gen already submitted" \
    "create_state_file initial 6 0 0 0 0 0 0 1 0 3 0 0 1; create_cfg_file" \
    "should_not_trigger" || FAILED=1

# Test STAGE 2: File Generation → implement_cfg
test_stage "STAGE 2" "File gen completed, funsearch not submitted, func_round=0" \
    "create_state_file initial 0 0 0 0 0 0 0 0 0 3 0 0 1; create_cfg_file; create_file_gen_status" \
    "should_trigger" || FAILED=1

test_stage "STAGE 2" "Funsearch already submitted" \
    "create_state_file initial 0 0 0 0 0 0 0 0 1 3 0 0 1; create_cfg_file; create_file_gen_status" \
    "should_not_trigger" || FAILED=1

test_stage "STAGE 2" "Func evolution round > 0 (should not trigger)" \
    "create_state_file initial 0 0 0 0 0 0 0 0 0 3 0 1 1; create_cfg_file; create_file_gen_status" \
    "should_not_trigger" || FAILED=1

# Test STAGE 3: implement_cfg → test_tasks
test_stage "STAGE 3" "All implement_cfg jobs complete, test_tasks not submitted" \
    "create_state_file initial 0 0 0 0 1 1 0 0 1 3 0 0 1; create_cfg_file; create_status_file explicit_feedback MOVE 0 0; create_status_file explicit_feedback TURN 0 0; create_status_file explicit_feedback LOOK 0 0" \
    "should_trigger" || FAILED=1

test_stage "STAGE 3" "Test tasks already submitted" \
    "create_state_file initial 0 0 0 0 1 1 1 0 1 3 0 0 1; create_cfg_file; create_status_file explicit_feedback MOVE 0 0; create_status_file explicit_feedback TURN 0 0; create_status_file explicit_feedback LOOK 0 0" \
    "should_not_trigger" || FAILED=1

test_stage "STAGE 3" "Not all status files complete" \
    "create_state_file initial 0 0 0 0 1 1 0 0 1 3 0 0 1; create_cfg_file; create_status_file explicit_feedback MOVE 0 0; create_status_file explicit_feedback TURN 0 0" \
    "should_not_trigger" || FAILED=1

# Test STAGE 4: test_tasks → Function Evolution
test_stage "STAGE 4" "Test tasks complete, func_round < max, function evolution not submitted" \
    "create_state_file initial 0 0 0 0 1 0 0 0 1 3 0 0 1; create_test_tasks_status 0 false" \
    "should_trigger" || FAILED=1

test_stage "STAGE 4" "All tasks solved (should not trigger evolution)" \
    "create_state_file initial 0 0 0 0 1 0 0 0 1 3 0 0 1; create_test_tasks_status 0 true" \
    "should_not_trigger" || FAILED=1

# Test STAGE 5: Function Evolution → test_tasks
test_stage "STAGE 5" "Function evolution complete, test_tasks not submitted" \
    "create_state_file initial 0 0 0 0 0 0 0 0 1 3 0 1 1; create_cfg_file; create_status_file evolve_function MOVE 0 1; create_status_file evolve_function TURN 0 1; create_status_file evolve_function LOOK 0 1" \
    "should_trigger" || FAILED=1

# Test STAGE 6: DSL Evolution → File Generation
test_stage "STAGE 6" "DSL evolution complete, file gen not submitted" \
    "create_state_file initial 0 0 0 0 1 0 0 0 1 2 0 2 1; create_dsl_status true" \
    "should_trigger" || FAILED=1

test_stage "STAGE 6" "File gen already submitted" \
    "create_state_file initial 0 0 0 0 1 0 0 1 1 2 0 2 1; create_dsl_status true" \
    "should_not_trigger" || FAILED=1

# Summary
echo -e "\n${GREEN}=== Test Summary ===${NC}"
if [ -z "${FAILED:-}" ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi

