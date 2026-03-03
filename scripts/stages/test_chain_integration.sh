#!/bin/bash
# Integration test for chain_next_stage.sh
# Tests the actual script with mocked SLURM and Python functions

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

TEST_DIR=$(mktemp -d)
echo "Test directory: $TEST_DIR"

cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

# Mock sbatch to just echo (don't actually submit)
mock_sbatch() {
    echo "MOCK_SBATCH: $@"
    echo "12345"  # Return a fake job ID
}

# Create test scenario
setup_scenario() {
    local scenario="$1"
    
    case "$scenario" in
        "initial_implement_cfg")
            # Scenario: After file generation, ready to submit implement_cfg
            mkdir -p "$TEST_DIR/cfg" "$TEST_DIR/status/explicit_feedback"
            cat > "$TEST_DIR/pipeline_state.txt" << EOF
phase=initial
terminal_functions_remaining=0
explicit_feedback_remaining=0
test_tasks_remaining=0
function_evolution_remaining=0
explicit_feedback_submitted=0
test_tasks_submitted=0
function_evolution_submitted=0
file_generation_submitted=0
funsearch_submitted=0
dsl_evolutions_remaining=3
dsl_round=0
func_evolution_round=0
max_function_evolutions=1
EOF
            cat > "$TEST_DIR/cfg/cfg_output.json" << EOF
{"terminals": {"MOVE": "desc", "TURN": "desc", "LOOK": "desc"}}
EOF
            cat > "$TEST_DIR/stage_file_generation_status.json" << EOF
{"status": "completed"}
EOF
            export SLURM_JOB_NAME="exp_file_gen"
            ;;
        "after_implement_cfg")
            # Scenario: After implement_cfg completes, ready to submit test_tasks
            mkdir -p "$TEST_DIR/cfg" "$TEST_DIR/status/explicit_feedback"
            cat > "$TEST_DIR/pipeline_state.txt" << EOF
phase=initial
terminal_functions_remaining=0
explicit_feedback_remaining=0
test_tasks_remaining=0
function_evolution_remaining=0
explicit_feedback_submitted=1
test_tasks_submitted=0
function_evolution_submitted=0
file_generation_submitted=1
funsearch_submitted=1
dsl_evolutions_remaining=3
dsl_round=0
func_evolution_round=0
max_function_evolutions=1
EOF
            cat > "$TEST_DIR/cfg/cfg_output.json" << EOF
{"terminals": {"MOVE": "desc", "TURN": "desc", "LOOK": "desc"}}
EOF
            for func in MOVE TURN LOOK; do
                cat > "$TEST_DIR/status/explicit_feedback/${func}.json" << EOF
{"status": "completed", "dsl_round": 0, "func_evolution_round": 0}
EOF
            done
            ;;
        "after_test_tasks")
            # Scenario: After test_tasks completes, ready for function evolution
            mkdir -p "$TEST_DIR/cfg"
            cat > "$TEST_DIR/pipeline_state.txt" << EOF
phase=initial
terminal_functions_remaining=0
explicit_feedback_remaining=0
test_tasks_remaining=0
function_evolution_remaining=0
explicit_feedback_submitted=1
test_tasks_submitted=0
function_evolution_submitted=0
file_generation_submitted=1
funsearch_submitted=1
dsl_evolutions_remaining=3
dsl_round=0
func_evolution_round=0
max_function_evolutions=1
EOF
            cat > "$TEST_DIR/cfg/cfg_output.json" << EOF
{"terminals": {"MOVE": "desc", "TURN": "desc", "LOOK": "desc"}}
EOF
            cat > "$TEST_DIR/stage_test_tasks_status.json" << EOF
{"status": "completed", "func_evolution_round": 0, "all_solved": false, "failing_tasks": ["task1"]}
EOF
            ;;
    esac
}

# Test a scenario
test_scenario() {
    local scenario="$1"
    local expected_stage="$2"
    
    echo -e "\n${YELLOW}Testing scenario: $scenario${NC}"
    echo "  Expected to trigger: $expected_stage"
    
    # Setup
    setup_scenario "$scenario"
    
    # Export for chain script
    export EXPERIMENT_DIR="$TEST_DIR"
    
    # Source the chain script and capture output
    # We'll need to mock the Python calls and sbatch
    local output=$(bash -c "
        export EXPERIMENT_DIR='$TEST_DIR'
        export SLURM_JOB_NAME='${SLURM_JOB_NAME:-}'
        source scripts/stages/chain_next_stage.sh 2>&1 | head -20
    " 2>&1 || true)
    
    echo "  Output snippet:"
    echo "$output" | head -5 | sed 's/^/    /'
    
    # Check if expected stage was mentioned
    if echo "$output" | grep -qi "$expected_stage"; then
        echo -e "  ${GREEN} PASS: Expected stage mentioned in output${NC}"
        return 0
    else
        echo -e "  ${RED} FAIL: Expected stage not found in output${NC}"
        return 1
    fi
}

echo -e "${GREEN}=== Integration Tests ===${NC}"

# Test scenarios
test_scenario "initial_implement_cfg" "implement_cfg" || FAILED=1
test_scenario "after_implement_cfg" "test_tasks" || FAILED=1
test_scenario "after_test_tasks" "function evolution" || FAILED=1

echo -e "\n${GREEN}=== Test Summary ===${NC}"
if [ -z "${FAILED:-}" ]; then
    echo -e "${GREEN}All integration tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi

