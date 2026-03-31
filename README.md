# DSL_Generator

In this project, we are working on general DSL generator pipeline that can be used for multiple domains with minimal information about the domain itself.

## Quick Start

**Full Pipeline**: Use `scripts/submit_with_config.sh` to run the complete pipeline on vulcan.
Please update the config script before running to adjust hyperparameters.

**Interactive Node**: `salloc --account=aip-lelis --gres=gpu:4 --mem=256G --time=04:00:00`

## Running Individual Pipeline Stages

### 1. File Generation (CFG + Function Implementations)

Generate function-specific prompts and implementations for a specific DSL version:

```bash
python src/pipeline/stages/stage_file_generation.py \
    --experiment_dir experiments/experiment_20260302_155324_4209548 \
    --dsl_round 0 \
    --func_evolution_round 0 \
    --recipes_path craft/resources/recipes.yaml
```

**Key Parameters:**
- `--dsl_round`: DSL version number (0, 1, 2, ...)
- `--func_evolution_round`: Function evolution round within the DSL version (0, 1, 2, ...)
- `--recipes_path`: Path to the recipes YAML file

### 2. Test Tasks (Program Synthesis + Evaluation)

Test specific tasks with a function version:

```bash
python src/pipeline/stages/stage_test_tasks.py \
    --experiment_dir experiments/experiment_20260302_155324_4209548 \
    --tasks get[gem] get[iron] get[wood] get[grass] get[gold] make[plank] make[stick] make[cloth] make[rope] make[bridge] make[bundle] make[flag] make[bed] make[axe] make[shears] make[ladder] make[goldarrow] \
    --dsl_round 2 \
    --func_evolution_round 0 \
    --max_attempts 50
```

**Key Parameters:**
- `--tasks`: List of tasks to test (space-separated)
- `--dsl_round`: DSL round (selects cfg_output_<round>.json)
- `--func_evolution_round`: Function implementation version to use
- `--max_attempts`: Maximum synthesis attempts per task

### 3. DSL Evolution

Evolve the DSL based on failed programs from a specific DSL version:

```bash
python src/pipeline/stages/stage_evolve_dsl.py \
    --experiment_dir experiments/experiment_20260302_155324_4209548 \
    --failing_tasks get[gem] get[iron] get[wood] get[grass] get[gold] make[stick] make[cloth] make[rope] make[bridge] make[bundle] make[flag] make[bed] make[axe] make[shears] make[ladder] make[goldarrow] \
    --recipes_path craft/resources/recipes.yaml \
    --max_retries 10 \
    --dsl_version 0
```

**Key Parameters:**
- `--failing_tasks`: Tasks that failed in the current DSL version
- `--dsl_version`: DSL version to evolve FROM (reads cfg_output_X.json, saves cfg_output_{X+1}.json)
- `--max_retries`: Maximum evolution attempts

**Important**: DSL evolution only reads failed programs from the specified `--dsl_version` (cfg_version field in synthesis_results.json), ensuring it only considers failures from the current DSL iteration.

### 4. Function Evolution

Evolve specific function implementations:

```bash
python src/pipeline/stages/stage_evolve_functions.py \
    --experiment_dir experiments/experiment_20260302_155324_4209548 \
    --dsl_round 0 \
    --func_evolution_round 0 \
    --failing_tasks get[gem] make[axe] \
    --recipes_path craft/resources/recipes.yaml
```

### 5. Single Function Evolution

Evolve a single function implementation:

```bash
python src/pipeline/stages/stage_evolve_function_single.py \
    --experiment_dir experiments/experiment_20260302_155324_4209548 \
    --dsl_round 0 \
    --func_evolution_round 0 \
    --function_name PICKUP \
    --failing_tasks get[gem] get[wood] \
    --recipes_path craft/resources/recipes.yaml
```

## File Structure and Versioning

### DSL Versions
- `cfg/cfg_output_0.json` - Initial DSL (DSL round 0)
- `cfg/cfg_output_1.json` - First evolution (DSL round 1)
- `cfg/cfg_output_2.json` - Second evolution (DSL round 2)

### Function Versions
- `final_functions/PICKUP_dsl0_func0.py` - Initial PICKUP implementation for DSL 0
- `final_functions/PICKUP_dsl0_func1.py` - First evolution of PICKUP for DSL 0
- `final_functions/PICKUP_dsl1_func0.py` - Initial PICKUP implementation for DSL 1

### Results Tracking
- `results_tracking/synthesis_results.json` - All program synthesis results with inventory traces
- `results_tracking/interactions.json` - Environment interaction counts
- `results_tracking/explicit_feedback/` - Explicit feedback results and plots



### To run individual programs generated during the pipeline and to see grid at each step use this - 

```bash
cd /home/avani/projects/aip-lelis/avani/DSL_Generator && source new_dsl_env/bin/activate && python tests/run_program_with_inventory.py \
  --experiment_dir experiments/experiment_20260308_124523_15556 \
  --task "make[flag]" \
  --program "TURN(RIGHT);PICKUP;TURN(LEFT);MOVE(NORTH);TURN(RIGHT);PICKUP;TURN(LEFT);TURN(LEFT);PICKUP;TURN(RIGHT);TURN(RIGHT);MOVE(EAST);MOVE(EAST);MOVE(EAST);MOVE(EAST);MOVE(EAST);MOVE(EAST);TURN(LEFT);MOVE(NORTH);MOVE(NORTH);MOVE(NORTH);MOVE(NORTH);TURN(LEFT);MOVE(WEST);TURN(RIGHT);MOVE(NORTH);MOVE(NORTH);TURN(RIGHT);MOVE(EAST);CRAFT(STICK,WORKSHOP2);CRAFT(CLOTH,WORKSHOP2);TURN(LEFT);MOVE(WEST);MOVE(WEST);TURN(LEFT);MOVE(SOUTH);MOVE(SOUTH);MOVE(SOUTH);MOVE(SOUTH);MOVE(SOUTH);MOVE(SOUTH);MOVE(SOUTH);CRAFT(FLAG,WORKSHOP0)"
```
#### Arguments - 

 `--experiment_dir` is where the cfg is read from and that is the directory where the grid logs would be stored 


## Things hardcoded - 
`target` for number of grids in 



### Testing dsl prompts-
`python tests/test_dsl_evolution_prompts.py \
  --failure-prompts \
    prompt_specifications/failure_analysis.txt \
    prompt_specifications/failure_analysis_v2.txt \
    prompt_specifications/failure_analysis_v3.txt \
  --cfg-prompts \
    prompt_specifications/cfg_evolution.txt \
    prompt_specifications/cfg_evolution_v2.txt


python tests/test_dsl_evolution_prompts.py \
  --failure-prompts \
    prompt_specifications/failure_analysis.txt \
    prompt_specifications/failure_analysis_1.txt \
  --cfg-prompts \
    prompt_specifications/cfg_evolution.txt 

##NLD

`
python /home/avani/projects/aip-lelis/avani/DSL_Generator/tests/test_dsl_evolution_prompts.py \
  --failure-prompts \
    /home/avani/projects/aip-lelis/avani/DSL_Generator/prompt_specifications/failure_analysis_nld.txt \
  --cfg-prompts \
    /home/avani/projects/aip-lelis/avani/DSL_Generator/prompt_specifications/cfg_evolution.txt \
  --nld-paths \
    /home/avani/projects/aip-lelis/avani/DSL_Generator/prompt_specifications/nld.txt \
    /home/avani/projects/aip-lelis/avani/DSL_Generator/prompt_specifications/nld_wihtitemistool.txt \
  --recipes-path \
    /home/avani/projects/aip-lelis/avani/DSL_Generator/craft/resources/recipes.yaml \
  --failed-programs-path \
    /home/avani/projects/aip-lelis/avani/DSL_Generator/scripts/data/dsl0_failed_programs.txt \
  --out-dir \
    /home/avani/projects/aip-lelis/avani/DSL_Generator/experiments/prompt_tests/failure_analysis_nld_compare
`