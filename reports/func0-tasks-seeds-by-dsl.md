# Func0 Task Seed Results by DSL

Source files:

- `experiments/pipeline_hf_20260423_181855_1747661/task_runs/test_tasks/dsl0/func0/*/results_tracking/program_synthesis_seed_outcomes.jsonl`
- `experiments/pipeline_hf_20260423_181855_1747661/results_tracking/dsl1/func0/tasks/*/program_synthesis_seed_outcomes.jsonl`
- `experiments/pipeline_hf_20260423_181855_1747661/results_tracking/dsl2/func0/tasks/*/program_synthesis_seed_outcomes.jsonl`
- Baseline (final eval on the same 10 seeds): `experiments/baseline_task_env_20260423_181215_1601634/results_tracking/baseline_final_eval.json`

## DSL Summary

| DSL | Tasks Solved | Total Tasks | Seeds Solved | Total Seeds |
|---|---:|---:|---:|---:|
| dsl0 | 14 | 17 | 116 | 170 |
| dsl1 | 14 | 17 | 120 | 170 |
| dsl2 | 14 | 17 | 115 | 170 |
| baseline (`baseline_task_env_20260423_181215_1601634`) | 15 | 17 | 144 | 170 |

*(“Tasks solved” = at least one seed solved for that task; matches the `*_ solved?` columns in the table below.)*

## Per-task Seed Solves (func0)

| Task | baseline seeds solved | dsl0 seeds solved | dsl1 seeds solved | dsl2 seeds solved | delta (dsl1 - dsl0) | delta (dsl1 - baseline) | delta (dsl2 - dsl1) | delta (dsl2 - baseline) | baseline solved? | dsl0 solved? | dsl1 solved? | dsl2 solved? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| get_gem | 10/10 | 0/10 | 0/10 | 0/10 | 0 | -10 | 0 | -10 | yes | no | no | no |
| get_gold | 0/10 | 0/10 | 0/10 | 0/10 | 0 | 0 | 0 | 0 | no | no | no | no |
| get_grass | 10/10 | 3/10 | 10/10 | 10/10 | +7 | 0 | 0 | 0 | yes | yes | yes | yes |
| get_iron | 10/10 | 2/10 | 10/10 | 10/10 | +8 | 0 | 0 | 0 | yes | yes | yes | yes |
| get_wood | 10/10 | 0/10 | 10/10 | 10/10 | +10 | 0 | 0 | 0 | yes | no | yes | yes |
| make_axe | 10/10 | 9/10 | 9/10 | 9/10 | 0 | -1 | 0 | -1 | yes | yes | yes | yes |
| make_bed | 9/10 | 10/10 | 5/10 | 7/10 | -5 | -4 | +2 | -2 | yes | yes | yes | yes |
| make_bridge | 10/10 | 10/10 | 10/10 | 10/10 | 0 | 0 | 0 | 0 | yes | yes | yes | yes |
| make_bundle | 10/10 | 9/10 | 10/10 | 10/10 | +1 | 0 | 0 | 0 | yes | yes | yes | yes |
| make_cloth | 9/10 | 10/10 | 10/10 | 10/10 | 0 | +1 | 0 | +1 | yes | yes | yes | yes |
| make_flag | 9/10 | 10/10 | 7/10 | 5/10 | -3 | -2 | -2 | -4 | yes | yes | yes | yes |
| make_goldarrow | 0/10 | 8/10 | 0/10 | 0/10 | -8 | 0 | 0 | 0 | no | yes | no | no |
| make_ladder | 9/10 | 8/10 | 6/10 | 2/10 | -2 | -3 | -4 | -7 | yes | yes | yes | yes |
| make_plank | 10/10 | 7/10 | 9/10 | 9/10 | +2 | -1 | 0 | -1 | yes | yes | yes | yes |
| make_rope | 10/10 | 10/10 | 9/10 | 10/10 | -1 | -1 | +1 | 0 | yes | yes | yes | yes |
| make_shears | 9/10 | 10/10 | 8/10 | 3/10 | -2 | -1 | -5 | -6 | yes | yes | yes | yes |
| make_stick | 9/10 | 10/10 | 7/10 | 10/10 | -3 | -2 | +3 | +1 | yes | yes | yes | yes |
