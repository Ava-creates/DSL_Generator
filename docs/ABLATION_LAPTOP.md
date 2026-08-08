# Laptop ablation (no SLURM)

Two ablation designs are supported on a laptop with the **Aleph API** (no GPU, no SLURM).

---

## A. Fixed-CFG ablation (recommended for your RQ)

**Isolates terminal-function generation** while holding the DSL fixed.

1. Fix CFG = DSL~1 from HF run 4 (`19/20` tasks) — same grammar for all arms
2. **FunSearch arm**: use run 4 results as-is (**not regenerated**)
3. Regenerate 9 terminal functions only for `llm_best_of_n` and `llm_chained` + explicit feedback
4. Program synthesis only — 20 tasks × 10 seeds — **no DSL evolution loop**

### Step 1 — Export assets from Vulcan (once)

On the cluster:

```bash
bash scripts/export_fixed_cfg_ablation_assets.sh \
  experiments/pipeline_hf_20260611_151047_run4_2104814 1 run4_dsl1.tar.gz
scp run4_dsl1.tar.gz your-laptop:~/Desktop/DSL_Generator/
```

On the laptop, unpack into a source tree the script can read:

```bash
mkdir -p experiments/pipeline_hf_20260611_151047_run4_2104814
tar -xzf run4_dsl1.tar.gz
# paths inside tarball preserve experiments/pipeline_hf_... layout
```

### Step 2 — Run ablation locally

```bash
export OPENAI_COMPAT_API_KEY="..."
export OPENAI_COMPAT_BASE_URL=https://inference.vulcan.alliancecan.ca
export OPENAI_COMPAT_MODEL=gpt-oss-120b
export MODEL_TYPE=openai_compat

# Smoke: one mode, 2 tasks
python scripts/run_fixed_cfg_ablation_local.py \
  --modes llm_chained \
  --total-samples 20 \
  --tasks get[wood] get[grass]

# Full ablation: llm_chained + llm_best_of_n (funsearch printed from source, not re-run)
python scripts/run_fixed_cfg_ablation_local.py
```

Results: `experiments/ablation_fixed_cfg_*` with tasks/seeds printed at the end. FunSearch row comes from `--source` at startup.

---

## B. Baseline ablation (no DSL)

Compare `llm_chained` vs `llm_best_of_n` for **direct Python per-task functions** (no CFG).

This is **not** the full DSL pipeline (that stays on Vulcan). It answers: *without a DSL, can simple LLM search + explicit feedback solve Craft tasks?*

## What runs (baseline)

For each of 20 Craft tasks (or a subset):

1. Build a task-specific prompt (task_env — no testcase grids, no GPU)
2. Generate candidates with `llm_chained` or `llm_best_of_n` (100 samples default)
3. Run explicit feedback (30 iterations)
4. Evaluate final functions on 10 seeds → `results_tracking/baseline_final_eval.json`

## Setup

```bash
git clone https://github.com/Ava-creates/DSL_Generator.git
cd DSL_Generator
git checkout laptop-ablation
git submodule update --init funsearch
```

The `funsearch` directory is a **git submodule** (not included in a plain clone). If you see `ModuleNotFoundError: No module named 'funsearch.implementation'`, run the submodule command above.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-laptop.txt
```

### Aleph API key

**Do not commit your key.** Either:

```bash
export OPENAI_COMPAT_API_KEY="your-key-here"
```

or copy the example and fill in:

```bash
cp key.txt.example key.txt
# edit key.txt — first non-empty line is the key
```

### Aleph endpoint (Alliance VPN usually required off-cluster)

```bash
export OPENAI_COMPAT_BASE_URL=https://inference.vulcan.alliancecan.ca
export OPENAI_COMPAT_CHAT_PATH=/v1/chat/completions
export OPENAI_COMPAT_MODEL=gpt-oss-120b
export OPENAI_COMPAT_MAX_PARALLEL=4
```

Smoke-test connectivity:

```bash
python -m src.utils.openai_compat_cold_start
```

## Run

**Smoke test (2 tasks, ~few minutes of API time):**

```bash
bash scripts/run_ablation_laptop.sh config/ablation_laptop_smoke.yaml --tasks "Get wood" "Get grass"
```

**Full ablation — llm_chained:**

```bash
bash scripts/run_ablation_laptop.sh config/ablation_laptop_llm_chained.yaml
```

**Full ablation — llm_best_of_n:**

```bash
bash scripts/run_ablation_laptop.sh config/ablation_laptop_llm_best_of_n.yaml
```

Results land under `experiments/ablation_laptop_*` (gitignored locally).

## Modes compared

| Mode | Behavior |
|------|----------|
| `llm_best_of_n` | Same prompt every sample; pick best by grid score |
| `llm_chained` | Each step sees previous function body + score |
| `funsearch` | Full FunSearch (cluster default; needs GPU for local vLLM) |

Both ablation modes use the same evaluator and explicit-feedback stage as the main pipeline.

## Troubleshooting

- **`No module named 'funsearch.implementation'`** → run `git submodule update --init funsearch`
- **403 / connection errors** → connect Alliance VPN
- **Slow** → lower `total_samples` in the yaml (smoke uses 20)
- **Import errors** → `pip install -r requirements-laptop.txt` from repo root
