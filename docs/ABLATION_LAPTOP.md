# Laptop ablation (no SLURM)

Two ablation designs are supported on a laptop with the **Aleph API** (no GPU, no SLURM).

---

## A. Fixed-CFG ablation (recommended for your RQ)

**Isolates terminal-function generation** while holding the DSL fixed.

1. Fix CFG = DSL~1 from HF run 4 (`19/20` tasks) — same grammar for all arms
2. **FunSearch arm**: use run 4 results as-is (**not regenerated**)
3. Regenerate terminal functions for `llm_best_of_n` and `llm_chained` + explicit feedback, evaluated on **the same DSL~1 grid specs** FunSearch used in run 4 (copied from `--source`, not regenerated). **Sample budget: 500 per terminal** (same as FunSearch on the cluster).
4. Program synthesis only — 20 tasks × **the same 10 seeds** as run 4 — **no DSL evolution loop**

### Step 1 — Export assets from Vulcan (once)

On the cluster (optional — tarball is also in the repo under `ablation_assets/`):

```bash
bash scripts/export_fixed_cfg_ablation_assets.sh \
  experiments/pipeline_hf_20260611_151047_run4_2104814 1 run4_dsl1.tar.gz
scp run4_dsl1.tar.gz your-laptop:~/Desktop/DSL_Generator/
```

**From GitHub (easiest on laptop):**

```bash
cd ~/Desktop/DSL_Generator
git pull origin laptop-ablation
tar -xzf ablation_assets/run4_dsl1.tar.gz
test -f experiments/pipeline_hf_20260611_151047_run4_2104814/cfg/cfg_output_1.json && echo OK
```

Or download only the tarball:

```bash
curl -L -o run4_dsl1.tar.gz \
  https://github.com/Ava-creates/DSL_Generator/raw/laptop-ablation/ablation_assets/run4_dsl1.tar.gz
cd ~/Desktop/DSL_Generator && tar -xzf ~/Downloads/run4_dsl1.tar.gz
```

### Step 2 — Run ablation locally

```bash
export OPENAI_COMPAT_API_KEY="..."
export OPENAI_COMPAT_BASE_URL=https://inference.vulcan.alliancecan.ca
export OPENAI_COMPAT_MODEL=gpt-oss-120b
export MODEL_TYPE=openai_compat

# Smoke: chained only, 2 tasks
python scripts/run_fixed_cfg_ablation_local.py \
  --modes llm_chained \
  --total-samples 20 \
  --tasks "get[wood]" "get[grass]"

# Full ablation: best-of-n first, then chained (order from config/ablation_fixed_cfg.yaml)
python scripts/run_fixed_cfg_ablation_local.py --config config/ablation_fixed_cfg.yaml
```

The script clones run-4 `grids/`, `function_specific_prompts/`, and rewrites embedded `_grid_spec_paths` so `llm_chained` / `llm_best_of_n` are scored on the identical terminal-function test grids. Program-synthesis seeds are read from run-4 outcomes (`0 5 10 … 45` by default).

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
git -C funsearch pull origin main
```

After pulling repo updates, refresh the submodule:

```bash
git pull origin laptop-ablation
git submodule update --init --remote funsearch
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
| `llm_chained` | Each step sees the **two most recent** function bodies + scores |
| `funsearch` | Full FunSearch (cluster default; needs GPU for local vLLM) |

Both ablation modes use the same evaluator and explicit-feedback stage as the main pipeline.

## Troubleshooting

- **`zsh: no matches found: get[wood]`** → quote tasks: `"get[wood]" "get[grass]"` (brackets are glob syntax in zsh)
- **`No module named 'funsearch.implementation'`** → run `git submodule update --init funsearch`
- **403 / connection errors** → connect Alliance VPN
- **Slow** → lower `total_samples` in the yaml (smoke uses 20)
- **Import errors** → `pip install -r requirements-laptop.txt` from repo root

---

## C. Lab machine (e.g. lula) — tmux (disconnect-safe)

No script changes needed. Run inside **tmux** so the job survives SSH disconnect / closing your laptop.

### One-time setup on lula

```bash
ssh lula
git clone https://github.com/Ava-creates/DSL_Generator.git
cd DSL_Generator
git checkout laptop-ablation
git submodule update --init funsearch
git pull origin laptop-ablation

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-laptop.txt

tar -xzf ablation_assets/run4_dsl1.tar.gz
test -f experiments/pipeline_hf_20260611_151047_run4_2104814/cfg/cfg_output_1.json && echo OK

cp key.txt.example key.txt   # Aleph key on first line (never commit)
export OPENAI_COMPAT_BASE_URL=https://inference.vulcan.alliancecan.ca
export OPENAI_COMPAT_MODEL=gpt-oss-120b
export MODEL_TYPE=openai_compat
export OPENAI_COMPAT_MAX_PARALLEL=4
```

### Test Aleph API (do this before a long run)

```bash
source .venv/bin/activate
export OPENAI_COMPAT_API_KEY="$(grep -v '^#' key.txt | head -1)"
python -m src.utils.openai_compat_cold_start --key-file key.txt
```

Expect `[cold_start] Model ready`. If this fails, fix network on lula before starting the ablation.

### Full ablation in tmux

```bash
ssh lula
cd ~/DSL_Generator
tmux new -s ablation

source .venv/bin/activate
export OPENAI_COMPAT_API_KEY="$(grep -v '^#' key.txt | head -1)"
export OPENAI_COMPAT_BASE_URL=https://inference.vulcan.alliancecan.ca
export OPENAI_COMPAT_MODEL=gpt-oss-120b
export MODEL_TYPE=openai_compat

python scripts/run_fixed_cfg_ablation_local.py --config config/ablation_fixed_cfg.yaml 2>&1 | tee ablation_run.log
```

**Detach** (job keeps running): `Ctrl-b` then `d` — then close laptop / disconnect SSH.

**Reattach later:**

```bash
ssh lula
tmux attach -t ablation
```

**Alternative without tmux:** `nohup python scripts/run_fixed_cfg_ablation_local.py --config config/ablation_fixed_cfg.yaml > ablation_run.log 2>&1 &`
