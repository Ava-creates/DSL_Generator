# Laptop ablation (no SLURM)

Run the **baseline ablation** on your laptop: compare `llm_chained` vs `llm_best_of_n` for generating per-task Python functions, using the **Aleph API** for LLM calls. No GPU, no cluster queue.

This is **not** the full DSL pipeline (that stays on Vulcan). It answers: *without a DSL, can simple LLM search + explicit feedback solve Craft tasks?*

## What runs

For each of 20 Craft tasks (or a subset):

1. Build a task-specific prompt (task_env — no testcase grids, no GPU)
2. Generate candidates with `llm_chained` or `llm_best_of_n` (100 samples default)
3. Run explicit feedback (30 iterations)
4. Evaluate final functions on 10 seeds → `results_tracking/baseline_final_eval.json`

## Setup

```bash
git clone git@github.com:Ava-creates/DSL_Generator.git
cd DSL_Generator
git checkout laptop-ablation

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

- **403 / connection errors** → connect Alliance VPN
- **Slow** → lower `total_samples` in the yaml (smoke uses 20)
- **Import errors** → `pip install -r requirements-laptop.txt` from repo root
