#!/usr/bin/env bash
# Pack minimal artifacts from a completed pipeline run for laptop fixed-CFG ablation.
set -euo pipefail
SOURCE="${1:-experiments/pipeline_hf_20260611_151047_run4_2104814}"
DSL_ROUND="${2:-1}"
OUT="${3:-ablation_assets/run4_dsl1.tar.gz}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
tar -czf "$OUT" \
  "$SOURCE/cfg/cfg_output_${DSL_ROUND}.json" \
  "$SOURCE/status/dsl${DSL_ROUND}/file_generation/status" \
  $(find "$SOURCE/function_specific_prompts" -name "*dsl${DSL_ROUND}*" 2>/dev/null) \
  $(find "$SOURCE/functions_generated" -name "*dsl${DSL_ROUND}*" 2>/dev/null) \
  "$SOURCE/grids" 2>/dev/null || true
echo "Wrote $OUT"
echo "On laptop: mkdir -p experiments/run4_dsl1_src && tar -xzf run4_dsl1.tar.gz -C experiments/run4_dsl1_src --strip-components=0"
