#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Launcher for the data-augmentation pipeline.
# Edit the values below, then run:   bash run.sh
# It forwards these parameters to run_pipeline.py, which always runs steps 1->5.
# Intermediate and output files are written to the current working directory,
# so run this from the folder that holds your dataset.
#
# Before the first run: implement call_llm() inside run_pipeline.py.
# For finer parameters (cluster count, readability gates, prompts, ...),
# edit the CONFIGURATION block in run_pipeline.py.
# -----------------------------------------------------------------------------
set -euo pipefail

# ---- Interpreter / script location ----
PYTHON="${PYTHON:-python3}"
SCRIPT="${SCRIPT:-run_pipeline.py}"

# ---- Dataset ----
INPUT="train.json"                   # original dataset (a JSON list)

# ---- LLM ----
# Secrets are passed through the environment (never on the command line).
export LLM_API_KEY=""                # your API key
export LLM_API_BASE=""               # custom endpoint; leave empty for the provider default
LLM_MODEL=""                         # model name, e.g. "gpt-4o" or "Qwen/Qwen3-235B-A22B"

# ---- Embeddings (step 4) ----
EMBEDDING_MODEL="sbert_model"   # use a multilingual model for non-English text

# ---- Main knobs ----
SEED=42
GEN_SCALE=                       # generated items = original count * this
RARE_THRESHOLD=                   # a relation is "rare" if frequency < this
TARGET_COMMON_RATIO=              # down-sample common relations to ratio * items
RARE_RELS_PER_ITEM=
SAVE_INTERVAL=10
REBUILD_REFERENCES=0                 # set to 1 to rebuild the step-4 reference pool

# ---- Build the command and run ----
ARGS=(
  --input "$INPUT"
  --llm-model "$LLM_MODEL"
  --embedding-model "$EMBEDDING_MODEL"
  --seed "$SEED"
  --gen-scale "$GEN_SCALE"
  --rare-threshold "$RARE_THRESHOLD"
  --target-common-ratio "$TARGET_COMMON_RATIO"
  --rare-rels-per-item "$RARE_RELS_PER_ITEM"
  --save-interval "$SAVE_INTERVAL"
)
if [ "$REBUILD_REFERENCES" = "1" ]; then
  ARGS+=( --rebuild-references )
fi

"$PYTHON" "$SCRIPT" "${ARGS[@]}"