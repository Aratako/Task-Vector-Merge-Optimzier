#!/bin/bash

# Pythonスクリプトに渡す引数を変数として定義
TARGET_MODEL="NTQAI/chatntq-ja-7b-v1.0"
BASE_MODEL="mistralai/Mistral-7B-v0.1"
TUNED_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
OPTIMIZE_MODE="layer"
N_TRIALS=30
CACHE_DIR="./models"
OPTUNA_SAMPLER="TPE"
WEIGHT_MIN=0
WEIGHT_MAX=2
OUTPUT_DIR="./model-chatvector"
OPTUNA_SEED=42
JGLUE_TASKS="jcommonsenseqa-1.1-0.3,marc_ja-1.1-0.2,jnli-1.3-0.2,jsquad-1.1-0.2"
JGLUE_LIMIT="500,500,500,500"

# Pythonスクリプトのパス
PYTHON_SCRIPT_PATH="./merge_task_vector_jglue.py"

# Pythonスクリプトを実行
/usr/bin/env python "$PYTHON_SCRIPT_PATH" \
  --target_model "$TARGET_MODEL" \
  --base_model "$BASE_MODEL" \
  --tuned_model "$TUNED_MODEL" \
  --optimize_mode "$OPTIMIZE_MODE" \
  --n_trials "$N_TRIALS" \
  --cache_dir "$CACHE_DIR" \
  --optuna_sampler "$OPTUNA_SAMPLER" \
  --weight_min "$WEIGHT_MIN" \
  --weight_max "$WEIGHT_MAX" \
  --output_dir "$OUTPUT_DIR" \
  --optuna_seed "$OPTUNA_SEED" \
  --jglue_tasks "$JGLUE_TASKS" \
  --jglue_limit "$JGLUE_LIMIT"
