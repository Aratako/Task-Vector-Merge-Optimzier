#!/bin/bash

# Pythonスクリプトに渡す引数を変数として定義
TARGET_MODEL="NTQAI/chatntq-ja-7b-v1.0"
BASE_MODEL="mistralai/Mistral-7B-v0.1"
TUNED_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
BENCH_NAME="example"
JUDGE_MODEL_NAME="gpt-4"
OPTIMIZE_MODE="layer"
N_TRIALS=30
CACHE_DIR="./models"
JUDGE_FILE="FastChat/fastchat/llm_judge/data/judge_prompts.jsonl"
PARALLEL=2
OPTUNA_SAMPLER="TPE"
WEIGHT_MIN=0
WEIGHT_MAX=2
OUTPUT_DIR="./model-chatvector"
OPTUNA_SEED=42

# Pythonスクリプトのパス
PYTHON_SCRIPT_PATH="./merge_task_vector_mt_bench.py"

# Pythonスクリプトを実行
/usr/bin/env python "$PYTHON_SCRIPT_PATH" \
  --target_model "$TARGET_MODEL" \
  --base_model "$BASE_MODEL" \
  --tuned_model "$TUNED_MODEL" \
  --bench_name "$BENCH_NAME" \
  --judge_model_name "$JUDGE_MODEL_NAME" \
  --optimize_mode "$OPTIMIZE_MODE" \
  --n_trials "$N_TRIALS" \
  --cache_dir "$CACHE_DIR" \
  --judge_file "$JUDGE_FILE" \
  --parallel "$PARALLEL" \
  --optuna_sampler "$OPTUNA_SAMPLER" \
  --weight_min "$WEIGHT_MIN" \
  --weight_max "$WEIGHT_MAX" \
  --output_dir "$OUTPUT_DIR" \
  --optuna_seed "$OPTUNA_SEED"
