@echo off

set TARGET_MODEL_1=NTQAI/chatntq-ja-7b-v1.0
set TARGET_MODEL_2=Elizezen/Antler-7B
set BASE_MODEL=mistralai/Mistral-7B-v0.1
set TUNED_MODEL=mistralai/Mistral-7B-Instruct-v0.2
set BENCH_NAME=example
set JUDGE_MODEL_NAME=gpt-4
set OPTIMIZE_MODE=layer
set N_TRIALS=30
set CACHE_DIR=.\models
set JUDGE_FILE=FastChat\fastchat\llm_judge\data\judge_prompts.jsonl
set PARALLEL=2
set OPTUNA_SAMPLER=TPE
set WEIGHT_MIN=0
set WEIGHT_MAX=2
set OUTPUT_DIR=.\model-chatvector
set OPTUNA_SEED=42

set PYTHON_SCRIPT_PATH=.\merge_task_vector_mt_bench_moe_2_experts.py

cd /d "%~dp0"

python "%PYTHON_SCRIPT_PATH%" ^
--target_model_1 "%TARGET_MODEL_1%" ^
--target_model_2 "%TARGET_MODEL_2%" ^
--base_model "%BASE_MODEL%" ^
--tuned_model "%TUNED_MODEL%" ^
--bench_name "%BENCH_NAME%" ^
--judge_model_name "%JUDGE_MODEL_NAME%" ^
--optimize_mode "%OPTIMIZE_MODE%" ^
--n_trials "%N_TRIALS%" ^
--cache_dir "%CACHE_DIR%" ^
--judge_file "%JUDGE_FILE%" ^
--parallel "%PARALLEL%" ^
--optuna_sampler "%OPTUNA_SAMPLER%" ^
--weight_min "%WEIGHT_MIN%" ^
--weight_max "%WEIGHT_MAX%" ^
--output_dir "%OUTPUT_DIR%" ^
--optuna_seed "%OPTUNA_SEED%"
