@echo off

set TARGET_MODEL=NTQAI/chatntq-ja-7b-v1.0
set BASE_MODEL=mistralai/Mistral-7B-v0.1
set TUNED_MODEL=mistralai/Mistral-7B-Instruct-v0.2
set OPTIMIZE_MODE=layer
set N_TRIALS=30
set CACHE_DIR=.\models
set OPTUNA_SAMPLER=TPE
set WEIGHT_MIN=0
set WEIGHT_MAX=2
set OUTPUT_DIR=.\model-chatvector
set OPTUNA_SEED=42
set JGLUE_TASKS=jcommonsenseqa-1.1-0.3,marc_ja-1.1-0.2,jnli-1.3-0.2,jsquad-1.1-0.2
set JGLUE_LIMIT=500,500,500,500

set PYTHON_SCRIPT_PATH=.\merge_task_vector_jglue.py

cd /d "%~dp0"
python "%PYTHON_SCRIPT_PATH%" ^
--target_model "%TARGET_MODEL%" ^
--base_model "%BASE_MODEL%" ^
--tuned_model "%TUNED_MODEL%" ^
--optimize_mode "%OPTIMIZE_MODE%" ^
--n_trials "%N_TRIALS%" ^
--cache_dir "%CACHE_DIR%" ^
--optuna_sampler "%OPTUNA_SAMPLER%" ^
--weight_min "%WEIGHT_MIN%" ^
--weight_max "%WEIGHT_MAX%" ^
--output_dir "%OUTPUT_DIR%" ^
--optuna_seed "%OPTUNA_SEED%" ^
--jglue_tasks "%JGLUE_TASKS%" ^
--jglue_limit "%JGLUE_LIMIT%"
