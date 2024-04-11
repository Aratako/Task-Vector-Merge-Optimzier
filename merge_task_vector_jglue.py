import argparse
import copy
import gc
import re

import numpy as np
import optuna
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from lm_evaluation_harness.scripts.main_eval import main


# 評価関数
def evaluate(model, tokenizer, config):
    eval_args = {
        "model": "hf-causal-experimental",
        "model_args": [model, tokenizer, config],
        "tasks": jglue_tasks,
        "num_fewshot": [3, 3, 3, 3],
        "batch_size": 1,
        "device": "cuda",
        "limit": jglue_limit,
        "no_cache": True,
    }
    results = main(eval_args, None, None)
    score = 0
    for task in jglue_tasks:
        task_result = results["results"][task]
        if "jsquad" in task:
            score += task_result["exact_match"] / 100
        else:
            score += task_result["acc"]
    return score / len(jglue_tasks)


def update_model_parameters(
    model, task_vectors, weights, num_params, optimize_mode, unique_params=None
):
    if optimize_mode == "all":
        for i, (k, v) in enumerate(model.state_dict().items()):
            new_v = v + (weights[i] * task_vectors[k].to(v.device))
            v.copy_(new_v)
    elif optimize_mode == "layer":
        for k, v in model.state_dict().items():
            if k == "model.embed_tokens.weight":
                new_v = v + (weights[0] * task_vectors[k].to(v.device))
                v.copy_(new_v)
            elif k == "model.norm.weight":
                new_v = v + (weights[num_params - 2] * task_vectors[k].to(v.device))
                v.copy_(new_v)
            elif k == "lm_head.weight":
                new_v = v + (weights[num_params - 1] * task_vectors[k].to(v.device))
                v.copy_(new_v)
            else:
                layer_index = int(re.findall(r"\d+", k)[0])
                new_v = v + (weights[layer_index + 1] * task_vectors[k].to(v.device))
                v.copy_(new_v)
    elif optimize_mode == "parameter":
        for i, (k, v) in enumerate(model.state_dict().items()):
            param_name = k.split(".weight")[0]
            param_name = param_name.replace("model.", "")
            param_name = (
                param_name.split(".")[-1] if "layers" in param_name else param_name
            )
            param_index = unique_params.index(param_name)
            new_v = v + (weights[param_index] * task_vectors[k].to(v.device))
            v.copy_(new_v)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--target_model", type=str, required=True, help="マージ対象のモデル"
)
parser.add_argument(
    "--base_model", type=str, required=True, help="ベクトル計算の元となるベースモデル"
)
parser.add_argument(
    "--tuned_model",
    type=str,
    required=True,
    help="ベクトル計算の元となるチューニング済みモデル",
)
parser.add_argument(
    "--optimize_mode",
    type=str,
    choices=["all", "layer", "parameter"],
    default="layer",
    help="探索のモード",
)
parser.add_argument(
    "--n_trials", type=int, default=30, help="Optunaによる探索の試行回数"
)
parser.add_argument(
    "--cache_dir",
    type=str,
    default="./models",
    help="モデルやトークナイザーをキャッシュするディレクトリのパス",
)
parser.add_argument(
    "--optuna_sampler",
    type=str,
    choices=["CMA-ES", "TPE"],
    default="TPE",
    help="Optunaのサンプラー",
)
parser.add_argument(
    "--weight_min", type=float, default=0, help="加算割合の探索範囲の最小値"
)
parser.add_argument(
    "--weight_max", type=float, default=2, help="加算割合の探索範囲の最大値"
)
parser.add_argument(
    "--output_dir", type=str, default="./merged_model", help="最終モデルの保存先"
)
parser.add_argument(
    "--optuna_seed", type=int, default=42, help="Optunaのサンプラーのシード値"
)
parser.add_argument(
    "--jglue_tasks",
    type=str,
    default="jcommonsenseqa-1.1-0.3,marc_ja-1.1-0.2,jnli-1.3-0.2,jsquad-1.1-0.2",
    help="評価するJGLUEタスクのリスト（カンマ区切り）",
)
parser.add_argument(
    "--jglue_limit",
    type=str,
    default="500,500,500,500",
    help="各JGLUEタスクの評価数のリスト（カンマ区切り）",
)

args = parser.parse_args()

target_model = args.target_model
base_model = args.base_model
tuned_model = args.tuned_model
optimize_mode = args.optimize_mode
n_trials = args.n_trials
cache_dir = args.cache_dir
optuna_sampler = args.optuna_sampler
weight_min = args.weight_min
weight_max = args.weight_max
output_dir = args.output_dir
optuna_seed = args.optuna_seed
jglue_tasks = args.jglue_tasks.split(",")
jglue_limit = [int(limit) for limit in args.jglue_limit.split(",")]

base_model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    cache_dir=cache_dir,
)
tuned_model = AutoModelForCausalLM.from_pretrained(
    tuned_model,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    cache_dir=cache_dir,
)
task_vectors = {
    k: tuned_model.state_dict()[k] - base_model.state_dict()[k]
    for k in base_model.state_dict()
}

# 不要になるので削除し、メモリを解放
del base_model
del tuned_model
gc.collect()

# モデルのロード
model = AutoModelForCausalLM.from_pretrained(
    target_model,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    cache_dir=cache_dir,
)
tokenizer = AutoTokenizer.from_pretrained(
    target_model,
    cache_dir=cache_dir,
)
config = AutoConfig.from_pretrained(target_model)
unique_params = None
if optimize_mode == "all":
    num_params = len(model.state_dict().items())
elif optimize_mode == "layer":
    num_params = model.config.num_hidden_layers + 3  # for mistral
elif optimize_mode == "parameter":
    param_list = []
    for k, v in model.state_dict().items():
        param_name = k.split(".weight")[0]
        param_name = param_name.replace("model.", "")
        param_name = param_name.split(".")[-1] if "layers" in param_name else param_name
        param_list.append(param_name)
    unique_params = list(set(param_list))
    num_params = len(unique_params)


# 最適化する関数
def objective(trial):
    model.load_state_dict(original_model_state)

    weights = [
        trial.suggest_float(f"weight_{i}", weight_min, weight_max)
        for i in range(num_params)
    ]

    update_model_parameters(
        model,
        task_vectors,
        weights,
        num_params,
        optimize_mode,
        unique_params,
    )

    # マージしたモデルでベンチマークのプロンプトに対し推論
    # 高速化のため一度GPUに移動
    model.to("cuda")
    score = evaluate(
        model, tokenizer, config
    )  # この関数はユーザーが定義する必要があります。
    model.to("cpu")

    return score


# 最適化プロセスの実行
if optuna_sampler == "CMA-ES":
    sampler = optuna.samplers.CmaEsSampler(seed=optuna_seed)
elif optuna_sampler == "TPE":
    sampler = optuna.samplers.TPESampler(seed=optuna_seed)
study = optuna.create_study(direction="maximize", sampler=sampler)

original_model_state = copy.deepcopy(model.state_dict())
study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

# 最適化された比率を取得
optimal_weights = [study.best_params[f"weight_{i}"] for i in range(num_params)]
# この時点でのmodelは最後のtrialの加算の影響を受けているので、一度最初の状態に戻す
model.load_state_dict(original_model_state)

# 最適な比率でモデルをマージ
update_model_parameters(
    model,
    task_vectors,
    optimal_weights,
    num_params,
    optimize_mode,
    unique_params,
)

# メモリ不足で保存できない場合があるため、task_vectorsとoriginal_model_stateを削除しておく
del task_vectors
del original_model_state
gc.collect()

# マージされたモデルを保存
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
