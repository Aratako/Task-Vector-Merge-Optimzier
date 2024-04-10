import argparse
import copy
import gc
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import optuna
import pandas as pd
import torch
import yaml
from click.testing import CliRunner
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from FastChat.fastchat.llm_judge.gen_model_answer import (
    run_eval_from_model_and_tokenizer,
)
from fastchat.llm_judge.common import (
    NEED_REF_CATS,
    check_data,
    load_judge_prompts,
    load_model_answers,
    load_questions,
    play_a_match_single,
)
from fastchat.llm_judge.gen_judgment import make_judge_single, make_match_single
from mergekit.mergekit.scripts.mixtral_moe import main


def create_moe_config_yaml(save_path):
    # YAMLファイルに保存するデータ構造
    data = {
        "base_model": "./temp_model_1",
        "gate_mode": "random",
        "dtype": "bfloat16",
        "experts": [
            {
                "source_model": "./temp_model_1",
                "positive_prompts": [
                    "",
                ],
                "negative_prompts": [""],
            },
            {
                "source_model": "./temp_model_2",
                "positive_prompts": [
                    "",
                ],
            },
        ],
        "tokenizer_source": "union",
    }

    # YAMLファイルを指定されたパスに保存
    with open(save_path, "w", encoding="utf-8") as file:
        yaml.dump(data, file, allow_unicode=True, sort_keys=False)


def moe_by_mergekit():
    # moe configを作成して保存
    create_moe_config_yaml(save_path="./config.yaml")
    # 上記のconfigを利用してMoEを実行。出来たモデルはout_pathに保存される
    runner = CliRunner()

    # コマンドライン引数をシミュレート
    result = runner.invoke(
        main,
        [
            "./config.yaml",
            "./moe_model_temp",
            "--device",
            "cuda",
            "--verbose",
            "--lazy-unpickle",
        ],
    )
    if result.exception:
        print(result.exception)


def create_responses(trial_number, model, tokenizer):
    model_id = "temp_model_mistral_" + str(trial_number)
    answer_file = (
        f"FastChat/fastchat/llm_judge/data/{bench_name}/model_answer/{model_id}.jsonl"
    )
    run_eval_from_model_and_tokenizer(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        question_file=question_file,
        question_begin=None,
        question_end=None,
        answer_file=answer_file,
        max_new_token=512,
        num_choices=1,
        num_gpus_per_model=1,
        num_gpus_total=1,
    )


def get_evaluate_result(trial_number):
    score_df = pd.read_json(
        f"FastChat/fastchat/llm_judge/data/{bench_name}/model_judgment/{judge_model_name}_single.jsonl",
        orient="records",
        lines=True,
    )
    model_id = "temp_model_mistral_" + str(trial_number)
    score_df = score_df[score_df["model"] == model_id]
    score = score_df["score"].mean()
    return score


def evaluate_and_save_to_json(
    judge_model, judge_tokenizer, models, mode="single", baseline_model=None, parallel=1
):
    # Load questions
    questions = load_questions(question_file, None, None)

    # Load answers
    model_answers = load_model_answers(answer_dir)
    ref_answers = load_model_answers(ref_answer_dir)

    # Load judge
    judge_prompts = load_judge_prompts(judge_file)

    if mode == "single":
        judges = make_judge_single(judge_model_name, judge_prompts)
        play_a_match_func = play_a_match_single
        output_file = f"FastChat/fastchat/llm_judge/data/{bench_name}/model_judgment/{judge_model_name}_single.jsonl"
        make_match_func = make_match_single
        baseline_model = None
    else:
        # ここは未実装です。
        raise ValueError

    check_data(questions, model_answers, ref_answers, models, judges)

    question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
    question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

    # Make matches
    matches = []
    matches += make_match_func(
        question_default, models, model_answers, judges["default"], baseline_model
    )
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math"],
        baseline_model,
        ref_answers,
    )
    matches += make_match_func(
        question_default,
        models,
        model_answers,
        judges["default-mt"],
        baseline_model,
        multi_turn=True,
    )
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math-mt"],
        baseline_model,
        ref_answers,
        multi_turn=True,
    )

    match_stat = {}
    match_stat["bench_name"] = bench_name
    match_stat["mode"] = mode
    match_stat["judge"] = judge_model_name
    match_stat["baseline"] = baseline_model
    match_stat["model_list"] = models
    match_stat["total_num_questions"] = len(questions)
    match_stat["total_num_matches"] = len(matches)
    match_stat["output_path"] = output_file

    # Play matches
    if parallel == 1:
        for match in tqdm(matches):
            play_a_match_func(
                match,
                output_file=output_file,
                judge_model=judge_model,
                judge_tokenizer=judge_tokenizer,
            )
    else:

        def play_a_match_wrapper(match):
            play_a_match_func(match, output_file=output_file)

        np.random.seed(0)
        np.random.shuffle(matches)

        with ThreadPoolExecutor(parallel) as executor:
            for match in tqdm(
                executor.map(play_a_match_wrapper, matches), total=len(matches)
            ):
                pass


# 評価関数
def evaluate(trial_number):
    model_id = "temp_model_mistral_" + str(trial_number)
    models = [model_id]
    evaluate_and_save_to_json(
        judge_model,
        judge_tokenizer,
        models,
        mode="single",
        baseline_model=None,
        parallel=parallel,
    )
    score = get_evaluate_result(trial_number)
    return score


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
    "--target_model_1", type=str, required=True, help="マージ対象のモデル1つ目"
)
parser.add_argument(
    "--target_model_2", type=str, required=True, help="マージ対象のモデル2つ目"
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
parser.add_argument("--bench_name", type=str, required=True, help="ベンチマーク名")
parser.add_argument(
    "--judge_model_name", type=str, default="gpt-4", help="評価モデル名"
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
    "--judge_file",
    type=str,
    default="FastChat/fastchat/llm_judge/data/judge_prompts.jsonl",
    help="評価用のプロンプトが含まれるファイルのパス",
)
parser.add_argument("--parallel", type=int, default=1, help="評価を並列で実行する数")
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

args = parser.parse_args()

target_model_1 = args.target_model_1
target_model_2 = args.target_model_2
base_model = args.base_model
tuned_model = args.tuned_model
bench_name = args.bench_name
judge_model_name = args.judge_model_name
optimize_mode = args.optimize_mode
n_trials = args.n_trials
cache_dir = args.cache_dir
judge_file = args.judge_file
parallel = args.parallel
optuna_sampler = args.optuna_sampler
weight_min = args.weight_min
weight_max = args.weight_max
output_dir = args.output_dir
optuna_seed = args.optuna_seed

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
model_1 = AutoModelForCausalLM.from_pretrained(
    target_model_1,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    cache_dir=cache_dir,
)
tokenizer_1 = AutoTokenizer.from_pretrained(
    target_model_1,
    cache_dir=cache_dir,
)
model_2 = AutoModelForCausalLM.from_pretrained(
    target_model_2,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    cache_dir=cache_dir,
)
tokenizer_2 = AutoTokenizer.from_pretrained(
    target_model_2,
    cache_dir=cache_dir,
)
original_model_state_1 = copy.deepcopy(model_1.state_dict())
original_model_state_2 = copy.deepcopy(model_2.state_dict())
unique_params = None
if optimize_mode == "all":
    num_params = len(model_1.state_dict().items())
elif optimize_mode == "layer":
    num_params = model_1.config.num_hidden_layers + 3  # for mistral
elif optimize_mode == "parameter":
    param_list = []
    for k, v in model_1.state_dict().items():
        param_name = k.split(".weight")[0]
        param_name = param_name.replace("model.", "")
        param_name = param_name.split(".")[-1] if "layers" in param_name else param_name
        param_list.append(param_name)
    unique_params = list(set(param_list))
    num_params = len(unique_params)


if judge_model_name == "cohere":
    judge_model = AutoModelForCausalLM.from_pretrained(
        "CohereForAI/c4ai-command-r-plus-4bit",
        cache_dir="./models",
        device_map="cuda",
    )
    judge_tokenizer = AutoTokenizer.from_pretrained(
        "CohereForAI/c4ai-command-r-plus",
        cache_dir="./models",
        device_map="cuda",
    )
else:
    judge_model = None
    judge_tokenizer = None
question_file = f"FastChat/fastchat/llm_judge/data/{bench_name}/question.jsonl"
answer_dir = f"FastChat/fastchat/llm_judge/data/{bench_name}/model_answer"
ref_answer_dir = f"FastChat/fastchat/llm_judge/data/{bench_name}/reference_answer"


# 最適化する関数
def objective(trial):
    model_1.load_state_dict(original_model_state_1)
    model_2.load_state_dict(original_model_state_2)

    weights = [
        trial.suggest_float(f"weight_{i}", weight_min, weight_max)
        for i in range(2 * num_params)
    ]

    update_model_parameters(
        model_1,
        task_vectors,
        weights[:num_params],
        num_params,
        optimize_mode,
        unique_params,
    )
    update_model_parameters(
        model_2,
        task_vectors,
        weights[num_params:],
        num_params,
        optimize_mode,
        unique_params,
    )

    # それぞれのモデルを一度保存する
    model_1.save_pretrained("./temp_model_1")
    tokenizer_1.save_pretrained("./temp_model_1")
    model_2.save_pretrained("./temp_model_2")
    tokenizer_2.save_pretrained("./temp_model_2")
    # mergekitによるMoE
    moe_by_mergekit()
    moe_model = AutoModelForCausalLM.from_pretrained(
        "./moe_model_temp",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    moe_tokenizer = AutoTokenizer.from_pretrained(
        "./moe_model_temp",
    )
    create_responses(trial.number, moe_model, moe_tokenizer)
    del moe_model
    torch.cuda.empty_cache()
    # モデルの評価
    score = evaluate(trial.number)  # この関数はユーザーが定義する必要があります。

    return score


# 最適化プロセスの実行
if optuna_sampler == "CMA-ES":
    sampler = optuna.samplers.CmaEsSampler(seed=optuna_seed)
elif optuna_sampler == "TPE":
    sampler = optuna.samplers.TPESampler(seed=optuna_seed)
study = optuna.create_study(direction="maximize", sampler=sampler)

study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

# 最適化された比率を取得
optimal_weights = [study.best_params[f"weight_{i}"] for i in range(2 * num_params)]
# この時点でのmodelは最後のtrialの加算の影響を受けているので、一度最初の状態に戻す
model_1.load_state_dict(original_model_state_1)
model_2.load_state_dict(original_model_state_2)

# 最適な比率でモデルをマージ
update_model_parameters(
    model_1,
    task_vectors,
    optimal_weights[:num_params],
    num_params,
    optimize_mode,
    unique_params,
)
update_model_parameters(
    model_2,
    task_vectors,
    optimal_weights[num_params:],
    num_params,
    optimize_mode,
    unique_params,
)

# メモリ不足で保存できない場合があるため、task_vectorsとoriginal_model_stateを削除しておく
del task_vectors
del original_model_state_1
del original_model_state_2
gc.collect()

# マージされたモデルを保存
model_1.save_pretrained(f"{output_dir}-1")
tokenizer_1.save_pretrained(f"{output_dir}-1")
model_2.save_pretrained(f"{output_dir}-2")
tokenizer_2.save_pretrained(f"{output_dir}-2")

eval_output = f"FastChat/fastchat/llm_judge/data/{bench_name}/model_judgment/{judge_model_name}_single.jsonl"
if os.path.exists(eval_output):
    os.remove(eval_output)
if os.path.exists(answer_dir) and os.path.isdir(answer_dir):
    shutil.rmtree(answer_dir)
