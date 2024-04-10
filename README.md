# Task-Vector-Merge-Optimzier

## 概要

[Sdff-Ltba/LightChatAssistant-2x7B](https://huggingface.co/Sdff-Ltba/LightChatAssistant-2x7B)で行われているようなLLMにおけるTask Vectorの加算によるマージにおいて、その加算割合の最適化をOptunaを用いて行うスクリプトです。

実際にこのスクリプトを使い作成したモデル例はこちらです。

[Aratako/LightChatAssistant-2x7B-optimized-experimental](https://huggingface.co/Aratako/LightChatAssistant-2x7B-optimized-experimental)

現在はMistral-7Bベースのモデルにのみ対応しています。（変更はそこまで難しくないので、他モデルに適用したい場合は適宜修正してください）

大まかなステップは以下の通りです。

1. ある加算割合で対象モデルにTask Vectorを加算
2. ステップ1で出来上がったモデルを用いて何らかのタスクに対する評価を行い、評価のスコアを取得
3. そのスコアを評価値とし、それを最大化するように加算割合を最適化（ベイズ最適化ベース/進化計算ベース）

現在、[lm-evaluation-harness](https://github.com/Stability-AI/lm-evaluation-harness)によるJGLUEベースの評価と、[MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)によるLLM as a judgeベースの評価による最適化に対応しています。

なお、素人の制作物なのでコードが読みづらい&バグがある可能性もありますが、ご理解ください。（特にJGLUE側はほぼ検証しておりません）

## 環境準備

以下のような手順で必要なライブラリをインストールしてください。

1. リポジトリをクローン

   ```sh
    git clone https://github.com/Aratako/Task-Vector-Merge-Optimzier.git
   ```

2. 使う評価指標に応じた環境準備
   1. MT-Benchを使う場合

        ```sh
        cd FastChat
        pip install -e ".[model_worker,llm_judge]"
        ```

   2. lm-evaluation-harness（JGLUE）を使う場合

        ```sh
        cd lm-evaluation-harness
        pip install -e ".[ja]"
        ```

3. その他ライブラリのインストール

    ```sh
    cd ..
    pip install -r requirements.txt
    ```

4. スクリプトの実行
   1. .shファイルの中身を変更し、主要なパラメータを設定（詳細は後述）
   2. .shファイルを実行（Windows向けに.batファイルは用意してありますが、完全に未検証です）

## マージスクリプトについて

現在、4種類の最適化スクリプトを用意してあります。

1. [lm-evaluation-harness](https://github.com/Stability-AI/lm-evaluation-harness)を用いた、JGLUEによる評価をもとにした最適化

    JGLUEタスクに対する評価値を最大化するように加算割合を最適化します。
   1. merge_task_vector_jglue.py
   2. run_merge_jglue.sh
   3. run_merge_jglue.bat（未検証）
2. [MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)を用いた、LLM as a judge的な評価をもとにした最適化

    MT-Benchの形式で用意した何らかのベンチマークの評価値を最大化するように加算割合を最適化します。なお、MT-Benchベースの方では、ベンチマークを行うための指示プロンプトや出力評価用のプロンプトがカスタマイズ可能であり、LLMによる評価値を採用する場合基本的にどのようなものも採用可能です。
   1. merge_task_vector_mt_bench.py
   2. run_merge_mt_bench.sh
   3. run_merge_mt_bench.bat（未検証）
3. [MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)を用いた、LLM as a judge的な評価をもとにした2モデルのMoEを前提とした最適化

    MT-Benchの形式で用意した何らかのベンチマークの評価値を最大化するように加算割合を最適化します。ただし、2モデルのMoEを行う前提で、MoEした後のモデルの出力の評価を最大化します。なお、MoEの設定はハードコードしてしまっているので、適宜修正してください。
   1. merge_task_vector_mt_bench_moe_2_experts.py
   2. run_merge_mt_bench_moe_2_experts.sh
   3. run_merge_mt_bench_moe_2_experts.bat（未検証）
4. [MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)を用いた、LLM as a judge的な評価をもとにした4モデルのMoEを前提とした最適化

    MT-Benchの形式で用意した何らかのベンチマークの評価値を最大化するように加算割合を最適化します。ただし、4モデルのMoEを行う前提で、MoEした後のモデルの出力の評価を最大化します。なお、MoEの設定はハードコードしてしまっているので、適宜修正してください。
   1. merge_task_vector_mt_bench_moe_4_experts.py
   2. run_merge_mt_bench_moe_4_experts.sh
   3. run_merge_mt_bench_moe_4_experts.bat（未検証）

## スクリプトの引数について

### 全スクリプト共通のもの

#### Task Vectorの加算に関連するパラメータ

更新後のモデルは以下のような式で表されます。この式における最適な`weight`（加算割合）を探索することになります。

$model_{new} = model_{target} + weight \times (model_{tuned} - model_{base})$

以下は、この計算に関連するパラメータ類です。

1. **--target_model**

    Task Vectorの加算対象となるモデルです。huggingface repositryを指定するか、ローカルのものを指定してください。

2. **--base_model**

    Task Vectorの算出において、引く側に使われるモデルです。huggingface repositryを指定するか、ローカルのものを指定してください。

3. **--tuned_model**

    Task Vectorの算出において、引かれる側に使われるモデルです。huggingface repositryを指定するか、ローカルのものを指定してください。

4. **--weight_min, --weight_max**

    加算割合の探索における最小値と最大値です。`weight_min`から`weight_max`の範囲を探索します。デフォルトは0と2です。

#### Optunaによる最適化に関連するパラメータ

1. **--optimize_mode**

    探索空間の設定方法です。私の方で、`all`, `layer`, `parameter`という3つを実装してあります。なお、Mistral-7Bベースのモデル以外では実装できていないので、他で試す場合は適宜修正してください。

    1. `all`: 全てのレイヤーの全ての重みへの加算割合を探索対象とします。例えばMistral-7Bベースのモデルでは、1モデルあたり291個の加算割合を探索することとなります。
    2. `layer`: 各レイヤーに対して1つずつの加算割合を探索対象とします。例えばMistral-7Bベースのモデルでは、1モデル当たり32個の`hidden_layers`とその他3つのレイヤー、計35個の加算割合を探索することとなります。
    3. `parameter`: 各パラメータに対して1つずつの加算割合を探索対象とします。例えばMistral-7Bベースのモデルでは、`down_proj`, `gate_proj`, `k_proj`などの計12個の加算割合を探索することとなります。

2. **--optuna_sampler**

    探索に使われるOptunaのサンプラーです。これにより最適化戦略が決まります。`TPE`を指定することで、TPEというベイズ最適化ベースのアルゴリズムによる探索が行われます`"CMA-ES`を指定することで、CMA-ESという進化計算ベースのアルゴリズムによる探索が行われます。デフォルトは`TPE`です。

3. **--n_trials**

    Optunaによる探索の最大試行回数です。

4. **--optuna_seed**

    Optunaのサンプラーに渡されるシード値です。これを統一すると結果の再現性が得られます。

#### その他のパラメータ

1. **--cache_dir**

    huggingfaceからモデルをダウンロードする際にキャッシュとして使われるディレクトリです。

2. **--output_dir**

    最後にベストスコアだったモデルを作成し保存する際の保存先のフォルダです。なお、MoEを行うスクリプトでは、MoE後のモデルではなくベースとなる各モデルを保存します。

### JGLUEによる評価に関連したパラメータ

JGLUEの評価指標に使われるタスクはあらかじめ決まっているものなので、基本的には利用したいタスクを指定するだけです。

1. **--jglue_tasks**

    評価したいJGLUEタスクをカンマ区切りの文字列で指定します。
    対応タスクの一覧については[lm-evaluation-harness](https://github.com/Stability-AI/lm-evaluation-harness)のREADMEを確認してください。

2. **--jglue_limit**

    --jglue_tasksで指定した各タスクについて、ここで指定した数値分のみを評価対象として評価を行います。

### MT-Benchによる評価に関連したパラメータ

[MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)における評価は、自作プロンプトに対して推論→それをGPT-4などを使って評価→そのスコアの平均値を評価値とするという流れなので、かなり自由度が高いです。ベンチマーク時に使う質問のリストや評価プロンプトをカスタマイズ可能であり、特定のタスク（例えばロールプレイなど）に特化して評価を行うことが出来ます。

1. **--bench_name**

    実際に実行するベンチマークの名前です。デフォルトのmt-benchも使えますが、基本的には質問含め自作してください。

    `FastChat/fastchat/llm-judge/data/{bench_name}`フォルダにベンチマーク時にプロンプトとして利用したいもののリストを`question.jsonl`という名前で格納してください。フォーマットなどは[こちら](https://github.com/Aratako/Task-Vector-Merge-Optimzier/blob/main/FastChat/fastchat/llm_judge/data/example/question.jsonl)の例を参考にしてください。

    ここで指定されたbench_nameをもとに、"FastChat/fastchat/llm-judge/data/{bench_name}"においてベンチマーク時の出力の保存や評価が行われます。

2. **--judge_model**

    モデルの出力を評価するLLMです。GPT-4などの高性能なLLMに出力を評価させることで、絶対的な答えのないStory Writingなどのタスクも評価することが出来ます。

    デフォルトは`gpt-4`です。この場合、OpenAIのAPIキーの設定が必要なので以下のように設定してください。（Azure OpenAIも利用可能だったと記憶しています）

    ```sh
    export OPENAI_API_KEY=***
    ```

    その他、`gpt-3.5`やAnthropicの`Claude-2`などにデフォルトで対応しています。また、`Cohere`を指定すると、`Command-R Plus`の4bit量子化モデルによる評価が行えるように改変してあります。

3. **--judge_file**

    上記の`judge_model`によって出力を評価する際に使われるプロンプトを記述したファイルです。デフォルトは`FastChat/fastchat/llm_judge/data/judge_prompts.jsonl`となっており、この中の`single-v1`と`single-v1-multi-turn`が使われます。

    この部分のプロンプトを変えることで、評価方法をカスタマイズできます。私の変更例では、日本語以外の返答が行われた場合評価を下げるように指定しています。

4. **--parallel**

    上記のLLMによる評価を並列で行う数です。デフォルトのまま`parallel=2`とすると、GPT-4による評価が並列で2つ走ります。
