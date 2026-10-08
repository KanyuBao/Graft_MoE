# GRAFT-MoE

GRAFT-MoE 是用于 OLMoE 和 Qwen MoE 的专家扩容、继续预训练与评估实现。流程包括模型下载、DCLM 数据准备、专家描述符采集、功能分组、专家复制、预热、训练、最终评估和结果汇总。配置集中在 `protocol.json`，所有正式入口均位于本目录。

## 目录

- [环境与资源](#环境与资源)
- [文件与配置](#文件与配置)
- [下载模型与准备数据](#下载模型与准备数据)
- [描述符分组与预热](#描述符分组与预热)
- [训练方法与运行命令](#训练方法与运行命令)
- [进度恢复与失败处理](#进度恢复与失败处理)
- [保存重载与单独评估](#保存重载与单独评估)
- [评估协议与结果汇总](#评估协议与结果汇总)
- [验证范围](#验证范围)

## 环境与资源

在 Linux、Python 3.10 或 3.11 环境下运行，使用支持 BF16 的 NVIDIA GPU。训练采用单卡 BF16 模型、CPU FP32 主参数与 Adafactor 优化器，并启用梯度检查点。CPU 优化器会占用主机内存并产生 CPU/GPU 数据传输开销。

在本 README 所在目录执行所有命令。已有可用的 PyTorch 2.3.0 / CUDA 12.1 环境时，直接安装其余依赖：

```bash
python -m pip install -r requirements.txt -c constraints.txt
```

新环境可按以下顺序安装；系统 NVIDIA 驱动仍需与 CUDA 运行时兼容：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt -c constraints.txt
nvidia-smi
```

`requirements.txt` 固定了 Transformers 4.57.1、Datasets 3.1.0 和 lm-eval 0.4.9.1 等版本。不要在同一实验根目录中更换依赖、任务定义、代码或协议后继续旧实验。

资源规划需要同时考虑以下三项：

| 资源 | 运行要求 |
| --- | --- |
| GPU 显存 | 每个任务的完整模型、扩容专家、梯度和激活必须装入一张 GPU。多卡不会合并显存。`continue` 训练全部参数，资源需求与只训练新增专家的方法不同。 |
| CPU RAM | 需要容纳 FP32 主参数、优化器状态与分组更新时的临时张量；并行两个任务会叠加内存与带宽需求。 |
| 磁盘 | 需要保存原始模型、HF 缓存、原文池、各 tokenizer 的数据、描述符、断点和最终权重。断点提交期间新旧状态会短暂共存；全量矩阵还会保留每个训练任务的最终权重。 |

完成输入准备后运行 `run_graftmoe.py --check`，可查看所选模型和方法的 GPU、RAM、磁盘估算。估算用于队列准入，不等同于实际峰值保证；实际资源取决于模型配置、环境与并行负载。默认另保留 `disk_reserve_gib` 指定的空闲空间。

`--gpus 4 5 --max-parallel 2` 表示在物理 GPU 4、5 上最多同时运行两个独立任务，每张卡一个任务。它不是 DDP、模型并行或多卡联合训练；每个子进程内部使用 `cuda:0`。调度器同时检查空闲显存、主机内存和磁盘空间，资源不足时等待，超过 `resource_wait_minutes` 后退出并保留已有进度。

## 文件与配置

| 文件 | 用途 |
| --- | --- |
| `protocol.json` | 模型路径、数据路径、输出目录、种子、训练和评估参数 |
| `project_config.py` | 统一解析协议及相对路径 |
| `download_models.py` | 下载官方模型并校验本地模型 |
| `prepare_data.py` | 准备共享原文池、分别分词并检查 DCLM 数据 |
| `collect_descriptors.py` | 在训练校准子集上采集专家描述符 |
| `prepare_eval_data.py`、`task_data.py` | 下载下游任务数据并验证离线缓存与任务指纹 |
| `group_utils.py` | 基于描述符生成专家分组、分配和复制方案 |
| `run_graftmoe.py` | 输入检查、共享准备、资源调度、训练与自动汇总 |
| `graftmoe_core.py`、`baseline_utils.py`、`confidence_routing.py` | 模型扩容、训练、路由和指标实现 |
| `status_graftmoe.py` | 只读查看任务完成度 |
| `load_graftmoe_checkpoint.py` | 校验并重建已保存模型 |
| `evaluate_graftmoe.py` | 对保存的模型单独评估 |
| `summarize_graftmoe.py` | 汇总已完成结果、生成表格与恢复曲线 |

先检查 `protocol.json` 中的以下字段，将大文件目录放在有足够容量的磁盘上：

| 配置项 | 默认目录 | 含义 |
| --- | --- | --- |
| `models.<model>.path` | `models/<model>/` | 原始模型、tokenizer 与下载记录目录 |
| `models.<model>.data` | `data/processed/<model>/` | 该模型 tokenizer 对应的 `train.npy`、`valid.npy`、`test.npy`、`calibration.npy` 和 `report.json` |
| `models.<model>.descriptors` | `artifacts/descriptors/<model>/` | 描述符与来源记录目录 |
| `benchmark_ready_root` | `data/benchmarks/` | 下游任务准备清单；实际数据仍在 HF 缓存中 |
| `output_root` | `outputs/main/` | 实验根目录，包含配置快照、任务记录、权重与汇总 |

配置中的相对路径以 **配置文件所在目录** 为基准解析，支持 `~` 和环境变量。命令行传入的 `--root`、`--raw-dir`、`--run-dir` 等路径以调用命令时的工作目录为基准。需要独立实验时，使用新的输出根目录；不要让两份不同协议写入同一个根目录。

## 下载模型与准备数据

### 1. 固定 Hugging Face 缓存

下载、离线检查、训练和后续评估使用同一个用户、Python 环境及缓存位置。以下命令把缓存放到当前项目下；也可改为容量更大的绝对路径：

```bash
export HF_HOME="$PWD/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
```

在线准备期间需要能访问 Hugging Face。若所用数据源要求认证，先在当前环境完成 Hugging Face 登录。离线机器需要复制实际模型、数据和完整 HF 缓存，仅复制准备清单不能替代数据。

### 2. 下载模型

默认模型及扩容设置如下。共享专家不计入 Qwen 的路由专家数量。

| 键 | 官方模型 | 路由专家数量 | 每 token 的 top-k |
| --- | --- | --- | --- |
| `olmoe` | `allenai/OLMoE-1B-7B-0125` | 64 → 96 | 8，扩容前后不变 |
| `qwen` | `Qwen/Qwen1.5-MoE-A2.7B` | 60 → 90 | 4，扩容前后不变 |

```bash
python -u download_models.py download --config protocol.json --models olmoe qwen
python -u download_models.py check --config protocol.json --models olmoe qwen
```

下载器使用 HF 缓存、固定解析后的模型提交版本，并记录下载文件的 SHA-256；下载位置取自配置中的 `path`。中断后重复相同下载命令；`check` 只检查本地文件与哈希，不使用 GPU。需要指定模型 revision 时，可在对应模型配置中填写 `revision`，或对单个模型下载传入 `--revision <commit>`。

已有本地 safetensors checkpoint 时，可直接把 `path` 指向该目录并跳过下载步骤；`download_models.py check` 要求下载器生成的 manifest，不用于登记任意现有模型目录。后续数据准备、描述符采集和训练仍会检查相应输入来源。模型文件、tokenizer、分词数据和描述符存在来源关联，不能用另一个 checkpoint 或 tokenizer 悄悄替换。

### 3. 准备 DCLM 数据

默认使用 `mlfoundations/dclm-baseline-1.0-parquet` 的 `train` split 和 `text` 字段，通过流式读取和有界 shuffle 收集有限原文，不下载整个 DCLM 数据集。两个模型共享同一份原文池，但必须分别使用各自 tokenizer 生成 token ID。原文缓存默认位于 `data/raw/dclm_subset/`。

脚本在分词前按文档哈希划分 train/valid/test，删除完全相同的文本和重复的非空 source ID；文档间追加一个 EOS，再在各 split 内打包并截取准确 token 配额。该检查不包含近重复检测，也不构成与下游 benchmark 的完整去污染证明。

```bash
python -u prepare_data.py prepare --config protocol.json --models olmoe qwen
python -u prepare_data.py check --config protocol.json --models olmoe qwen
```

每个模型的输出还包含 `calibration_train_indices.npy`、各 split 的文档清单、tokenizer 文件副本和原文 manifest，可追溯打包数据及校准抽样来源。

默认序列长度为 2048，数据预算如下：

| split | block 数量 | 输入 token 数量 | 用途 |
| --- | ---: | ---: | --- |
| `train` | 16,384 | 33,554,432 | 继续预训练，一次遍历 |
| `valid` | 2,048 | 4,194,304 | 训练监控与最终完整验证 |
| `test` | 2,048 | 4,194,304 | 固定最终步骤的报告 |
| `calibration` | 512 | 1,048,576 | 从 train 中无放回抽取；用于描述符、分组、预热与梯度评分 |

每个 block 独立执行一次 causal shift，因此训练的 **输入 token** 预算为 `4096 steps × 4 accumulation × 2048 = 33,554,432`，约 33.55M；有效预测目标为 `4096 × 4 × 2047 = 33,538,048`。每个微批次只有一个 block，累积 4 个微批次后进行一次优化器更新。校准集是训练集的子集，不能当作额外独立测试集，也不额外扩大上述继续预训练预算。

使用自备原文时，每个 JSONL 对象提供 `text` 字段，并为该配方指定新的原文缓存目录。以下命令替代前面的默认数据准备命令；若已准备过默认数据，还需先将配置中两个模型的 `data` 改为新的空目录：

```bash
python -u prepare_data.py prepare --config protocol.json --models olmoe qwen \
  --local-jsonl /path/to/corpus_part1.jsonl /path/to/corpus_part2.jsonl \
  --raw-dir data/raw/local_corpus
```

已有完整原文缓存时可离线重新分词或复用已完成数据。对于前面按默认参数准备的 HF 原文缓存：

```bash
python -u prepare_data.py prepare --config protocol.json --models olmoe qwen --offline
```

若准备时使用了自定义 `--raw-dir`，离线时也必须传入同一路径。完整原文缓存会保存来源：纯 `--offline` 可继承已固定的 HF revision 或本地 JSONL 来源，无需再次提供原始 JSONL 文件；模型集合、种子、block 预算与 shuffle buffer 等准备参数仍须一致。仅准备一个模型时，建议为它指定独立 `--raw-dir`，不要再用不同模型集合写入原有缓存。重复相同命令会复用已完成的产物；未完成的分词阶段可能重算。改变配方时使用新目录。

### 4. 准备下游评估数据

下游任务准备不加载模型，也不占用 GPU。每个任务在独立子进程中完成，默认每项超时 900 秒；等待时每 60 秒输出堆栈，便于区分导入、网络和缓存锁问题。

```bash
python -u prepare_eval_data.py download --config protocol.json
python -u prepare_eval_data.py check --config protocol.json
```

`download` 填充缓存并保存任务清单；`check` 设置离线模式，重新核对完整任务集合的数据指纹和任务源码哈希。只有完整检查通过才生成 `offline_ready.json`，供正式训练使用。

单项超时或下载失败后，可以保留已下载缓存并重试：

```bash
python -u prepare_eval_data.py download --config protocol.json --tasks mmlu --timeout 3600 --refresh
python -u prepare_eval_data.py check --config protocol.json --timeout 3600
```

完成下载与完整检查后，训练阶段自动启用 `HF_HUB_OFFLINE=1` 和 `HF_DATASETS_OFFLINE=1`。任务定义、数据指纹或缓存位置改变时，应重新准备与检查，不能把旧标记当作可用缓存的证明。

## 描述符分组与预热

输入模型和 DCLM 数据就绪后运行：

```bash
python -u collect_descriptors.py --config protocol.json --models olmoe qwen --device cuda:0
```

该命令在一张 GPU 上依次加载两个模型，仅读取本地 checkpoint、tokenizer 和 `calibration.npy`。结果写入各模型的 `descriptors` 目录，包括 `report.json`、逐层 `layer_XX.pt`、投影矩阵与校准索引。采集过程不读取 `valid` 或 `test`。

描述符输出目录必须为空或不存在，采集命令不会覆盖或自动续写已有非空目录。若 OLMoE 已完成、Qwen 尚未开始，只需改为 `--models qwen`；某个模型的采集中断时，先保留其失败记录，在配置中为该模型指定新的描述符目录后重新采集。

默认 512 个校准 block 分为 384 个 fit block 和 128 个 check block；每层分别从两部分选取 256 个共同 probe token，专家响应投影到 128 维。fit 用于构造描述符、功能分组及预热；check 只报告留出诊断，不参与选种子、挑分组或决定预热停止点。此处分区以 block 为单位，同一原文的不同 block 可能分别进入 fit 和 check，不宣称文档完全隔离。

训练入口会自动完成后续准备，无须手动调用内部模块：

1. 根据 fit 描述符逐层聚为 8 组，按组流量分配新增专家名额，并选择组内代表专家作为复制源；复制对应路由权重，保持 top-k 不变。
2. 为指定种子生成分组与复制计划，保存在 `prepared/<model>/groups/`。`eu_gn` 另行准备梯度评分。
3. 对启用预热的方法执行逐层、固定 64 步的混合输出蒸馏：原层作为教师，更新扩容层的路由器与新增专家。预热报告记录 check NMSE 前后变化。
4. 对启用置信路由的方法，在正式训练时对低置信候选执行簇内采样；预热和全部验证、测试及推理使用确定性的原生路由。

置信路由先取全局 top-k 候选，再计算候选在本簇中的条件概率与其他专家的最大条件概率之差。差值低于 `routing_tau=0.2` 时，以 `max(0, 1 − step / total_steps)` 的概率在簇内采样。实现排除该 token 已选中的其他专家，保持 k 个不同专家；新增专家继承父专家的簇标签。混合权重取实际选中专家的全局路由概率，并保留原模型的归一化约定。Qwen 共享专家分支保留。

## 训练方法与运行命令

默认种子为 `42 43 44`，每个训练任务运行 4096 个优化器步骤。扩容方法仅训练新增专家和整个路由器；旧专家及其他骨干参数冻结。`continue` 训练原模型全部参数，`original` 不训练。

| CLI 方法名 | 专家复制与分组 | 预热 | 训练路由 |
| --- | --- | --- | --- |
| `original` | 原始 checkpoint，仅评估 | 无 | 原生 |
| `continue` | 不扩容，全参数继续训练 | 无 | 原生 |
| `random_copy` | 随机选择父专家 | 无 | 原生 |
| `traffic_copy` | 按 fit 校准流量降序选择父专家；默认扩容比例下前 32/30 位各复制一次 | 无 | 原生 |
| `eu_gn` | 梯度评分选择父专家；每位父专家至多新增 3 个副本 | 无 | 原生 |
| `graftmoe` | 功能分组、按流量分配与组内代表复制 | 有 | 置信感知簇内路由 |
| `graftmoe_native` | 与 `graftmoe` 相同 | 有 | 原生，用于路由消融 |
| `cluster_no_warm` | 与 `graftmoe` 相同 | 无 | 置信感知簇内路由 |
| `random_groups_warm` | 随机分组后分配与复制 | 有 | 置信感知簇内路由 |

`traffic_copy` 按专家流量排名选取复制源；GRAFT-MoE 则按簇总流量，用最大余数法分配各簇的新增名额。两者的分配规则不同。

`eu_gn` 对校准批次的语言模型梯度先求均值、再计算平方和，按分数降序贪心复制，每位父专家最多新增 3 个副本，并为新增路由 bias 加入 `U(-0.001, 0.001)` 扰动。它是代码中明确定义的适配基线，不声称完整复现其他论文。

训练目标为语言模型交叉熵加负载均衡项，默认权重 `balance_weight=0.01`；均衡项使用实际 expert assignment 的份额。主训练初始学习率 `1e-4`、warmup 128 步、weight decay `0.01`，随后按余弦退火至初始学习率的 10%。

### 先运行少量正式任务

下面只请求 OLMoE 的 `graftmoe`、种子 42，自动附带该模型的一次 `original` 评估，共 2 个任务。预算仍为正式 4096 步。

```bash
python -u run_graftmoe.py --config protocol.json --root outputs/olmoe_graftmoe_seed42 \
  --models olmoe --methods graftmoe --seeds 42 --gpus 4 --max-parallel 1 --check

python -u run_graftmoe.py --config protocol.json --root outputs/olmoe_graftmoe_seed42 \
  --models olmoe --methods graftmoe --seeds 42 --gpus 4 --max-parallel 1
```

`--check` 只进行输入来源、协议和资源检查，不训练。实际运行还会检查下游评估缓存。需要仅评估原始模型时：

```bash
python -u run_graftmoe.py --config protocol.json --root outputs/original_only \
  --models olmoe qwen --methods original --gpus 4 5 --max-parallel 2
```

比较完整方法与路由消融的一次种子，可在新的根目录运行：

```bash
python -u run_graftmoe.py --config protocol.json --root outputs/routing_comparison \
  --models olmoe --methods graftmoe graftmoe_native --seeds 42 --gpus 4 5 --max-parallel 2
```

### 运行完整矩阵

不传 `--models`、`--methods` 和 `--seeds` 时，运行两个模型、八种训练方法、三个种子，并为每个模型评估一次原始 checkpoint：`2 × 8 × 3 + 2 = 50` 个结果任务。描述符、分组及梯度评分等共享准备不计入这 50 个任务。

```bash
python -u run_graftmoe.py --config protocol.json --root outputs/full \
  --gpus 4 5 --max-parallel 2 --check

python -u run_graftmoe.py --config protocol.json --root outputs/full \
  --gpus 4 5 --max-parallel 2
```

长任务可把主进程日志写入文件：

```bash
nohup python -u run_graftmoe.py --config protocol.json --root outputs/full \
  --gpus 4 5 --max-parallel 2 > full_run.log 2>&1 &
```

不要同时启动两个调度器写入同一个实验根目录。更改种子、方法、预算或其他配置时，应使用新的根目录；恢复时保持原命令与代码版本。

## 进度恢复与失败处理

查看全量矩阵的状态：

```bash
python status_graftmoe.py --root outputs/full
```

状态输出分别列出已记录训练步数、可恢复断点步数、已完成下游任务数和本机进程状态。需要机器可读输出时追加 `--json`。进程状态反映当前主机，训练日志中存在进度不表示进程仍在运行。

每个任务的目录为 `runs/<model>_<method>_seed<seed>/`。原始模型任务使用 `seed0` 标识，它不是另一个训练种子。

| 位置 | 内容 |
| --- | --- |
| `protocol_frozen.json` | 解析路径与命令行选择后的协议快照 |
| `requested_matrix.json` | 本次请求的模型、方法、种子和调度参数 |
| `run_sources.json` | 代码来源校验记录 |
| `logs/<model>_<method>_seed<seed>.log` | 每个任务的独立日志，重启时追加 |
| `last_queue_failures.json` | 最近一次队列收集到的失败任务及日志位置 |
| `runs/.../train.jsonl` | 训练步、NLL、负载均衡项、耗时和显存峰值 |
| `runs/.../latest.json`、`checkpoint_<slot>/` | 已原子提交的训练断点与优化器状态 |
| `runs/.../complete.json` | 训练、最终评估和所需保存全部完成后的结果记录 |

默认每 512 步进行一次固定子集验证并保存断点。中断后修复日志中的原因，重复原来的 `run_graftmoe.py` 命令即可：已完成任务会跳过，未完成任务从最近一次成功提交的断点恢复。断点包含 FP32 主参数、优化器与随机数状态；尚未提交的步骤会重新计算。

训练达到 `4096/4096` 后仍需完成全量 valid/test、六项下游任务、推理性能测量和最终权重保存。只有 `complete.json` 表示整个任务完成；日志中的训练步数不能代替最终完成记录。原始模型任务没有训练步骤，但也必须完成全部评估。

显存不足时，先查看所失败的方法与日志，再降低并行数或使用更大显存的单卡。主机内存或磁盘不足时，释放资源后恢复。不要通过编辑完成标记、删除来源校验或混入不同配置来绕过错误。移动大文件后应考虑路径与来源记录的关联，复制一个新的协议到新目录重新开始比修改现有实验快照更可控。

## 保存重载与单独评估

默认 `save_final_weights=true`，所有完成的训练方法、种子均保留最终可重载权重：

| 方法 | 最终产物 | 重载所需材料 |
| --- | --- | --- |
| 七种扩容方法 | `final_delta/` 中的路由器与新增专家权重 | 对应原始 checkpoint、`recipe.json`、`plan.json` 与 delta 清单 |
| `continue` | `final_model/` 中的完整 HF 模型权重 | 完整模型保存目录、运行记录和 loader 来源检查所需的原始 checkpoint |
| `original` | 不复制原始权重 | 配置记录的原始 checkpoint |

delta 是参数增量保存格式，不是可直接交给 `AutoModelForCausalLM.from_pretrained()` 的完整模型目录。请保留准确的原始 checkpoint 与运行目录，使用提供的 loader 重建。

默认 `keep_optimizer_checkpoints=false`：在最终结果和权重保存完成后清理该任务的优化器断点以节省磁盘。最终权重可用于评估与推理；它不包含继续原训练所需的完整优化器状态。若需要保留该状态，在首次启动前把该选项设为 `true`。

只读检查一个已保存任务的元数据与必需文件是否存在；此步骤不加载张量、不占用 GPU，实际权重哈希在重载时检查：

```bash
python load_graftmoe_checkpoint.py --run-dir outputs/full/runs/olmoe_graftmoe_seed42 --check
```

在 Python 中加载：

```python
from load_graftmoe_checkpoint import load_run

model, tokenizer = load_run(
    "outputs/full/runs/olmoe_graftmoe_seed42",
    "cuda:0",
)
```

loader 会验证来源及保存文件，并返回关闭梯度、处于 eval 模式的模型与 tokenizer。此时置信探索关闭，使用正式评估时的确定性原生路由。

对保存模型单独评估。默认使用 `cuda:0`，重算完整 valid split 的 NLL、PPL 和路由统计：

```bash
python -u evaluate_graftmoe.py --run-dir outputs/full/runs/olmoe_graftmoe_seed42
```

需要在物理 GPU 4 上重算完整 test 以及全部冻结下游任务时：

```bash
CUDA_VISIBLE_DEVICES=4 python -u evaluate_graftmoe.py \
  --run-dir outputs/full/runs/olmoe_graftmoe_seed42 --device cuda:0 --split test --tasks
```

省略 `--tasks` 时只评估语言建模与路由；`--tasks hellaswag arc_challenge` 只追加指定任务。下游评估要求原实验根目录中的数据清单和对应 HF 缓存仍可用。命令使用运行时保存的协议，不读取当前编辑后的 `protocol.json`。

每次输出到新的 `runs/.../evaluations/<split>_<时间>_<后缀>/`，也可用 `--output` 指定一个尚不存在的目录。产物包括 `evaluation.json`、`valid.json` 或 `test.json`、逐 block NLL、索引和实际路由计数；指定下游任务时另有任务指标及逐样本记录。独立评估不覆盖原始 `complete.json`，也不会自动纳入训练结果汇总。

## 评估协议与结果汇总

### Valid、test 与下游任务

训练监控固定使用 valid 中相同的 256 个 block，每 512 步评估一次；原模型参考、复制后、预热后及训练曲线使用一致的子集。最终在第 4096 步模型上评估完整 2048 个 valid block 和 2048 个 test block。最终模型按固定训练预算确定，不根据 test 或 check 子集挑选最佳步骤。

NLL 使用标准 causal shift 的预测目标；PPL 为 `exp(NLL)`。不同模型使用不同 tokenizer，因此只在同一模型/tokenizer 内比较 PPL，不用 OLMoE 与 Qwen 的绝对 PPL 高低判断模型优劣。

下游任务使用固定的 lm-eval 0.4.9.1 定义、固定随机种子、batch size 1，不应用 chat template：

| 任务 | few-shot | 报告指标 |
| --- | ---: | --- |
| `hellaswag` | 0 | `acc_norm,none` |
| `arc_challenge` | 0 | `acc_norm,none` |
| `piqa` | 0 | `acc,none` |
| `winogrande` | 0 | `acc,none` |
| `mmlu` | 5 | `acc,none` |
| `gsm8k` | 8 | `exact_match,flexible-extract` |

每项保存指标与逐样本记录。汇总表将任务分数转为百分数，并报告六项分数的算术平均；该平均值应与各项分数同时解读。

### 汇总文件

训练入口在所有请求任务完成后自动汇总，也可手动对现有结果运行：

```bash
python summarize_graftmoe.py --root outputs/full
```

| `summary/` 下的文件 | 内容 |
| --- | --- |
| `completeness.json` | 请求数量、完成数量及缺失任务；首先检查此文件 |
| `per_run_results.csv` | 每个已完成任务的 PPL、NLL、路由、任务分数、资源与耗时 |
| `main_results.csv`、`main_results.tex` | 各方法已完成种子的均值、样本标准差及完成种子数 |
| `paired_differences.csv` | 同种子 `graftmoe` 与各基线的配对 test NLL 差及描述性区间 |
| `curves.csv` | 固定 valid 子集上的训练恢复曲线数据 |
| `recovery.csv` | 首次观测恢复的 token 数、是否右删失与超额 NLL 曲线面积 |
| `recovery_tokens.pdf`、`recovery_tokens.png` | 以主训练输入 token 为横轴的曲线 |
| `recovery_time.pdf`、`recovery_time.png` | 以记录的主训练更新耗时为横轴的曲线 |
| `INTERPRETATION.txt` | 指标与结论边界说明 |

恢复曲线图仅在存在已完成训练任务的曲线记录时生成。

未完成矩阵仍可汇总已有完成结果，但会明确标记缺项。表格只统计已完成的请求种子，不能把 `seeds_finished=1` 当作三种子结果；单次原模型评估没有训练种子标准差。配对区间使用训练种子与连续 32-block 组的分层 bootstrap，属于描述性区间，不是多重比较校正后的显著性结论。

恢复点是固定评估间隔上的第一次 `valid NLL ≤ 原模型 valid NLL`，不是连续时间中的精确恢复时刻。未恢复记录为右删失，不填造一个恢复值。主训练时间包括 CPU 优化器更新，排除描述符、分组、预热、验证和断点 I/O；相关准备与预热成本单独记录。不同并发负载会影响 CPU/I/O 和时间指标。推理性能测量区分完整序列 prefill 与 KV-cache greedy decode，不代表任意部署场景的吞吐量。

### 新专家路由占比

`new_assignment_share` 的定义是：每层新增专家实际收到的 assignment 次数，除以该层所有路由专家的 assignment 次数，再对层求平均。统计来自实际专家调用；一个 token 选择 top-k 个专家就产生 k 次 assignment。Qwen 共享专家另行检查，不计入该分母。

该字段在 JSON/CSV 中是 **0–1 的比例**：例如 `0.228` 显示为百分数才是 `22.8%`。OLMoE 的 32/96 和 Qwen 的 30/90 均为 1/3，因此均匀分配参照都是约 33.3%；这只是参照，不是期望结果或优化目标。占比不能解释成“至少调用过一个新专家的 token 比例”，也不能单独证明功能专门化或下游效果提升。

正式结论应同时检查矩阵完成度、三个种子的波动、配对结果、PPL/NLL、各下游任务与成本。代码不会补造缺失种子、未完成任务的最终值或预期提升。

