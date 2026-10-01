# PML 两个新模型 500-Record 复现实验运行手册

## 1. 实验口径

论文应如实报告：

```text
Qwen3-1.7B-Base: 3,000-record primary study
Qwen3.5-2B-Base: 500-record stratified compact replication
Ministral-3-3B-Base-2512: 500-record stratified compact replication
```

两个新模型使用完全相同的 500 个 record IDs。固定样本位于：

```text
pml/data/pml_replication_500_stratified_seed113/record_ids.txt
```

SHA256：

```text
1b408955d4a911e42a525b2ca41122f8a758422526314190a161dbb196247934
```

样本覆盖 9 个数据集、14 个 domain；326 条来自 2024--2026 数据，174 条来自
classic-pre-2024 数据。

## 2. 固定实验配置

```text
records = 500 stratified records
methods = random, mean_difference, logistic, rfm_agop_top1
alphas = -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5
reference model = Qwen3-1.7B, 28 layers
reference layers = -21, -17
layer mapping = normalized residual depth
seed = 113
resume = enabled
```

Qwen3.5 有 24 个 text layers，自动映射结果为：

```text
-21 / 28-layer reference -> -18 / 24-layer target
-17 / 28-layer reference -> -15 / 24-layer target
```

Ministral 的层数在下载完成后读取 `text_config.num_hidden_layers`，脚本自动计算并将
完整映射写入 `relative_layer_mapping.json`。

## 3. 新增代码

- 通用多模态 text-backbone loader 和 layer resolver：
  `pml/src/predictive_memory_localization/common.py`
- 通用 hook：`pml/src/predictive_memory_localization/generation_utils.py`
- 确定性分层抽样：
  `pml/src/predictive_memory_localization/analysis/select_stratified_replication_records.py`
- 相对深度层映射：
  `pml/src/predictive_memory_localization/analysis/resolve_relative_intervention_layers.py`
- 通用 500-record runner：
  `pml/scripts/run_pml_compact_replication_500_model.sh`
- Qwen3.5 wrapper：
  `pml/scripts/run_pml_replication_500_qwen3_5_2b.sh`
- Ministral wrapper：
  `pml/scripts/run_pml_replication_500_ministral_3_3b.sh`
- 双模型串行队列：
  `pml/scripts/run_pml_replication_500_qwen35_ministral.sh`

## 4. 下载完成检查

脚本会检查：

1. `config.json` 是否存在；
2. weight index 中声明的所有 shard 是否存在；
3. shard 总字节数是否达到 index 的 `total_size`；
4. CUDA 是否可见；
5. text layer 是否能解析；
6. intervention layer 是否在合法范围。

当前检查状态：Qwen3.5 的配置和 index 已下载，但正式 weight shard 尚未完成；
Ministral 目录尚未出现。此时启动脚本会安全退出，不会创建伪结果。

## 5. Smoke Test

模型下载完成后，先分别运行 4-record、单方法 smoke。smoke 使用独立输出目录，
不会污染正式结果。

### Qwen3.5

```bash
cd /data/neural_controllers

MAX_RECORDS=4 \
METHODS="mean_difference" \
RUN_POSTPROCESS=0 \
EXPERIMENT_ROOT=/data/neural_controllers/pml/results/fresh_multidomain_3000/stage_activation_pml/replication_500_smoke/qwen3_5_2b_base \
./pml/scripts/run_pml_replication_500_qwen3_5_2b.sh
```

### Ministral

```bash
cd /data/neural_controllers

MAX_RECORDS=4 \
METHODS="mean_difference" \
RUN_POSTPROCESS=0 \
EXPERIMENT_ROOT=/data/neural_controllers/pml/results/fresh_multidomain_3000/stage_activation_pml/replication_500_smoke/ministral_3_3b_base_2512 \
./pml/scripts/run_pml_replication_500_ministral_3_3b.sh
```

每个 smoke 会测试两个自动映射层。任何 record failure 都会使脚本退出，并阻止后处理。

## 6. 正式运行命令

两个 smoke 均通过后，在 tmux 中串行运行：

```bash
cd /data/neural_controllers

mkdir -p \
  /data/neural_controllers/pml/results/fresh_multidomain_3000/stage_activation_pml/replication_500/logs

set -o pipefail

./pml/scripts/run_pml_replication_500_qwen35_ministral.sh \
  2>&1 | tee -a \
  /data/neural_controllers/pml/results/fresh_multidomain_3000/stage_activation_pml/replication_500/logs/qwen35_ministral_500.log
```

中断后直接重新执行同一命令。不要设置 `OVERWRITE=1`。

单独续跑 Qwen3.5：

```bash
RUN_MINISTRAL=0 ./pml/scripts/run_pml_replication_500_qwen35_ministral.sh
```

单独续跑 Ministral：

```bash
RUN_QWEN35=0 ./pml/scripts/run_pml_replication_500_qwen35_ministral.sh
```

## 7. 输出目录

```text
pml/results/fresh_multidomain_3000/stage_activation_pml/replication_500/
├── qwen3_5_2b_base/
│   ├── outcomes/
│   ├── prediction_dataset/
│   ├── prediction_analysis/
│   ├── prediction_dataset_strict_later/
│   └── prediction_analysis_strict_later/
└── ministral_3_3b_base_2512/
    ├── outcomes/
    ├── prediction_dataset/
    ├── prediction_analysis/
    ├── prediction_dataset_strict_later/
    └── prediction_analysis_strict_later/
```

每个模型共 4 methods × 2 layers × 500 records = 4,000 method--layer record paths。
脚本在全部 8 条 path 完整且没有 unresolved failure 后，才开始 prediction 和 strict-later
CPU 后处理。

## 8. 时间估计

Qwen3-1.7B 主实验的 36,000 method--layer record paths 实测约 54 GPU-hours。按 4,000
paths 缩放，并考虑模型大小、混合架构、重复加载和保守 buffer：

| 模型 | Outcome GPU 时间 | CPU 后处理 | 保守墙钟时间 |
|---|---:|---:|---:|
| Qwen3.5-2B | 7--10 h | 0.5--1 h | 8--12 h |
| Ministral-3-3B | 11--17 h | 0.5--1 h | 12--20 h |
| 合计 | 18--27 GPU-h | 1--2 h | 20--32 h |

加上两个 smoke、一次 OOM 降 batch 或坏 record 调试，建议按 24--36 小时安排。单张
RTX 5090 连续运行约 1--1.5 天，实际计划预留 2 天。

## 9. 论文可支持的结论

500-record replication 主要用于检验：

1. learned directions 是否仍优于 random；
2. target leverage 是否仍不等于 clean selectivity；
3. static localization `L` 是否仍缺少稳定增益；
4. alpha=0.1 response 是否仍能预测 alpha=0.25/0.5 later outcomes；
5. 这些趋势是否跨 Qwen3.5 hybrid architecture 和 Ministral architecture 保留。

不能把每个新模型的样本量写成 3,000，也不应在 500 条上对稀有 medium outcome 做
过强的模型间排名。

