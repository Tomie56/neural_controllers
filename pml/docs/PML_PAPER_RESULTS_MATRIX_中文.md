# Predictive Memory Localization 论文结果矩阵

更新时间：2026-06-27

这份文档把当前 PML 实验结果整理成论文写作可直接使用的矩阵。核心问题不是“知识在哪里”，而是：

> 内部记忆定位 / 表征几何信号什么时候能预测干预是否有效、局部、鲁棒，以及什么时候只能作为风险诊断？

当前最稳的论文主线是：

> localization heatmap 本身不是贡献；真正有用的是 predictive geometry。我们用 intervention strength/path curve 验证定位信号是否能预测 target onset、side-effect onset、clean window 和 editing fragility。

论文表格已自动汇总到：

```text
pml/results/paper_tables/PML_PAPER_TABLES.md
```

对应 CSV：

```text
pml/results/paper_tables/path_summary.csv
pml/results/paper_tables/path_prediction.csv
pml/results/paper_tables/editing_transfer.csv
pml/results/paper_tables/case_audit.csv
pml/results/paper_tables/agop_geometry_diagnostics.csv
pml/results/paper_tables/feature_audit.csv
pml/results/paper_tables/figure_index.csv
```

后续写论文时优先引用这些 paper-facing tables；原始结果文件仍是 authoritative source。

重建命令：

```bash
./pml/scripts/rebuild_pml_reports.sh
```

默认不会跑模型，也不会使用 GPU；它只刷新现有结果上的报告、表格和状态文件。

## 1. 主研究问题到证据矩阵

| 研究问题 | 实验 | 当前证据 | 结论强度 | 能 claim 什么 |
|---|---|---|---|---|
| RQ1: learned localization / direction 是否比 random 更能影响 target？ | 3000 commonsense suppression baseline | learned directions target drop 明显强于 random；最佳 `mean_difference -1.0` target delta `-0.0699`，`logistic -1.0` `-0.0684` | 强 | learned activation directions carry target-specific leverage |
| RQ2: endpoint target success 是否等价于 clean intervention？ | 3000 path geometry audit | learned directions target-any `0.62-0.64`，clean-window 只有 `0.32-0.34` | 强 | endpoint success overstates intervention quality |
| RQ2b: 更密 alpha 下，最小有效强度是否自然稳定？ | held-out 100 dense-alpha path audit | learned directions target-any `0.73-0.74`，clean-window exists `0.68-0.71`，但 strict clean-window path 只有 `0.09-0.11`，unstable 约 `0.39-0.41` | 中等 | minimum effective strength must be path-conditioned, not just first target onset |
| RQ3: AGOP/RFM 是否是好的 direct steering method？ | held-out 144 AGOP direct suppression | `agop_topk_project` target-any `0.8403`，damage-any `0.9028`，clean-window `0.0903`，damage-first `0.5417` | 强边界结果 | broad AGOP top-k is high-leverage but not clean; AGOP is better framed as diagnostic geometry |
| RQ4: localization/path features 能否预测 path type？ | path prediction analysis | 3000 baseline features: clean-window AUROC `0.5994`，usable-path `0.6123`，no-effect `0.6385`; AGOP+method 对 damage-first AUROC `0.6736-0.6874` | 中等 | current features predict path class better than exact onset strength |
| RQ5: path geometry 能否预测 ROME standard efficacy？ | held-out 144 path-to-editing transfer | path-only standard AUROC `0.5324`; controlled over diagnostics 后 delta AUROC `-0.0647` | 负结果 | path geometry does not reliably predict standard editing success beyond difficulty diagnostics |
| RQ6: path geometry 能否预测 editing fragility / superficial success？ | held-out 144 transfer + stability + case audit | fragile path-only AUROC `0.6931`; controlled delta `+0.0888/+0.0918`; bootstrap CI 跨 0；risk_5_plus fragile `0.1579` vs risk_0 `0.0455` | 有前景但未定 | path geometry is a promising diagnostic for fragile/superficial risk, not yet a settled predictor |
| RQ7: path geometry 是否能预测 locality damage？ | held-out 144 transfer + stability | locality damage path-only AUROC `0.5385`; controlled delta `-0.1688`; stability delta AUROC `-0.1095` CI `[-0.1988,-0.0235]` | 负结果 | current path geometry does not improve locality-damage prediction |
| RQ8: target onset 之前是否已有 general capability damage？ | representative 50 capability-onset probe | logistic capability-any `0.4000`，capability-first `0.1600`；mean_difference capability-any `0.4200`，capability-first `0.1000`；strict clean-window 仍只有 `0.02-0.10` | 中等，代表性样本非分布估计 | capability damage exists and sometimes appears first; minimum effective strength must include capability onset |

## 2. 关键结果表

### 2.1 3000 Suppression Baseline

数据：

```text
pml/data/suppression_commonsense_3000_valid.jsonl
```

结果：

```text
pml/results/commonsense_3000_baselines/
```

已完成方法：

- `mean_difference`
- `logistic`
- `linear`
- `random`

核心数字：

| 结果 | 数值 | 解释 |
|---|---:|---|
| best target mean delta | `mean_difference -1.0 = -0.0699` | learned direction 确实能压 target |
| second best target mean delta | `logistic -1.0 = -0.0684` | logistic 接近 mean-difference |
| neighbor damage at strong suppression | `0.29-0.30` | 强 suppression 会破坏 neighbor |
| target suppression prediction | AUROC `0.6058` | localization features 有中等偏弱预测力 |
| neighbor damage prediction | AUROC `0.5575` | side effect 更难预测 |

论文解释：

> learned directions are not random perturbations, but target suppression and locality diverge quickly under stronger intervention.

### 2.2 Path Geometry Audit

结果：

```text
pml/results/path_geometry_audit/
```

主表：

```text
method_strength_summary.csv
path_type_distribution.csv
strength_path_rows.csv
strength_curve_summary.json
```

关键数字：

| Dataset | Method | target any | clean-window | damage-first | no-effect | collapse |
|---|---|---:|---:|---:|---:|---:|
| 3000 baseline | mean_difference | 0.6356 | 0.3425 | 0.1977 | 0.3644 | 0.0158 |
| 3000 baseline | logistic | 0.6231 | 0.3405 | 0.2014 | 0.3769 | 0.0131 |
| 3000 baseline | linear | 0.6315 | 0.3217 | 0.1991 | 0.3685 | 0.0125 |
| 3000 baseline | random | 0.4736 | 0.2098 | 0.1270 | 0.5264 | 0.0044 |
| held-out 144 | agop_top1 | 0.6111 | 0.3264 | 0.1806 | 0.3889 | 0.0000 |
| held-out 144 | agop_topk_project | 0.8403 | 0.0903 | 0.5417 | 0.1597 | 0.2014 |

论文解释：

> target-any success is not a sufficient metric. Clean-window is the right object if the goal is reliable intervention.

### 2.3 Path Prediction

结果：

```text
pml/results/path_geometry_audit/strength_prediction/
pml/results/path_geometry_audit/strength_prediction_with_method/
```

关键数字：

| Setting | Feature group | Outcome | AUROC |
|---|---|---|---:|
| 3000 baseline | baseline features | clean-window | 0.5994 |
| 3000 baseline | baseline features | usable-path | 0.6123 |
| 3000 baseline | baseline features | damage-first | 0.5847 |
| 3000 baseline | baseline features | no-effect | 0.6385 |
| held-out 144 + method | AGOP spectrum | usable-path | 0.6664 |
| held-out 144 + method | AGOP spectrum | damage-first | 0.6736 |
| held-out 144 + method | AGOP topvec | damage-first | 0.6874 |

论文解释：

> localization geometry currently predicts categorical path behavior better than exact minimum alpha.

### 2.4 ROME / Editing Transfer

结果：

```text
pml/results/commonsenseqa_balanced_followup_144/rome/path_editing_transfer/
```

关键数字：

| Outcome | Features | AUROC | AP | 解释 |
|---|---|---:|---:|---|
| standard_success_bool | agop_top1 path | 0.5324 | 0.4932 | weak |
| robust_success_bool | agop_top1 path | 0.6367 | 0.4349 | moderate path-only |
| fragile_success_bool | both paths | 0.6931 | 0.3265 | strongest path-only signal |
| locality_damage_bool | both paths | 0.5385 | 0.1783 | weak |
| weak_post_margin_bool | agop_top1 path | 0.5825 | 0.6538 | weak/moderate |

Controlled over pre-edit diagnostics：

| Outcome | Comparison | delta AUROC | delta AP | 解释 |
|---|---|---:|---:|---|
| fragile_success_bool | diagnostic + agop_top1 path | +0.0918 | +0.0504 | positive |
| fragile_success_bool | diagnostic + both paths | +0.0888 | +0.1338 | positive |
| robust_success_bool | diagnostic + agop_top1 path | -0.0205 | +0.0031 | no gain |
| standard_success_bool | diagnostic + agop_top1 path | -0.0647 | -0.0714 | no gain |
| locality_damage_bool | diagnostic + both paths | -0.1688 | -0.1215 | worse |

论文解释：

> path geometry is not a general edit-success predictor. Its strongest value is diagnosing whether a nominally successful edit may be fragile or superficial.

### 2.4b Path Prediction Feature Audit

结果：

```text
pml/results/path_geometry_audit/feature_audit/
pml/results/paper_tables/feature_audit.csv
```

关键观察：

| Setting | Outcome | Top features | 解释 |
|---|---|---|---|
| 3000 baseline | clean-window | `feature_saliency_entropy`, `feature_saliency_concentration`, `feature_direction_pairwise_cosine_max` | clean path 与 saliency 分布形状和方向一致性有关 |
| 3000 baseline | usable-path | `feature_saliency_entropy`, `feature_direction_pairwise_cosine_min`, `feature_saliency_concentration` | 可用路径更依赖分布式 saliency / direction agreement |
| 3000 baseline | damage-first | `feature_mean_threshold_accuracy`, `feature_saliency_entropy`, `feature_max_gap` | damage-first 与 probe 分离度和 saliency 结构有关 |
| 3000 baseline | no-effect | negative `feature_saliency_entropy`, negative `direction_pairwise_cosine_min` | no-effect 更像低 saliency 分散 / 方向不稳定 |
| held-out 144 | AGOP usable-path | `agop_layer_entropy_concentration`, `agop_min_topk_ratio`, `agop_max_topk_ratio` | AGOP spectrum concentration 对 usable path 有诊断价值 |
| held-out 144 | AGOP damage-first | `agop_layer_entropy_concentration`, `agop_mean_topk_ratio`, `agop_mean_top1_ratio` | damage-first 与 AGOP spectrum/top-k 结构相关 |

注意：

> 这些 top coefficients 是标准化 logistic 模型的解释性诊断，不是因果 attribution。它们说明哪些 feature family 在 path-type prediction 中反复出现，但不能单独解释机制。

### 2.4c AGOP/RFM Diagnostic Geometry

结果：

```text
pml/results/paper_tables/agop_geometry_diagnostics.csv
pml/results/commonsense_500_agop_features/prediction_analysis/agop_prediction_summary.json
pml/results/commonsense_500_agop_neighbor_overlap/prediction_analysis/overlap_prediction_summary.json
```

关键数字：

| Feature family | Outcome | AUROC | AP | 解释 |
|---|---|---:|---:|---|
| AGOP aggregate | target success | 0.6100 | 0.3837 | AGOP spectrum/scale features 对 target leverage 有中等预测力 |
| AGOP aggregate + method/coef | target success | 0.6741 | 0.4636 | 加入干预方法和强度后，target 可压性预测更强 |
| AGOP aggregate | neighbor damage | 0.5941 | 0.2665 | AGOP 本身对 side effect 有弱到中等诊断力 |
| AGOP neighbor overlap | neighbor damage | 0.6325 | 0.2865 | target-neighbor overlap geometry 比纯 AGOP aggregate 更贴近副作用 |
| AGOP neighbor overlap | clean success | 0.6312 | 0.3272 | overlap geometry 也能预测是否存在 clean success |
| AGOP neighbor overlap + method/coef | clean success | 0.6572 | 0.3616 | 加入方法和强度后，clean success 诊断更强 |

论文解释：

> AGOP/RFM 的价值不是直接拿 top-k 子空间大步 steering，而是提供可预测的几何诊断：它能衡量 target leverage、target-neighbor overlap 和 clean success risk。当前 500 条结果支持“AGOP as diagnostic geometry”，而不是“AGOP as clean direct intervention”。

注意：

> 100 条 smoke overlap 的 AUROC 更高，但不作为主 claim；500 条结果更稳，应该作为论文主体证据。

### 2.5 Stability Check

结果：

```text
pml/results/commonsenseqa_balanced_followup_144/rome/path_editing_transfer/stability/
```

关键数字：

| Outcome | Comparison | Metric | Delta | 95% CI | 解释 |
|---|---|---|---:|---|---|
| fragile_success_bool | diagnostic + agop_top1 path | AUROC | +0.0884 | [-0.0199, 0.2070] | positive but unstable |
| fragile_success_bool | diagnostic + agop_top1 path | AP | +0.1322 | [-0.0315, 0.3312] | positive but unstable |
| fragile_success_bool | diagnostic + both paths | AUROC | +0.0800 | [-0.0455, 0.2095] | positive but unstable |
| robust_success_bool | diagnostic + agop_top1 path | AUROC | -0.0580 | [-0.1106, -0.0136] | negative |
| locality_damage_bool | diagnostic + both paths | AUROC | -0.1095 | [-0.1988, -0.0235] | negative |

论文解释：

> fragile-risk signal is promising but underpowered; robust/locality gains are not supported.

### 2.6 Case Audit

结果：

```text
pml/results/commonsenseqa_balanced_followup_144/rome/path_editing_transfer/case_audit/
```

关键数字：

| Bucket | n | standard success | robust success | fragile success | 解释 |
|---|---:|---:|---:|---:|---|
| risk_0 | 22 | 0.4091 | 0.3636 | 0.0455 | risk-free bucket has low fragile rate |
| risk_5_plus | 95 | 0.5263 | 0.3684 | 0.1579 | high-risk bucket has more fragile edits |
| clean_0 | 56 | 0.3929 | 0.2857 | 0.1071 | no clean path has lower robust success |
| clean_3_plus | 55 | 0.5818 | 0.4000 | 0.1818 | clean score helps standard/robust but is not enough |

代表性 failure modes：

- high path-risk fragile:
  - `earn_money_job`
  - `fish_eaten_raw`
  - `skin_covers_body`
  - `teachers_after_teaching`
- high path-risk but robust counterexamples:
  - `index_card_return`
  - `anaconda_habitat`
  - `attached_homes_apartment`

论文解释：

> path geometry should be interpreted as a risk flag. It surfaces failure modes, but representation-level path risk is not equivalent to parameter-editing failure.

## 3. 图表索引

### Figure 1: Endpoint success vs clean intervention

候选图：

```text
pml/results/path_geometry_audit/plots/commonsense_3000_baselines_target_success_rate.png
pml/results/path_geometry_audit/plots/commonsense_3000_baselines_clean_success_rate.png
pml/results/path_geometry_audit/plots/commonsense_3000_baselines_path_type_distribution.png
```

要表达：

> learned directions improve target success, but only a subset yields clean-window paths.

### Figure 2: AGOP top-k boundary result

候选图：

```text
pml/results/path_geometry_audit/plots/heldout_144_agop_direct_target_success_rate.png
pml/results/path_geometry_audit/plots/heldout_144_agop_direct_neighbor_damage_rate.png
pml/results/path_geometry_audit/plots/heldout_144_agop_direct_path_type_distribution.png
```

要表达：

> AGOP top-k has high target leverage but causes broad damage and collapse.

### Figure 3: Onset geometry

候选图：

```text
pml/results/path_geometry_audit/plots/commonsense_3000_baselines_target_vs_damage_onset.png
pml/results/path_geometry_audit/plots/commonsense_3000_baselines_damage_minus_target_onset_hist.png
pml/results/path_geometry_audit/plots/heldout_144_agop_direct_target_vs_damage_onset.png
pml/results/path_geometry_audit/plots/heldout_144_agop_direct_damage_minus_target_onset_hist.png
```

要表达：

> minimum effective strength is meaningful only relative to damage onset.

### Figure 4: Path-to-editing transfer

目前更适合做 table 而不是 plot：

```text
pml/results/commonsenseqa_balanced_followup_144/rome/path_editing_transfer/path_editing_controlled_deltas.csv
pml/results/commonsenseqa_balanced_followup_144/rome/path_editing_transfer/stability/stability_controlled_deltas.csv
```

要表达：

> path geometry adds possible fragile-risk signal but not standard efficacy/locality signal.

### Figure 5: Case audit / failure taxonomy

候选表：

```text
pml/results/commonsenseqa_balanced_followup_144/rome/path_editing_transfer/case_audit/path_editing_case_audit_buckets.csv
pml/results/commonsenseqa_balanced_followup_144/rome/path_editing_transfer/case_audit/path_editing_case_audit_cases.csv
```

要表达：

> high-risk path signatures enrich fragile cases, but robust counterexamples remain.

## 4. 论文贡献建议

当前最稳的 contribution 写法：

1. **Problem reframing**  
   We ask when memory localization predicts intervention, instead of where knowledge is localized.

2. **Path-level evaluation**  
   We introduce strength/path outcomes: target onset, damage onset, clean-window, damage-first, no-effect, collapse.

3. **Boundary result for AGOP/RFM direct intervention**  
   AGOP/RFM top-k geometry is high-leverage but not clean as a direct suppression subspace.

4. **Predictive but limited geometry**  
   Existing localization/path features predict path type moderately, but exact onset and locality damage remain hard.

5. **Transfer to editing fragility**  
   Activation path geometry does not predict standard ROME efficacy beyond diagnostics, but gives a promising signal for fragile/superficial editing risk.

## 5. 不能过度 claim 的点

不要写：

- AGOP/RFM 是最好的 steering 方法。
- localization 可以直接告诉我们编辑哪一层。
- path geometry 已经稳定预测 ROME 成功率。
- PML 能检测 truthfulness。
- high path-risk 必然导致 fragile editing。

应该写：

- AGOP/RFM provides diagnostic geometry for memory-sensitive subspaces.
- localization must be validated by predictive intervention outcomes.
- path geometry is useful for risk diagnosis and failure taxonomy.
- current evidence is strongest for path type and fragile-risk diagnostics, weaker for exact onset and locality prediction.
- capability-onset is measured on a path-representative 50-record subset, but it is not a population-rate estimate for the full dataset.

## 6. 当前缺口

### 6.1 Capability-onset probe

当前状态：

```text
pml/results/path_representative_50_capability_onset_probe/STATUS.md
```

状态是 `complete_with_analysis`。该 probe 已完成 `mean_difference` 和 `logistic` 两个方法，并生成 summary、path audit 和 representative outcome summary。

已补：

- `capability_onset_strength`
- `capability_minus_target_onset`
- `capability-first` path type
- representative selection reason 下的 capability damage rate

关键结果：

| Method | target-any | neighbor damage-any | capability damage-any | clean-window | damage-first | capability-first | unstable | capability onset median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic | 0.8000 | 0.5200 | 0.4000 | 0.0200 | 0.2200 | 0.1600 | 0.3800 | 0.1500 |
| mean_difference | 0.7000 | 0.5200 | 0.4200 | 0.1000 | 0.1600 | 0.1000 | 0.3400 | 0.2000 |

### 6.2 更大 held-out fragile-risk 验证

当前 fragile-risk transfer 只有 held-out 144，bootstrap CI 跨 0。下一步如果要把 claim 做强，需要：

- 增大 held-out editing transfer 样本；
- 或者在 500 ROME subset 上重建同样的 path geometry；
- 或者专门采样 high path-risk / low path-risk matched pairs 做 editing。

### 6.3 Exact onset prediction

当前 coarse alpha grid 下 onset 多落在 `0.1/0.25`。要真正研究 minimum effective strength，需要 dense alpha：

```text
0, -0.02, -0.05, -0.08, -0.1, -0.15, -0.2, -0.25, -0.35, -0.5, -0.75, -1.0
```

但 dense-alpha 不应重复 3000 baseline，而应在 representative subset 上做高分辨率 path measurement。

## 7. 当前一句话结论

> Predictive Memory Localization should be evaluated as predictive geometry over intervention paths. In current experiments, localization/path signals moderately predict path type and expose AGOP top-k as high-leverage but damaging; they do not reliably predict standard ROME efficacy, but provide a promising, still underpowered diagnostic signal for fragile or superficial editing risk.
