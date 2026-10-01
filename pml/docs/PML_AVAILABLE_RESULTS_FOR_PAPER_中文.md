# PML 已有实验结果与论文使用建议

## 1. 当前最重要的状态更新

Qwen3-1.7B frozen multidomain 3,000-record 主 outcome 和 grouped prediction
已经全部完成：

```text
records = 3,000
method--layer rows = 36,000
methods = linear, logistic, mean_difference, random,
          matched_norm_random, rfm_agop_top1
layers = -21, -17
alphas = -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5
prediction splits = record, dataset, domain
```

主 prediction 报告位于：

```text
pml/results/fresh_multidomain_3000/stage_activation_pml/main_3000/
qwen3_1_7b/prediction_analysis/PML_MAIN_PREDICTION_REPORT.md
```

因此论文现在应以这组 3,000-record uniform experiment 为主，历史实验只负责解释
现象、路径定义和 failure boundary。

## 2. 主文必须使用的结果

### 2.1 Frozen Multidomain 3,000 Outcome Study

#### 已完成内容

- 3,000 records；
- 6 个实现名称、2 层、7 个 alpha；
- 3 target、3 neighbor、4 capability probes；
- 12 个 method--layer paths 全部完成，无 unresolved failure。

#### 主要结果

| Method | Layer | Target-any | Clean-any | Neighbor damage | Capability damage | No effect |
|---|---:|---:|---:|---:|---:|---:|
| Random | -21 | 12.03 | 11.23 | 10.33 | 11.00 | 87.97 |
| Mean difference | -21 | 15.37 | 14.47 | 10.00 | 10.47 | 84.63 |
| Logistic | -21 | 15.40 | 14.70 | 11.23 | 11.70 | 84.60 |
| Linear | -21 | 14.43 | 13.43 | 11.37 | 11.00 | 85.57 |
| RFM/AGOP top-1 | -21 | 15.80 | 15.03 | 10.53 | 11.60 | 84.20 |
| Random | -17 | 11.37 | 10.60 | 9.10 | 8.60 | 88.63 |
| Mean difference | -17 | 14.03 | 13.17 | 9.73 | 9.30 | 85.97 |
| Logistic | -17 | 13.53 | 13.00 | 9.07 | 9.17 | 86.47 |
| Linear | -17 | 13.40 | 12.70 | 8.83 | 9.17 | 86.60 |
| RFM/AGOP top-1 | -17 | 12.63 | 12.03 | 9.47 | 9.10 | 87.37 |

百分数均为 record-level path rates。`matched_norm_random` 与 `random` 数值完全
相同，不应作为独立 baseline 放入论文。

#### 可支持的结论

1. learned directions 的 target-any 比 random 高约 1.3--3.8 percentage points；
2. -21 层整体比 -17 层更强；
3. RFM/AGOP top-1 在 -21 层 target-any 和 clean-any 最高，但优势不大；
4. 当前 coarse-grid `clean-any` 是“存在至少一个 clean strength”，不能与 dense
   study 的 strict stable clean window 直接比较；
5. 结果支持 modest but consistent leverage，不支持“learned direction 大幅提升”或
   “某方法显著最优”，直到 paired CI 完成。

#### 论文位置

**主文核心 Table 1/2。** 需要补 paired learned-minus-random confidence intervals
和 dataset/domain macro averages 后定稿。

### 2.2 Frozen Multidomain Prediction Ablation

#### 已完成内容

- record-grouped、dataset-grouped、domain-grouped CV；
- logistic regression 和 random forest；
- all、learned、RFM cohorts；
- `B`、`B+M`、`B+M+L`、`+G`、单独的动态 `+R`；
- AUROC、AP、balanced accuracy、F1、Brier 和 fold bootstrap interval。

#### 最关键结果：静态 L 基本没有增益

Record-held-out、all cohort、logistic：

| Outcome | B+M AUROC | B+M+L AUROC | Delta |
|---|---:|---:|---:|
| Target-any | 0.7005 | 0.7004 | -0.0001 |
| Clean-any | 0.6943 | 0.6944 | +0.0001 |
| Neighbor damage | 0.6639 | 0.6665 | +0.0026 |
| Capability damage | 0.5903 | 0.5905 | +0.0003 |

Dataset/domain-held-out 下，`+L` 的增益也通常接近零。

#### 动态 R 有明显增益

| Outcome | B+M AUROC | B+M+L+R AUROC | Delta |
|---|---:|---:|---:|
| Target-any | 0.7005 | 0.7170 | +0.0165 |
| Clean-any | 0.6943 | 0.7126 | +0.0183 |
| Neighbor damage | 0.6639 | 0.8854 | +0.2215 |
| Capability damage | 0.5903 | 0.9008 | +0.3105 |
| Suppression success at medium strength | 0.6768 | 0.8410 | +0.1642 |
| Clean suppression at medium strength | 0.6696 | 0.8358 | +0.1662 |

Dataset/domain transfer 下，`R` 对 neighbor/capability damage 的大幅增益仍保留；
对 target/clean path 的增益较小但为正。

#### RFM Geometry G 的结果

RFM cohort、record-held-out：

| Outcome | B+M AUROC | B+M+L+G AUROC | Delta |
|---|---:|---:|---:|
| Target-any | 0.6893 | 0.7074 | +0.0181 |
| Clean-any | 0.6833 | 0.7011 | +0.0177 |
| Neighbor damage | 0.6658 | 0.6791 | +0.0133 |
| Capability damage | 0.5888 | 0.5760 | -0.0128 |

Geometry 有 modest、outcome-specific 的信号，不应写成普遍增益。

#### 可支持的论文主结论

> Static localization features do not add meaningful predictive information
> beyond baseline difficulty and experimental context. A low-strength response
> diagnostic is substantially more informative, especially for collateral
> damage.

这是当前最强、最完整的 PML 主结果，应采用论文的 Branch B：

```text
static localization is insufficient;
dynamic low-cost response is needed.
```

#### 论文位置

**主文核心 Table 3 + Figure。** 这是标题、Abstract 和 Conclusion 都应围绕的结果。

## 3. 主文可使用的历史机制/边界结果

### 3.1 Dense-Alpha Held-Out 100

| Direction | Target-any | Clean exists | Strict clean | Damage first | Unstable |
|---|---:|---:|---:|---:|---:|
| Mean difference | 74 | 68 | 11 | 20 | 41 |
| Logistic | 73 | 70 | 9 | 24 | 39 |
| Linear | 73 | 71 | 10 | 21 | 41 |
| Random | 60 | 57 | 4 | 15 | 41 |

#### 价值

它直接说明“存在一个 clean alpha”与“存在稳定 clean path”不同，是 PML path
定义最有说服力的证据。

#### 论文位置

**主文一个紧凑表或图。** 不要与 3,000-record coarse clean-any 混成同一指标。

### 3.2 AGOP Direct Intervention Boundary, 144 Records

| Direction | Target-any | Damage-any | Clean window | Damage first |
|---|---:|---:|---:|---:|
| AGOP top-1 | 61.1 | 40.3 | 32.6 | 18.1 |
| AGOP top-k projection | 84.0 | 90.3 | 9.0 | 54.2 |

#### 价值

这是“sensitivity geometry 不等于 selectivity geometry”的最强 failure boundary。

#### 论文位置

**主文 Figure 的一个 panel 或小表。** 必须说明它来自独立的 144-record cohort。

## 4. 适合作为 Bonus 或附录的结果

### 4.1 Stage 2B Dense Strength Selector

#### 已完成

- 500 records、27-point dense grid 训练；
- 100 independent records 外部验证；
- methods = mean difference、logistic、random；
- layers = -13、-17、-21 validation；
- localization、baseline、candidate alpha、early response feature ablation。

#### 关键结果

- localization-only AUROC 多为 0.48--0.56；
- 加 baseline/candidate alpha 后约 0.60；
- 加 early response 后最强模型 AUROC 0.891、AUPRC 0.332；
- suppression clean rate 有小幅提升，enhancement 基本持平；
- selector 仍偏向极端 alpha，不能替代 dense scan。

#### 论文位置

**Bonus subsection 或 Appendix。** 它支持动态诊断，但不应变成第二条主线。

### 4.2 Coarse Commonsense Path Study, 2,969 Records

学习方向 target-any 约 62--64%，random 为 47.4%；学习方向 clean window
约 32--34%，neighbor damage 约 44--45%。

#### 论文位置

可作为 **Appendix robustness/history**，或者 Introduction 中一句 motivating
result。现在已有 uniform 3,000-record 主实验，不建议再给它一整张主文大表。

### 4.3 Historical Path Prediction

- 2,969-record baseline clean-window AUROC 0.599；
- usable-path AUROC 0.612；
- no-effect AUROC 0.639；
- AGOP damage-first AUROC 0.674--0.687。

#### 论文位置

**Appendix。** 已被新的 3,000-record grouped ablation 完整替代。

### 4.4 Historical AGOP Geometry Prediction, 500 Records

- AGOP aggregate target-success AUROC 0.610；
- neighbor-damage AUROC 0.594；
- 加 method coefficient 后约 0.622--0.674；
- target--neighbor overlap 对 clean/damage AUROC 约 0.63--0.66。

#### 论文位置

**Appendix mechanism evidence。** 主文使用新的 RFM `+G` ablation 即可。

### 4.5 ROME / Editing Transfer

144-record path transfer：

- standard success path-only AUROC 0.532；
- robust success AUROC 0.637；
- fragile success AUROC 0.693；
- locality damage AUROC 0.539；
- controlled fragility delta AUROC 约 +0.09，但 interval 跨零；
- efficacy 和 locality controlled increments 为负。

#### 论文位置

**Appendix 或 Discussion 一段。** 只能支持 activation path 对 fragility 有 tentative
diagnostic signal，不能支持普遍 editing transfer。

### 4.6 Stage 0 / Stage 1 / Stage 2 Layer Pilots

包括：

- alpha=0 margin sanity；
- 50-record evaluator smoke；
- 500-record dense layer sweep；
- vector-family pilot；
- PCA-diff weak signal；
- layer `-13/-17/-21` 比 `-1/-5/-9` 更有效的 pilot evidence。

#### 论文位置

**Appendix reproducibility 和 layer-selection justification。** 不作为独立贡献。

### 4.7 Case Audit

现有 ROME/path case buckets 可以作为失败案例来源，但样本较小、差异并不稳定。

#### 论文位置

**Appendix qualitative examples。** 应展示真实 prompt/output，而不是只报 bucket rate。

## 5. 已有结果但不建议放入当前论文

### 5.1 Current Endpoint Stage 2C / 2D

实验已经完成，不是“没有结果”：

- Stage 2C：100 records，3 methods，2 layers，9 alphas；
- Stage 2D：60 selected records，3 methods，2 layers，9 alphas。

但 baseline target endpoint correct rate 仅 5%，Stage 2C 最佳 target gain 只有 1
percentage point，Stage 2D 所有路径 target gain 为 0。结果主要反映 target endpoint
construction/answer matching 无效，而不能验证 margin path。

#### 使用建议

**不要作为主文 endpoint validation。** 可以在内部失败记录或 Appendix threats 中
简述，但投稿前应修正 endpoint evaluator 后重跑。

### 5.2 Gradient-Response Structural Estimator

Stage 2E sign accuracy 约 0.50，不能可靠预测方向；结构估计器没有达到替代 scan
的标准。

#### 使用建议

不放主文。若 strength bonus 讨论“为什么一阶近似不足”，可放 Appendix negative
result。

### 5.3 Adaptive / Reduced Alpha Scan

120 matched paths 上：

- target-any accuracy 0.858；
- damage-any accuracy 0.950；
- clean suppression/enhancement accuracy 0.883/0.925；
- suppression/enhancement onset MAE 约 0.156/0.153；
- 但部分 adaptive grids 有 19--31 个 alpha，未真正稳定减少 dense 27-point cost；
- GO/NO-GO 总结为 FAIL。

#### 使用建议

不放主文。可作为 strength-selector Appendix 中的失败尝试，不能声称 efficient
strength search 已解决。

### 5.4 Stage Activation PML 1,000 Outcome

该实验取 frozen 文件前缀，domain/dataset 分布不均，并已被 3,000-record uniform
主实验替代。

#### 使用建议

不放论文结果，只作为 pipeline validation。

### 5.5 Incomplete Stage 3 Full 3,000 Attempt

旧 Stage 3 目录只有部分 raw scores 和 failures，没有完整 summary。

#### 使用建议

不引用、不报告。

## 6. 推荐的最终论文实验结构

### Main Text

1. Frozen multidomain 3,000 outcome table；
2. Frozen grouped prediction ablation：`B+M`、`+L`、RFM `+G`、dynamic `+R`；
3. Dense 100 path-stability result；
4. AGOP top-1/top-k leverage--damage boundary；
5. 后续修正后的 endpoint validation；
6. 后续第二模型 compact replication。

### Bonus / Short Main Paragraph

7. Strength selector / onset prediction，突出 static localization weak、early response
   strong，但 selection utility modest。

### Appendix

8. 2,969-record historical coarse path；
9. historical prediction 和 feature coefficients；
10. 500-record AGOP geometry；
11. layer/vector-family pilots；
12. ROME transfer 和 case audit；
13. gradient/adaptive scan negative results（可选）。

### Exclude

14. current invalid endpoint pilot；
15. single-domain/prefix 1,000 outcome；
16. incomplete Stage 3 attempt；
17. duplicated matched-norm random rows。

## 7. 当前论文应采用的主故事

已有结果不支持“静态 localization features 能预测 clean intervention”的正面路线。
它支持更清楚的边界结论：

> Learned directions provide modest target leverage over random controls, but
> static localization features add almost no predictive value beyond baseline
> difficulty and experimental context. Low-strength response diagnostics are
> substantially more informative, particularly for collateral damage.

Dense path 和 AGOP boundary 进一步解释为什么：单点有效性、可选强度的存在、稳定
clean control 和 sensitivity geometry 是不同的对象。

