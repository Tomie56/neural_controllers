# PML 3,000-Record 主实验最终结果总结

## 1. 完整性审计

本次 Qwen3-1.7B frozen multidomain 主实验已经完整结束：

- 3,000 个唯一 record，9 个数据集、14 个 domain；
- 36,000 个 method--layer outcome rows；
- 6 个实现名称、2 个层位置、7 个 steering strength；
- prediction 共 3,510 个 fold task；
- 3,510/3,510 状态均为 `completed`，没有失败任务和重复 task key；
- record、dataset、domain 三类 split 各 1,170 个任务；
- logistic regression 和 random forest 各 1,755 个任务。

`matched_norm_random` 与 `random` 的全部汇总数字完全相同，因此论文中不能把二者
视为两个独立 baseline。

## 2. Outcome 主结果

| Method | Layer | Target-any | Clean-any | Neighbor damage | Capability damage | No effect |
|---|---:|---:|---:|---:|---:|---:|
| Random | -21 | 12.03 | 11.23 | 10.33 | 11.00 | 87.97 |
| Mean difference | -21 | 15.37 | 14.47 | 10.00 | 10.47 | 84.63 |
| Logistic | -21 | 15.40 | 14.70 | 11.23 | 11.70 | 84.60 |
| Linear | -21 | 14.43 | 13.43 | 11.37 | 11.00 | 85.57 |
| RFM/AGOP top-1 | -21 | **15.80** | **15.03** | 10.53 | 11.60 | **84.20** |
| Random | -17 | 11.37 | 10.60 | 9.10 | 8.60 | 88.63 |
| Mean difference | -17 | 14.03 | 13.17 | 9.73 | 9.30 | 85.97 |
| Logistic | -17 | 13.53 | 13.00 | 9.07 | 9.17 | 86.47 |
| Linear | -17 | 13.40 | 12.70 | 8.83 | 9.17 | 86.60 |
| RFM/AGOP top-1 | -17 | 12.63 | 12.03 | 9.47 | 9.10 | 87.37 |

表中均为百分比。主要观察如下：

1. learned directions 的 target-any 比同层 random 高约 1.3--3.8 个百分点；
2. -21 层通常比 -17 层更容易产生 target effect；
3. RFM/AGOP top-1 在 -21 层得到最高 target-any 和 clean-any，但相对其他 learned
   directions 的优势很小，不足以宣称方法显著胜出；
4. 84.2%--88.6% 的路径仍然是 no-effect，说明严格阈值下可 steering 的 record 是
   少数，而不是普遍现象；
5. medium-strength suppression/enhancement success 仅约 2.7%--5.0%，主结论应是
   “存在有限但一致的可控窗口”，不能写成“大规模稳定控制”；
6. `clean_any` 只表示 coarse grid 上至少存在一个 clean alpha，不等同于 dense scan
   中连续且稳定的 clean window。

跨数据集检查显示：将四种 learned direction 平均后，target-any 和 clean-any 在 9/9
数据集上均高于 random；但 neighbor/capability damage 仅在 6/9 数据集上更高。这说明
target leverage 的跨数据集方向较一致，而 collateral damage 的变化更依赖数据域。

## 3. Pre-Intervention Prediction：静态定位特征不足

Record-held-out、all cohort、logistic regression：

| Outcome | B+M AUROC/AP | B+M+L AUROC/AP | AUROC delta |
|---|---:|---:|---:|
| Target-any | 0.7005 / 0.2484 | 0.7004 / 0.2499 | -0.0001 |
| Clean-any | 0.6943 / 0.2318 | 0.6944 / 0.2334 | +0.0001 |
| Neighbor damage | 0.6639 / 0.1631 | 0.6665 / 0.1640 | +0.0026 |
| Capability damage | 0.5903 / 0.1379 | 0.5905 / 0.1382 | +0.0003 |

Dataset-held-out 和 domain-held-out 的结论一致：`+L` 对上述四类 outcome 的 AUROC
增益通常只有约 0--0.002。Random forest 下也没有稳定的 `+L` 增益，部分任务甚至
下降。因此当前最稳健的主结论是：

> baseline difficulty、method 和 context 已经包含主要的 pre-intervention signal；
> 当前静态 localization summary 不能可靠预测某个 instance 是否存在有效或干净的
> intervention path。

这个 negative result 是主实验最清晰、最稳健的发现之一。

## 4. RFM Geometry：小幅且 outcome-specific

RFM cohort、record-held-out、logistic regression：

| Outcome | B+M AUROC | B+M+L+G AUROC | Delta |
|---|---:|---:|---:|
| Target-any | 0.6893 | 0.7074 | +0.0181 |
| Clean-any | 0.6833 | 0.7011 | +0.0177 |
| Neighbor damage | 0.6658 | 0.6791 | +0.0133 |
| Capability damage | 0.5888 | 0.5760 | -0.0128 |

Random forest 复现了 target/clean 的小幅正增益，但 capability damage 同样下降。
因此 `G` 可以作为 RFM 的机制分析或 bonus result，不能写成通用预测器。

## 5. Low-Strength Response：强诊断信号，但不是纯预干预测

`R` 使用 alpha=0.1 下 target、neighbor 和 capability 的正负方向响应。Logistic
regression 的 record-held-out 结果为：

| Outcome | B+M AUROC/AP | B+M+L+R AUROC/AP | AUROC delta |
|---|---:|---:|---:|
| Target-any | 0.7005 / 0.2484 | 0.7170 / 0.2946 | +0.0165 |
| Clean-any | 0.6943 / 0.2318 | 0.7126 / 0.2793 | +0.0183 |
| Neighbor damage | 0.6639 / 0.1631 | 0.8854 / 0.6237 | +0.2215 |
| Capability damage | 0.5903 / 0.1379 | 0.9008 / 0.6445 | +0.3105 |
| Suppression at 0.5 | 0.6768 / 0.0783 | 0.8410 / 0.2481 | +0.1642 |
| Clean suppression at 0.5 | 0.6696 / 0.0703 | 0.8358 / 0.2262 | +0.1662 |

Dataset/domain-held-out 下，damage prediction 仍约为 0.88--0.90 AUROC，medium-strength
success 约为 0.82 AUROC。Random forest 也复现了强增益，并对 target/clean path 得到
更高的非线性预测性能。

但是必须严格限定解释：

- `R` 是一次低强度 intervention 后的动态诊断，不是纯 pre-intervention feature；
- 对 alpha=0.5 outcome，它可被解释为 early-response-to-later-outcome prediction；
- 对 `*_any_path`，label 本身覆盖 alpha=0.1，因此 `R` 与 label 有部分直接重合，不能
  把高 AUROC 当作独立的事前预测能力；
- 论文中应把 `+R` 标成 secondary diagnostic / probe-then-decide，而不是主 PML predictor。

## 6. 当前论文可以安全写出的结论

1. Learned directions 在多域数据上提供 modest、跨数据集一致的 target leverage；
2. 有效且无明显 collateral damage 的路径只占少数，steerability 是 instance-specific；
3. 当前静态 localization statistics 不足以识别这些 instance；
4. RFM geometry 只提供小幅且 outcome-specific 的额外信息；
5. 一次低强度 probe 能显著改善后续强度下的成功和风险判断，尤其是 collateral
   damage，但它属于动态诊断策略；
6. 主论文不应宣称 RFM 显著优于所有 learned directions，也不应宣称已解决纯
   pre-intervention instance-level steerability prediction。

## 7. 论文前仍需补的统计与实验

### 不需要 GPU 的立即任务

- 对 learned-minus-random outcome 做 paired bootstrap confidence intervals；
- 给 outcome 表增加 dataset macro-average 和 domain macro-average；
- 报告 prediction 的 calibration、risk--coverage 和 probe-then-decide utility；
- 将 alpha=0.1 从 any-path label 中剔除，重新构造“early response predicts later path”
  的严格无重合标签；
- 检查 class imbalance 下 AP、Brier 和阈值选择，避免只报告 AUROC。

### 投稿前高优先级 GPU 实验

- 修正 endpoint evaluator 后重新做 endpoint validation；
- 在第二模型上复现最小主结论：learned-vs-random outcome、`+L` 静态不足、early
  response 对 later outcome 的动态诊断价值。

## 8. 原始产物

- Outcome/prediction 报告：
  `pml/results/fresh_multidomain_3000/stage_activation_pml/main_3000/qwen3_1_7b/prediction_analysis/PML_MAIN_PREDICTION_REPORT.md`
- Prediction summary：
  `pml/results/fresh_multidomain_3000/stage_activation_pml/main_3000/qwen3_1_7b/prediction_analysis/prediction_summary.csv`
- Fold-level results：
  `pml/results/fresh_multidomain_3000/stage_activation_pml/main_3000/qwen3_1_7b/prediction_analysis/fold_results.jsonl`
- Final prediction dataset：
  `pml/results/fresh_multidomain_3000/stage_activation_pml/main_3000/qwen3_1_7b/prediction_dataset/pml_path_prediction_dataset.csv`

