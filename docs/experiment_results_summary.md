# 已运行实验结果摘要

本项目已经完成以下四组 `m00` 实验：

1. `AD vs NORMAL`
2. `MCI vs NORMAL`
3. `AD vs MCI`
4. `AD/MCI/NORMAL` 三分类

## 1. 指标总览

| Task | Model | Accuracy | Balanced Accuracy | F1 | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| AD vs NORMAL | Linear SVM | 0.7692 | 0.7721 | 0.7500 | 0.9071 |
| AD vs NORMAL | RBF SVM | 0.7564 | 0.7507 | 0.7164 | 0.8730 |
| MCI vs NORMAL | Linear SVM | 0.6917 | 0.6896 | 0.7413 | 0.7258 |
| MCI vs NORMAL | RBF SVM | 0.7167 | 0.7045 | 0.7703 | 0.7300 |
| AD vs MCI | Linear SVM | 0.6364 | 0.6068 | 0.4737 | 0.6788 |
| AD vs MCI | RBF SVM | 0.7000 | 0.6204 | 0.4590 | 0.6846 |
| AD/MCI/NORMAL | Linear SVM | 0.5130 | 0.5171 | 0.5065 | 0.7093 |
| AD/MCI/NORMAL | RBF SVM | 0.4935 | 0.4934 | 0.4880 | 0.6989 |

## 2. 结果解读

### 2.1 最适合做主实验的任务

`AD vs NORMAL` 最适合作为主实验：

1. 区分度最高
2. 结果最稳定
3. 最容易与讲义中的最大间隔、核函数、软间隔联系起来

### 2.2 线性核与 RBF 核的比较

本次结果并不是所有任务都“RBF 一定更强”：

1. `AD vs NORMAL` 中线性核表现更好
2. `MCI vs NORMAL` 中 RBF 略优
3. `AD vs MCI` 中 RBF 在 Accuracy 上较优，但整体仍较难
4. 三分类整体难度较高，线性核略稳

这很适合写进报告分析部分：核函数并非必然越复杂越好，效果取决于任务本身的可分性与特征表示方式。

### 2.3 任务难度排序

按当前结果看，任务难度大致为：

1. 最容易：`AD vs NORMAL`
2. 中等：`MCI vs NORMAL`
3. 更难：`AD vs MCI`
4. 最难：三分类 `AD/MCI/NORMAL`

## 3. 推荐写法

如果你要写课程项目或报告，推荐：

1. 以 `AD vs NORMAL` 为主结果
2. 以 `MCI vs NORMAL` 和 `AD vs MCI` 作为扩展实验
3. 用三分类结果说明任务复杂度显著上升

## 4. 对应结果目录

1. `results/ad_vs_normal_m00/`
2. `results/mci_vs_normal_m00/`
3. `results/ad_vs_mci_m00/`
4. `results/multiclass_m00/`

每个目录下都包含：

1. `metrics_summary.csv`
2. `classification_report_*.json`
3. `confusion_matrix_*.png`
4. `best_model_*.joblib`
5. `experiment_config.json`
