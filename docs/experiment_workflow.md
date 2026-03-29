# MRI-SVM 实验执行流程

## 1. 环境准备

```powershell
py -m pip install -r requirements.txt
```

## 2. 主实验流程

### Step 1. 运行 `AD vs NORMAL` 基线实验

```powershell
py scripts/run_mri_svm_experiment.py --labels AD NORMAL --month 00 --output-dir results/ad_vs_normal_m00
```

### Step 2. 查看输出

重点查看：

1. `results/ad_vs_normal_m00/metrics_summary.csv`
2. `results/ad_vs_normal_m00/confusion_matrix_linear.png`
3. `results/ad_vs_normal_m00/confusion_matrix_rbf.png`
4. `results/ad_vs_normal_m00/roc_curve_linear.png`
5. `results/ad_vs_normal_m00/roc_curve_rbf.png`

### Step 3. 完成扩展任务

```powershell
py scripts/run_mri_svm_experiment.py --labels MCI NORMAL --month 00 --output-dir results/mci_vs_normal_m00
py scripts/run_mri_svm_experiment.py --labels AD MCI --month 00 --output-dir results/ad_vs_mci_m00
py scripts/run_mri_svm_experiment.py --labels AD MCI NORMAL --month 00 --output-dir results/multiclass_m00
```

## 3. 一键跑完整 study

```powershell
py scripts/run_full_study.py
```

默认会依次执行：

1. `AD vs NORMAL @ m00`
2. `MCI vs NORMAL @ m00`
3. `AD vs MCI @ m00`
4. `AD/MCI/NORMAL @ m00`

## 4. 报告撰写顺序

1. 先写数据与方法
2. 再填入 `AD vs NORMAL` 主实验结果
3. 再补难度扩展结果
4. 最后总结线性核与 RBF 核差异

## 5. 建议优先展示的结论

1. `AD vs NORMAL` 是否明显优于 `MCI vs NORMAL`
2. RBF 核是否优于线性核
3. Balanced Accuracy 与 F1 是否一致支持同一结论
4. 多分类任务是否比二分类显著更难
