import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
import os
import sys
import io

# 修复终端输出 emoji 报错的问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 尝试加载数据
data_paths = [
    r"C:\Users\sunwenbo\Desktop\adnimerge.csv",
    "adnimerge.csv"
]

df = None
for path in data_paths:
    if os.path.exists(path):
        print(f"✅ 成功找到数据文件: {path}")
        df = pd.read_csv(path, low_memory=False)
        break

if df is None:
    print("❌ 未找到 adnimerge.csv，请确保文件在桌面或当前目录下。")
    exit()

print(f"\n📊 数据集基本信息:")
print(f"   - 总样本数: {len(df)}")
print(f"   - 特征数量: {len(df.columns)}")

# ==========================================
# 任务一：支持向量分类 (SVC) - 区分正常人(CN)和阿尔茨海默症(AD)
# 知识点：间隔与支持向量、软间隔、核函数、对偶问题
# ==========================================
print("\n" + "="*60)
print("任务一：支持向量分类 (SVC) - CN vs AD")
print("="*60)

# 选取基线数据，且只看 CN 和 AD
df_svc = df[(df['VISCODE'] == 'bl') & (df['DX.bl'].isin(['CN', 'AD']))].copy()

# 使用更多特征进行多维度分析
features_svc = ['MMSE', 'CDRSB', 'AGE', 'ADAS11', 'FAQ']
df_svc = df_svc.dropna(subset=features_svc + ['DX.bl'])

print(f"\n📋 用于分类的样本数: {len(df_svc)}")
print(f"   - CN (正常人): {sum(df_svc['DX.bl'] == 'CN')}")
print(f"   - AD (阿尔茨海默症): {sum(df_svc['DX.bl'] == 'AD')}")

X_svc = df_svc[features_svc].values
y_svc = np.where(df_svc['DX.bl'] == 'AD', 1, 0)

# 数据标准化
scaler_svc = StandardScaler()
X_svc_scaled = scaler_svc.fit_transform(X_svc)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_svc_scaled, y_svc, test_size=0.2, random_state=42, stratify=y_svc
)

# ========== 1.1 对偶问题分析 ==========
print("\n--- 1.1 对偶问题分析 (Dual Problem) ---")
print("SVM通过求解对偶问题来找到最优分类边界。")
print("对偶变量 alpha_i > 0 的样本就是支持向量。\n")

svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train, y_train)

# 获取对偶系数 (dual coefficients)
dual_coef = svm_rbf.dual_coef_
support_indices = svm_rbf.support_
support_vectors = svm_rbf.support_vectors_

print(f"支持向量总数: {len(support_indices)}")
print(f"支持向量占比: {len(support_indices)/len(X_train)*100:.2f}%")
print(f"\n对偶系数统计:")
print(f"   - 最大绝对值: {np.max(np.abs(dual_coef)):.4f}")
print(f"   - 最小绝对值: {np.min(np.abs(dual_coef)):.4f}")
print(f"   - 平均值: {np.mean(np.abs(dual_coef)):.4f}")

# 预测和评估
y_pred = svm_rbf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 RBF核 SVM 分类准确率: {accuracy:.4f}")

# ========== 1.2 多核函数对比实验 ==========
print("\n--- 1.2 多核函数对比实验 (Kernel Comparison) ---")
print("比较不同核函数在相同数据上的表现\n")

kernels = {
    'linear': {'kernel': 'linear', 'C': 1.0},
    'poly': {'kernel': 'poly', 'C': 1.0, 'degree': 3, 'gamma': 'scale'},
    'rbf': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
    'sigmoid': {'kernel': 'sigmoid', 'C': 1.0, 'gamma': 'scale'}
}

kernel_results = {}
for name, params in kernels.items():
    svm = SVC(**params)
    # 使用5折交叉验证
    cv_scores = cross_val_score(svm, X_svc_scaled, y_svc, cv=5)
    svm.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, svm.predict(X_test))
    
    kernel_results[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_acc': test_acc,
        'n_support': len(svm.support_)
    }
    
    print(f"{name:8s} | CV准确率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f}) | "
          f"测试准确率: {test_acc:.4f} | 支持向量数: {len(svm.support_)}")

# ========== 1.3 软间隔参数 C 的影响分析 ==========
print("\n--- 1.3 软间隔与正则化参数 C 的影响分析 ---")
print("C值控制对误分类的惩罚程度：")
print("  - C小：允许更多误分类，决策边界更平滑（高偏差，低方差）")
print("  - C大：严格要求正确分类，可能导致过拟合（低偏差，高方差）\n")

C_values = [0.01, 0.1, 1, 10, 100]
C_results = {'C': [], 'train_acc': [], 'test_acc': [], 'n_support': []}

for C in C_values:
    svm_c = SVC(kernel='rbf', C=C, gamma='scale')
    svm_c.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, svm_c.predict(X_train))
    test_acc = accuracy_score(y_test, svm_c.predict(X_test))
    
    C_results['C'].append(C)
    C_results['train_acc'].append(train_acc)
    C_results['test_acc'].append(test_acc)
    C_results['n_support'].append(len(svm_c.support_))
    
    print(f"C={C:6.2f} | 训练准确率: {train_acc:.4f} | 测试准确率: {test_acc:.4f} | "
          f"支持向量: {len(svm_c.support_)}")

# ========== 1.4 网格搜索寻找最优参数 ==========
print("\n--- 1.4 网格搜索最优超参数 (Grid Search) ---")

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(
    SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0
)
grid_search.fit(X_svc_scaled, y_svc)

print(f"最优参数组合: {grid_search.best_params_}")
print(f"最优交叉验证准确率: {grid_search.best_score_:.4f}")

# 使用最优参数的模型
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

print(f"\n最优模型在测试集上的表现:")
print(f"准确率: {accuracy_score(y_test, y_pred_best):.4f}")
print(f"\n分类报告:")
print(classification_report(y_test, y_pred_best, target_names=['CN', 'AD']))

# ==========================================
# 任务二：支持向量回归 (SVR) - 预测认知评分
# 知识点：支持向量回归、epsilon-不敏感损失
# ==========================================
print("\n" + "="*60)
print("任务二：支持向量回归 (SVR) - 预测 MMSE 评分")
print("="*60)

# 使用更多特征预测 MMSE
df_svr = df[df['VISCODE'] == 'bl'].copy()
features_svr = ['AGE', 'CDRSB', 'ADAS11', 'ADAS13', 'FAQ', 'MOCA']
target_svr = 'MMSE'
df_svr = df_svr.dropna(subset=features_svr + [target_svr])

print(f"\n📋 用于回归的样本数: {len(df_svr)}")

X_svr = df_svr[features_svr].values
y_svr = df_svr[target_svr].values

# 标准化
scaler_X_svr = StandardScaler()
X_svr_scaled = scaler_X_svr.fit_transform(X_svr)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_svr_scaled, y_svr, test_size=0.2, random_state=42
)

# 不同 epsilon 值的对比
print("\n--- SVR 中 epsilon 参数的影响 ---")
print("epsilon 定义了不计算误差的'管道'宽度\n")

epsilon_values = [0.01, 0.1, 0.5, 1.0]
for eps in epsilon_values:
    svr = SVR(kernel='rbf', C=10.0, epsilon=eps)
    svr.fit(X_train_r, y_train_r)
    y_pred_r = svr.predict(X_test_r)
    mse = mean_squared_error(y_test_r, y_pred_r)
    rmse = np.sqrt(mse)
    
    print(f"epsilon={eps:.2f} | MSE: {mse:.4f} | RMSE: {rmse:.4f}")

# 最优 SVR 模型
svr_best = SVR(kernel='rbf', C=10.0, epsilon=0.1)
svr_best.fit(X_train_r, y_train_r)
y_pred_r_best = svr_best.predict(X_test_r)
mse_best = mean_squared_error(y_test_r, y_pred_r_best)

print(f"\n🎯 最优 SVR 模型 MSE: {mse_best:.4f}")
print(f"   RMSE: {np.sqrt(mse_best):.4f}")

# ==========================================
# 可视化
# ==========================================
print("\n" + "="*60)
print("生成可视化图表...")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 图1: 核函数对比
ax1 = axes[0, 0]
kernels_names = list(kernel_results.keys())
cv_means = [kernel_results[k]['cv_mean'] for k in kernels_names]
cv_stds = [kernel_results[k]['cv_std'] for k in kernels_names]

bars = ax1.bar(kernels_names, cv_means, yerr=cv_stds, capsize=5, 
               color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
ax1.set_ylabel('Cross-Validation Accuracy')
ax1.set_title('不同核函数性能对比')
ax1.set_ylim([0.5, 1.0])
for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
    ax1.text(i, mean + std + 0.01, f'{mean:.3f}', ha='center', fontsize=10)

# 图2: C值影响分析
ax2 = axes[0, 1]
ax2.semilogx(C_results['C'], C_results['train_acc'], 'o-', label='Train Accuracy', linewidth=2)
ax2.semilogx(C_results['C'], C_results['test_acc'], 's-', label='Test Accuracy', linewidth=2)
ax2.set_xlabel('C (Regularization Parameter)')
ax2.set_ylabel('Accuracy')
ax2.set_title('软间隔参数 C 对模型性能的影响')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 图3: 支持向量数量随C的变化
ax3 = axes[0, 2]
ax3.semilogx(C_results['C'], C_results['n_support'], 'D-', color='purple', linewidth=2)
ax3.set_xlabel('C (Regularization Parameter)')
ax3.set_ylabel('Number of Support Vectors')
ax3.set_title('支持向量数量随 C 的变化')
ax3.grid(True, alpha=0.3)

# 图4: 对偶系数分布
ax4 = axes[1, 0]
ax4.hist(np.abs(dual_coef.flatten()), bins=30, color='teal', edgecolor='black', alpha=0.7)
ax4.set_xlabel('|Dual Coefficient| (|alpha|)')
ax4.set_ylabel('Frequency')
ax4.set_title('对偶系数 (alpha) 分布直方图')
ax4.axvline(np.mean(np.abs(dual_coef)), color='red', linestyle='--', 
            label=f'Mean: {np.mean(np.abs(dual_coef)):.3f}')
ax4.legend()

# 图5: 混淆矩阵
ax5 = axes[1, 1]
cm = confusion_matrix(y_test, y_pred_best)
im = ax5.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax5.set_title('混淆矩阵 (最优模型)')
tick_marks = np.arange(2)
ax5.set_xticks(tick_marks)
ax5.set_yticks(tick_marks)
ax5.set_xticklabels(['CN', 'AD'])
ax5.set_yticklabels(['CN', 'AD'])
ax5.set_ylabel('True Label')
ax5.set_xlabel('Predicted Label')

# 在混淆矩阵中添加数值
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax5.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=14)

# 图6: SVR 预测结果
ax6 = axes[1, 2]
ax6.scatter(y_test_r, y_pred_r_best, alpha=0.6, color='blue', edgecolors='k')
ax6.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax6.set_xlabel('True MMSE Score')
ax6.set_ylabel('Predicted MMSE Score')
ax6.set_title(f'SVR 预测结果 (MSE={mse_best:.3f})')
ax6.legend()

plt.tight_layout()
plt.savefig('svm_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ 综合可视化结果已保存为 'svm_comprehensive_analysis.png'")

# ==========================================
# 总结
# ==========================================
print("\n" + "="*60)
print("📊 SVM 分析总结")
print("="*60)
print(f"""
1. 对偶问题: 模型使用了 {len(support_indices)} 个支持向量 ({len(support_indices)/len(X_train)*100:.1f}%)
   - 对偶系数反映了每个支持向量对决策边界的重要性

2. 核函数选择: 
   - 最优核函数: {max(kernel_results, key=lambda x: kernel_results[x]['cv_mean'])}
   - RBF核在处理非线性边界时表现最佳

3. 软间隔参数 C:
   - C值越大，模型越复杂，支持向量越少
   - 需要通过交叉验证选择合适的C值避免过拟合

4. SVR回归:
   - 使用 {len(features_svr)} 个特征预测 MMSE
   - 最终 RMSE: {np.sqrt(mse_best):.3f}

5. 最优模型参数: {grid_search.best_params_}
   - 交叉验证准确率: {grid_search.best_score_:.4f}
""")
