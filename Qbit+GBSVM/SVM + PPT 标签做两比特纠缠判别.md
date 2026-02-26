## 一 -- 用 PPT 判据给量子态打标签 (y)
1. 对每个两比特密度矩阵 $\rho$
	1. 做部分转置: $\rho^{T_A}$ 或 $\rho^{T_B}$（代码把 $4 \times 4$ reshape 成 $2 \times 2 \times 2 \times 2$ 再交换轴实现）
	2. 数值上取厄米部分: $(\rho^T + (\rho^T)^\dagger) / 2$，保证特征值稳定
	3. 计算最小特征值 $\lambda_{\min}$
2. 判别规则 (两比特 $2 \times 2$ 情况下是充要条件)
	1. 若部分转置后的 $\lambda_{\min} < 0$ (考虑容差 $\text{tol}$)，则纠缠 $(y=1)$
	2. 否则 可分 $(y=0)$
3. 对应实现见 `ppt.py`
## 二 --  把量子态变成可学习的数值特征 (X)
1. 对每个密度矩阵$\rho$ 计算 15 维泡利矩阵的期望值向量
2. 两比特泡利矩阵张量基共有 16 个算符(含$I\otimes I$) ,去掉常数项$I\otimes I$后得到 15 维, 得到 `X.shape =(N,15)`, 每行是一个量子态的特征
3. 这步由 `pauli_expectation_15(rho)` 完成, 见 `features.py`
## 三 -- 生成数据集并保存
1. 一族 Werner-like 态（参数 λ 随机采样）
   $$
   \rho(\lambda)=\lambda\left|\Phi^+\right\rangle\langle\Phi^+|+\left(1-\lambda\right)\frac I4,\quad\lambda\in[0,1]
   $$
   其中：$\left | \Phi ^+ \right \rangle = \frac {\left | 00\right \rangle + \left | 11\right \rangle }{\sqrt {2}}$ (Bell 态), $\frac{I}{4}$是两比特的最大混合态
2. 随机混合态（Ginibre 方式采样）
	1. $G\in\mathbb{C}^{4\times4},\quad A=GG^\dagger,\quad\rho=\frac{A}{\mathrm{Tr}(A)}$
	2. $G$ 是随机复高斯矩阵
3. 对每个样本:
	- `X_i = pauli_expectation_15(rho_i)`
	- `y_i = 1 if is_entangled_ppt(rho_i) else 0`
4. 拼成矩阵 X 和标签 y, 保存为 entanglement_2q.npz (含 meta)。
- 见 `dataset.py`
## 四 --  训练 SVM 基线模型（带标准化）
1. 划分训练/测试集: `train_test_split(..., stratify=y) `保持类别比例一致
2. 用 `Pipeline: StandardScaler() + SVC(kernel='rbf')`
   如果不标准化, SVM 的距离尺度不合理
3. 用 GridSearchCV 在**训练集**做 5 折交叉验证搜索
	1. 超参数搜索: 
		1. $C$：惩罚/正则强度（影响边界软硬）C越大越容易过拟合
		2. $\gamma$：RBF 核的宽度（影响边界复杂度）
	2. 5 折交叉验证（5-fold CV）:把训练集分成 5 份（每份约 20%）
		1. 第 1 次：用 4 份训练、1 份验证
		2. 第 2 次：换另一份做验证
		3. ...
		4. 共 5 次，每一份都当过一次验证集
4. GridSearchCV 
	1. 列出参数网格里所有组合, 16 组 C×gamma 
	2. 对每一组参数
		1. 做 5 折交叉验证
		2. 计算每一折的评分 `scoring="roc_auc" `
			1. `AUC = 0.5`：和随机猜差不多
			2. `AUC = 1.0`：完美区分两类
		3. 把 5 折得分求平均，得到该参数组的“CV 平均分”
		4. 选出平均分最高的那组参数，称为 `best_params_`
		5. `refit=True` 的，会用“整份训练集”再训练一次最佳参数的模型，得到` best_estimator_`
## 五 -- 参数与输出结果
1. 阈值参数: Decision threshold == 0.7 (纠缠的误报更少，但会漏掉一部分纠缠)
2. 基于粒球支持向量机的量子纠缠探测研究