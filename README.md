
# Dry Bean Dataset 干豆多分类项目（博士级复现实验说明）

本仓库为“干豆（Dry Bean）”形态学特征多分类项目完整实现，面向学术复现与工程化展示。代码覆盖数据清洗、特征工程、三类经典/拓展模型训练与对比评估、噪声鲁棒性测试、实验图表与论文生成。

## 一、数据概述
- 数据来源：Dry_Bean_Dataset_Dirty（已分为 train/val/test），数据包含若干数值形态特征（长度、宽度、面积、周长、形状指标等）与类别标签 `Class`。
- 数据质量：含字符脏值（如 `?`、含单位字符串）、重复样本、标签噪声（大小写/字符错误）、部分特征存在缺失。
- 存放路径：`data/Dry_Bean_Dataset_Dirty_train.csv`、`data/..._val.csv`、`data/..._test.csv`。

## 二、数据处理（可复现且可追踪）
处理步骤按顺序且每步可切换：
1. 读取：使用 `pd.read_csv(..., na_values=['?'])` 将 `?` 视作缺失值；合并用于 EDA。
2. 清洗：去除重复样本；字符串数值清理（去单位）、`pd.to_numeric(errors='coerce')`；中位数填充数值缺失、众数填充类别缺失；3σ 剔除显著异常值。
3. 标签净化：统一大小写、常见替换（如 `0`→`O` 类误写）并使用 `LabelEncoder` 编码为整数标签。
4. 特征工程：`VarianceThreshold` 去低方差特征；`StandardScaler` 标准化；`PCA(n_components=0.95)` 保留 95% 方差。
5. 鲁棒性准备：在评估阶段对测试集添加不同强度高斯噪声（std=0.1,0.3,0.5）以测试模型稳定性。

以上逻辑实现于 `src/preprocess.py`，每一步均记录可复现随机种子与中间尺寸。

## 三、模型实现
实现并比较以下模型（具体超参见 `src/models.py`）：
- `LogisticRegression`：使用 `solver='saga'`，实现多轮训练以记录训练损失曲线（用于绘制 loss 曲线）。
- `RandomForestClassifier`：树模型基线，默认 sklearn 参数。
- `XGBClassifier`（XGBoost）：作为课外拓展。注意：根据 xgboost 版本差异，sklearn 接口可能不支持直接返回历时 `evals_result`，代码中已做兼容处理（若不支持则不记录逐轮 loss，但模型仍会训练并评估）。

训练、评估、绘图等封装于 `src/models.py` 与 `src/train_eval.py`，并由统一命令入口驱动。

## 四、评估指标与输出
- 自动保存：`output/model_compare.csv`（训练/验证/测试精度、过拟合差、推理时间/样本、不同噪声下精度等）。
- 自动绘图：`output/acc_compare.png`、`output/noise_robust.png`、`output/overfit_gap.png`、`output/loss_curves.png`（若有训练历史）。
- 论文文档：使用 `md_to_docx.py` 将 `论文.md` 转换成 `论文.docx`（包含图表引用并嵌入图片）。

## 五、使用方式（命令行、无 UI）
所有操作均通过命令行完成且不弹出图形界面（Matplotlib 后端已设为 `Agg`，适合服务器/无头环境）：

安装依赖：
```bash
pip install -r requirements.txt
```

统一入口（根目录）：
```bash
python run_cli.py all       # 运行完整流程：加载->清洗->训练->评估->生成论文
python run_cli.py train     # 仅训练阶段（生成模型文件/训练历史）
python run_cli.py eval      # 使用训练模型进行评估并生成图表/表格
python run_cli.py report    # 生成论文 docx（运行 md_to_docx.py）
```

也可以直接运行旧入口（仍受支持）：
```bash
python src/run.py
```

## 六、复现实验建议（博士级要求）
1. 环境记录：建议使用虚拟环境并记录 `pip freeze > requirements.txt`（本仓库已包含一个基础 `requirements.txt`）。
2. 随机性控制：在需要严格复现时，设置全局随机种子（`numpy.random.seed` 与 sklearn 的 `random_state`）。
3. 版本敏感说明：XGBoost 的 sklearn wrapper 在不同版本中对 `eval_metric` / `evals_result` 的支持存在差异；若需完整的逐轮 loss，请改用 xgboost 原生 API（`xgboost.DMatrix` + `xgboost.train`），仓库可在 `src/models.py` 中轻松替换。

## 七、文件结构（简要）
```
README.md
requirements.txt
data/  # 原始三份 csv
output/  # 所有图表与表格输出
src/
	data_loader.py
	preprocess.py
	models.py
	train_eval.py
	run.py
run_cli.py  # 统一命令行入口
md_to_docx.py
论文.md / 论文.docx
```

## 八、下一步（可选）
- 若需要更严格的博士级补充：添加实验设计章节、统计显著性检验（如 McNemar Test）、混淆矩阵与类别不平衡分析、特征重要性与可解释性（SHAP）分析、超参搜索（Grid/Random/Bayesian）并记录实验日志（建议使用 MLflow）。

如需我代为添加任一扩展（例如切换 XGBoost 到原生 API 以记录逐轮 loss、添加 SHAP 分析或实验日志），回复要做的项目与我将自动实现并提交到仓库。
