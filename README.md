# 树模型变量拟合项目

这是一个使用树模型进行变量拟合的Python项目，支持决策树、随机森林、梯度提升树等多种树模型，适用于回归和分类任务。

## 项目结构

```
tree_model_fitting/
├── tree_fitting.py          # 主要的树模型拟合代码
├── generate_data.py         # 数据生成脚本
├── requirements.txt         # 项目依赖
├── README.md               # 项目说明文档
├── regression_data.csv     # 回归示例数据（运行后生成）
├── classification_data.csv # 分类示例数据（运行后生成）
├── timeseries_data.csv     # 时间序列示例数据（运行后生成）
├── categorical_data.csv    # 分类变量示例数据（运行后生成）
└── best_model.pkl         # 训练好的最佳模型（运行后生成）
```

## 功能特性

### 支持的树模型
- **决策树** (Decision Tree)
- **随机森林** (Random Forest)
- **梯度提升树** (Gradient Boosting)

### 支持的任务类型
- **回归任务**: 预测连续数值
- **分类任务**: 预测离散类别

### 主要功能
- 数据加载和预处理
- 多种树模型训练和比较
- 超参数调优
- 模型性能评估
- 特征重要性分析
- 模型保存和加载
- 数据可视化

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 生成示例数据

```bash
python generate_data.py
```

这将生成四种类型的示例数据：
- `regression_data.csv`: 回归数据
- `classification_data.csv`: 分类数据
- `timeseries_data.csv`: 时间序列数据
- `categorical_data.csv`: 包含分类变量的数据

### 2. 运行树模型拟合

```bash
python tree_fitting.py
```

### 3. 使用自定义数据

```python
from tree_fitting import TreeModelFitting

# 创建回归任务实例
regressor = TreeModelFitting(task_type='regression')

# 加载数据
regressor.load_data('your_data.csv', target_column='target')

# 数据预处理
regressor.preprocess_data()

# 训练模型
regressor.train_models()

# 超参数调优
regressor.hyperparameter_tuning('随机森林')

# 评估模型
regressor.evaluate_model()

# 绘制特征重要性
regressor.plot_feature_importance()

# 保存模型
regressor.save_model('my_model.pkl')
```

## 详细使用说明

### TreeModelFitting 类

#### 初始化
```python
# 回归任务
regressor = TreeModelFitting(task_type='regression')

# 分类任务
classifier = TreeModelFitting(task_type='classification')
```

#### 主要方法

1. **load_data(file_path, target_column=None)**
   - 加载CSV或Excel数据文件
   - 自动识别目标变量或手动指定

2. **preprocess_data(test_size=0.2, random_state=42)**
   - 处理缺失值
   - 编码分类变量
   - 分割训练集和测试集

3. **train_models()**
   - 训练多种树模型
   - 进行交叉验证
   - 自动选择最佳模型

4. **hyperparameter_tuning(model_name='随机森林')**
   - 对指定模型进行网格搜索调优
   - 支持决策树、随机森林、梯度提升树

5. **evaluate_model()**
   - 评估模型性能
   - 回归：MSE, RMSE, MAE, R²
   - 分类：准确率、分类报告、混淆矩阵

6. **plot_feature_importance(top_n=10)**
   - 绘制特征重要性图
   - 返回特征重要性DataFrame

7. **save_model(filepath)** / **load_model(filepath)**
   - 保存和加载训练好的模型

## 数据格式要求

### 输入数据格式
- 支持CSV和Excel文件
- 数据应为数值型或可编码的分类变量
- 目标变量可以是数值型（回归）或离散型（分类）

### 示例数据格式
```csv
feature_1,feature_2,feature_3,feature_4,feature_5,target
1.2,3.4,5.6,7.8,9.0,12.34
2.1,4.3,6.5,8.7,1.0,23.45
...
```

## 模型性能评估

### 回归任务
- **均方误差 (MSE)**: 预测值与真实值差的平方的平均值
- **均方根误差 (RMSE)**: MSE的平方根
- **平均绝对误差 (MAE)**: 预测值与真实值差的绝对值的平均值
- **R² 得分**: 模型解释的方差比例

### 分类任务
- **准确率**: 正确预测的样本比例
- **分类报告**: 包含精确率、召回率、F1分数
- **混淆矩阵**: 各类别的预测结果矩阵

## 特征重要性

树模型可以自动计算特征重要性，帮助理解哪些特征对预测最重要：
- 基于特征在树中的分裂次数和分裂质量
- 值越大表示特征越重要
- 可用于特征选择和模型解释

## 超参数调优

项目支持对以下模型进行超参数调优：

### 决策树
- `max_depth`: 树的最大深度
- `min_samples_split`: 分裂内部节点所需的最小样本数
- `min_samples_leaf`: 叶节点的最小样本数

### 随机森林
- `n_estimators`: 树的数量
- `max_depth`: 树的最大深度
- `min_samples_split`: 分裂内部节点所需的最小样本数
- `min_samples_leaf`: 叶节点的最小样本数

### 梯度提升树
- `n_estimators`: 树的数量
- `learning_rate`: 学习率
- `max_depth`: 树的最大深度
- `subsample`: 用于训练每棵树的样本比例

## 注意事项

1. **数据质量**: 确保输入数据质量良好，处理异常值和缺失值
2. **特征工程**: 根据业务需求进行适当的特征工程
3. **模型选择**: 不同模型适用于不同类型的数据和问题
4. **过拟合**: 注意避免过拟合，使用交叉验证和正则化
5. **计算资源**: 大数据集或复杂模型可能需要较多计算资源

## 扩展功能

### 添加新的树模型
```python
# 在 train_models() 方法中添加新模型
self.models['XGBoost'] = XGBRegressor(random_state=42)
```

### 自定义评估指标
```python
# 在 evaluate_model() 方法中添加自定义指标
from sklearn.metrics import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(self.y_test, y_pred)
```

## 常见问题

### Q: 如何处理分类变量？
A: 代码会自动将分类变量编码为数值，也可以手动进行独热编码。

### Q: 如何选择最佳模型？
A: 代码会自动进行交叉验证并选择得分最高的模型，也可以手动比较不同模型的性能。

### Q: 如何处理不平衡数据？
A: 可以在模型参数中设置 `class_weight='balanced'` 来处理不平衡数据。

### Q: 如何提高模型性能？
A: 可以尝试特征工程、超参数调优、集成方法或使用更复杂的模型。
