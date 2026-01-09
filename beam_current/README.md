# 束流数据处理项目

这是一个专门用于处理和分析束流数据的独立项目。

## 项目结构

```
beam_current/
├── src/                    # 源代码目录
│   ├── linear.py          # 线性回归模型
│   ├── tree_model.py      # 树模型（决策树、随机森林、XGBoost）
│   ├── mlp.py             # 多层感知机模型
│   ├── LSTM.py            # LSTM模型
│   ├── PLS.py             # 偏最小二乘模型
│   └── data_process.py    # 数据处理和分析脚本
├── data/                   # 数据目录
│   ├── raw/               # 原始数据
│   │   ├── 束流.csv
│   │   └── 束流.xlsx
│   └── processed/         # 处理后的数据
│       ├── split_data.npz
│       └── scaler.pkl
├── models/                 # 保存的模型文件
├── result/                 # 结果输出目录
├── requirements.txt        # 项目依赖
└── README.md              # 项目说明文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 数据分析

首先运行数据分析脚本，了解数据特征：

```bash
python src/data_process.py
```

这将生成：
- 束流数据统计分析
- 特征相关性分析
- 异常值检测
- 数据可视化图表

### 2. 训练模型

#### 线性回归
```bash
python src/linear.py
```

#### 树模型（决策树、随机森林、XGBoost）
```bash
python src/tree_model.py
```

#### MLP（多层感知机）
```bash
python src/mlp.py
```

#### LSTM
```bash
python src/LSTM.py
```

#### PLS（偏最小二乘）
```bash
python src/PLS.py
```

## 数据格式

### 输入数据格式
- CSV 或 Excel 文件
- 必须包含以 `feature` 开头的特征列
- 目标变量列名可以是 `束流` 或 `target`

### 示例数据格式
```csv
feature1,feature2,feature3,...,束流
1.2,3.4,5.6,...,12.34
2.1,4.3,6.5,...,23.45
...
```

## 输出结果

所有结果将保存在 `result/` 目录下，包括：
- 训练好的模型文件（.pkl）
- 模型评估指标（.txt, .json）
- 特征重要性分析（.csv）
- 可视化图表（.png）

## 模型说明

### 线性回归
- 快速训练，适合基线模型
- 结果保存在 `result/linear/` 目录

### 树模型
- 支持决策树、随机森林、XGBoost
- 自动进行超参数调优
- 结果保存在 `result/tree_model/` 目录

### MLP
- 多层感知机神经网络
- 需要数据标准化
- 结果保存在 `result/mlp/` 目录

### LSTM
- 长短期记忆网络
- 适合时间序列数据
- 结果保存在 `result/lstm/` 目录

### PLS
- 偏最小二乘回归
- 适合高维数据
- 结果保存在 `result/pls/` 目录

## 注意事项

1. **数据路径**：确保原始数据文件位于 `data/raw/` 目录
2. **结果目录**：每次运行会自动创建带时间戳的结果目录
3. **模型保存**：训练好的模型会保存在 `models/` 目录
4. **依赖安装**：首次使用前请安装所有依赖包

## 常见问题

### Q: 如何更换数据文件？
A: 将新的数据文件放在 `data/raw/` 目录，并修改代码中的数据文件名。

### Q: 如何调整模型参数？
A: 直接编辑对应的 Python 文件，修改模型初始化参数。

### Q: 结果保存在哪里？
A: 所有结果保存在 `result/` 目录下，按模型类型和时间戳组织。

