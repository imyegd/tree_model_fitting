# 束位监测数据处理项目

这是一个专门用于处理和分析束位监测数据的独立项目。

## 项目结构

```
beam_position/
├── src/                    # 源代码目录
│   ├── common/            # 共享工具模块
│   │   └── utils.py       # 通用工具函数
│   ├── 01_data_prep.py   # 数据预处理脚本
│   ├── 02_baseline_models.py  # 基线模型（线性回归、随机森林、XGBoost）
│   ├── position.py        # 位置数据处理
│   ├── plot.py           # 数据可视化
│   ├── XGboost.py        # XGBoost模型
│   └── PLS_test.py       # PLS测试脚本
├── data/                   # 数据目录
│   ├── raw/               # 原始数据
│   │   ├── 束位数据.csv
│   │   ├── 束位数据.xlsx
│   │   ├── 束位监测数据.csv
│   │   └── 束位监测数据.xlsx
│   └── processed/         # 处理后的数据
│       ├── X_train_static_random.csv
│       ├── X_test_static_random.csv
│       ├── y_train_static_random.csv
│       └── y_test_static_random.csv
├── models/                 # 保存的模型文件
│   └── pls_model.pkl
├── result/                 # 结果输出目录
│   ├── baseline_models/   # 基线模型结果
│   ├── linear/            # 线性回归结果
│   └── position_analysis/ # 位置分析结果
├── requirements.txt        # 项目依赖
└── README.md              # 项目说明文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 数据预处理

首先运行数据预处理脚本，生成训练和测试数据：

```bash
python src/01_data_prep.py
```

这将：
- 加载束位监测数据和束位数据
- 进行时间对齐和特征工程
- 计算 Delta_X 和 Delta_Y（束位变化量）
- 生成训练集和测试集
- 保存处理后的数据到 `data/processed/` 目录

### 2. 训练基线模型

运行基线模型训练脚本：

```bash
python src/02_baseline_models.py
```

这将训练并评估：
- 线性回归（Linear Regression）
- 随机森林（Random Forest）
- XGBoost

结果保存在 `result/baseline_models/` 目录。

### 3. 其他模型和工具

#### XGBoost 模型
```bash
python src/XGboost.py
```

#### PLS 测试
```bash
python src/PLS_test.py
```

#### 数据可视化
```bash
python src/plot.py
```

这将生成：
- 束位轨迹图
- 时间序列分析图
- 异常值检测图

## 数据格式

### 输入数据格式

#### 束位监测数据 (束位监测数据.csv)
- 必须包含 `时间` 列（datetime格式）
- 必须包含以 `feature` 开头的特征列

#### 束位数据 (束位数据.csv)
- 必须包含 `时间` 列（H:M:S格式）
- 必须包含 `束位X` 和 `束位Y` 列

### 数据预处理流程

1. **时间对齐**：将束位数据的时间戳与束位监测数据对齐
2. **特征工程**：计算时间窗口内的统计特征（均值、标准差、最小值、最大值、中位数）
3. **目标变量计算**：计算 Delta_X 和 Delta_Y（相邻时间点的束位差值）
4. **数据划分**：按时间顺序划分训练集和测试集

## 输出结果

### 数据预处理输出
- `data/processed/X_train_static_random.csv` - 训练集特征
- `data/processed/X_test_static_random.csv` - 测试集特征
- `data/processed/y_train_static_random.csv` - 训练集目标（Delta_X, Delta_Y）
- `data/processed/y_test_static_random.csv` - 测试集目标

### 模型结果
- 模型文件（.pkl）
- 评估指标（R², MSE等）
- 预测对比图
- 特征重要性分析

## 模型说明

### 基线模型
- **线性回归**：快速训练，适合基线对比
- **随机森林**：集成学习，对多输出回归支持良好
- **XGBoost**：梯度提升，性能优秀

### 评估指标
- **R² (加权平均)**：整体模型拟合度
- **MSE**：均方误差
- **Delta_X R²**：X方向预测精度
- **Delta_Y R²**：Y方向预测精度

## 注意事项

1. **数据顺序**：必须先运行 `01_data_prep.py` 生成处理后的数据
2. **时间格式**：确保时间数据格式正确
3. **路径设置**：所有路径都是相对于项目根目录的
4. **依赖安装**：首次使用前请安装所有依赖包

## 常见问题

### Q: 数据预处理失败怎么办？
A: 检查原始数据文件是否存在，时间格式是否正确。

### Q: 如何调整特征工程方法？
A: 编辑 `src/01_data_prep.py` 中的 `feature_engineering_and_padding` 函数。

### Q: 如何添加新的模型？
A: 参考 `src/02_baseline_models.py` 的格式，添加新的模型训练代码。

### Q: 结果保存在哪里？
A: 所有结果保存在 `result/` 目录下，按模型类型组织。

