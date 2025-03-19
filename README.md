# 机器学习模型优化实践

这个项目展示了如何使用scikit-learn和XGBoost构建和优化机器学习模型，重点关注特征编码方法和超参数调优。

## 项目结构

```
.
├── input/                # 数据文件目录
├── models/               # 保存模型和参数的目录
├── notebooks/            # Jupyter notebooks
├── src/                  # 源代码
│   ├── lbl_rf.py         # 使用LabelEncoder和随机森林的实现
│   ├── ohe_logres.py     # 使用OneHotEncoder和逻辑回归的实现
│   ├── ord_xgb.py        # 使用OrdinalEncoder和XGBoost的实现
│   └── ord_xgb_num.py    # 混合特征(数值+分类)的XGBoost实现
└── requirements.txt      # 项目依赖
```

## 安装指南

1. 克隆仓库:
```bash
git clone https://github.com/你的用户名/机器学习模型优化实践.git
cd 机器学习模型优化实践
```

2. 创建和激活虚拟环境:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

3. 安装依赖:
```bash
pip install -r requirements.txt
```

## 使用方法

### 数据准备

1. 将Adult数据集(或您自己的数据集)放在`input`目录下

### 运行模型

```bash
# 运行基于OrdinalEncoder的XGBoost超参数优化
python src/ord_xgb_num.py
```

### 模型结果

模型训练后，会在`models`目录下生成:
- `best_params.json`: 存储最佳超参数
- `best_model.pkl`: 存储训练好的模型

## 主要功能

- 使用Pipeline和ColumnTransformer进行数据预处理
- 使用不同编码方法处理分类特征
- 使用RandomizedSearchCV进行超参数优化
- 模型评估和结果可视化

## 许可证

MIT