"""
使用OrdinalEncoder和XGBoost进行分类任务的交叉验证评估
使用sklearn的cross_val_score简化流程，保持分层K折交叉验证
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold


def run(n_folds=5, random_state=42, verbose=1):
    """
    使用cross_val_score执行序数编码XGBoost的交叉验证
    
    参数:
        n_folds: int, 交叉验证的折数，默认为5
        random_state: int, 随机种子，用于结果复现，默认为42
        verbose: int, 输出详细程度，默认为1
        
    返回:
        ndarray: 每个折的AUC评分
    """
    # 读取数据集
    df = pd.read_csv("../input/cat_train.csv")
    
    # 确定特征列（除id、target外的所有列）
    features = [
        col for col in df.columns if col not in ("id", "target", "kfold")
    ]
    
    # 准备特征矩阵和目标变量
    X = df[features]
    y = df.target.values
    
    # 定义将数据转换为字符串的函数
    def convert_to_string(X):
        return X.astype(str)

    # 创建分类特征的预处理流水线
    categorical_transformer = make_pipeline(
        # 步骤1: 将所有特征转换为字符串    
        FunctionTransformer(convert_to_string),
        # 步骤2: 填充缺失值 
        SimpleImputer(strategy='constant', fill_value='NONE'),
        # 步骤3: 对分类特征进行序数编码
        OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    )
    
    # 使用ColumnTransformer应用预处理流水线到指定特征列
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, features)
        ],
        remainder='drop'  # 丢弃未指定的列
    )
    
    # 创建完整的机器学习Pipeline
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            n_jobs=-1,
            max_depth=7,
            n_estimators=200,
            random_state=random_state
        ))
    ])
    
    # 使用StratifiedKFold确保分层交叉验证
    cv = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state
    )
    
    print("开始交叉验证...")
    
    # 使用cross_val_score计算AUC评分
    scores = cross_val_score(
        estimator=model_pipeline, 
        X=X, 
        y=y, 
        cv=cv, 
        scoring='roc_auc',
        verbose=verbose,
        error_score='raise',  # 错误时抛出异常，便于调试
        n_jobs=-1  # 使用所有可用CPU并行计算
    )
    
    # 打印每个折的AUC评分
    for fold, score in enumerate(scores):
        print(f"Fold {fold}: AUC = {score:.4f}")
    
    # 计算统计指标
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    # 打印汇总统计信息
    print("\n====== 交叉验证结果汇总 ======")
    print(f"平均AUC: {mean_score:.4f} ± {std_score:.4f}")
    print(f"最小AUC: {np.min(scores):.4f}")
    print(f"最大AUC: {np.max(scores):.4f}")
    print("===============================")
    
    return scores


if __name__ == "__main__":
    run(n_folds=5)