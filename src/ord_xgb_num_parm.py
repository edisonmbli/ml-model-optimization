"""
使用OrdinalEncoder处理分类特征，保留数值特征，配合XGBoost进行分类任务的交叉验证评估
使用sklearn的Pipeline、ColumnTransformer、RandomizedSearchCV实现，符合机器学习工程最佳实践
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import json
import pickle
import os
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV


def ensure_dir(directory):
    """
    确保目录存在，如果不存在则创建
    
    参数:
        directory: str, 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")


def convert_to_str(X):
    """
    将数据转换为字符串类型的函数
    
    参数:
        X: 输入数据
        
    返回:
        转换后的字符串类型数据
    """
    return X.astype(str)


def create_preprocessing_pipeline(numeric_features, categorical_features):
    """
    创建特征预处理流水线
    
    参数:
        numeric_features: 数值型特征列表
        categorical_features: 分类特征列表
        
    返回:
        ColumnTransformer: 组合了数值和分类特征处理的预处理器
    """
    # 创建数值特征的预处理流水线
    # 注意：树模型如XGBoost不需要标准化数值特征，因为它们对特征尺度不敏感
    numeric_transformer = make_pipeline(
        # 仅填充缺失值，无需标准化
        SimpleImputer(strategy='median')
    )
    
    # 创建分类特征的预处理流水线
    categorical_transformer = make_pipeline(
        # 步骤1: 将所有特征转换为字符串 (使用命名函数替代lambda)
        FunctionTransformer(convert_to_str),
        # 步骤2: 填充缺失值 
        SimpleImputer(strategy='constant', fill_value='NONE'),
        # 步骤3: 对分类特征进行序数编码
        OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    )
    
    # 使用ColumnTransformer组合不同类型特征的预处理流水线
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor


def optimize_hyperparameters(X, y, preprocessor, n_folds=5, random_state=42, n_iter=20, verbose=1):
    """
    使用RandomizedSearchCV优化XGBoost超参数
    
    参数:
        X: 特征矩阵
        y: 目标变量
        preprocessor: 特征预处理器
        n_folds: 交叉验证折数
        random_state: 随机种子
        n_iter: 随机搜索迭代次数
        verbose: 输出详细程度
        
    返回:
        dict: 最佳参数
        float: 最佳得分
        estimator: 训练好的最佳模型
    """
    # 创建模型流水线
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            n_jobs=-1,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss'  # 防止警告
        ))
    ])
    
    # 定义参数搜索空间
    param_distributions = {
        'classifier__eta': np.logspace(-3, 0, 1000),  # 学习率[0.001, 1.0]
        'classifier__gamma': np.logspace(-3, 2, 1000),  # 最小分裂损失[0.001, 100]
        'classifier__max_depth': np.arange(3, 15),  # 树的最大深度
        'classifier__min_child_weight': np.arange(1, 10),  # 最小子节点权重
        'classifier__lambda': np.logspace(-3, 3, 1000),  # L2正则化[0.001, 1000]
        'classifier__alpha': np.logspace(-3, 3, 1000),  # L1正则化[0.001, 1000]
        'classifier__subsample': np.linspace(0.5, 1.0, 100),  # 样本采样比例
        'classifier__colsample_bytree': np.linspace(0.5, 1.0, 100)  # 特征采样比例
    }
    
    # 设置交叉验证
    cv = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state
    )
    
    # 创建随机搜索
    random_search = RandomizedSearchCV(
        estimator=model_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=cv,
        verbose=verbose,
        random_state=random_state,
        n_jobs=-1,
        return_train_score=True
    )
    
    print("开始随机搜索超参数...")
    random_search.fit(X, y)
    print(f"最佳参数: {random_search.best_params_}")
    print(f"最佳AUC: {random_search.best_score_:.4f}")
    
    # 提取最佳参数（去掉classifier__前缀）
    best_params = {k.replace('classifier__', ''): v 
                  for k, v in random_search.best_params_.items()}
    
    return best_params, random_search.best_score_, random_search.best_estimator_


def evaluate_model(estimator, X, y, n_folds=5, random_state=42, verbose=1):
    """
    使用交叉验证评估模型性能
    
    参数:
        estimator: 训练好的模型
        X: 特征矩阵
        y: 目标变量
        n_folds: 交叉验证折数
        random_state: 随机种子
        verbose: 输出详细程度
        
    返回:
        ndarray: 每个折的AUC评分
    """
    # 使用StratifiedKFold确保分层交叉验证
    cv = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state
    )
    
    print("开始交叉验证...")
    
    # 使用cross_val_score计算AUC评分
    scores = cross_val_score(
        estimator=estimator, 
        X=X, 
        y=y, 
        cv=cv, 
        scoring='roc_auc',
        verbose=verbose,
        error_score='raise',
        n_jobs=-1
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


def run(n_folds=5, random_state=42, verbose=1, optimize=True, n_iter=20):
    """
    执行XGBoost模型的超参数优化和评估
    
    参数:
        n_folds: int, 交叉验证的折数，默认为5
        random_state: int, 随机种子，用于结果复现，默认为42
        verbose: int, 输出详细程度，默认为1
        optimize: bool, 是否执行超参数优化，默认为True
        n_iter: int, 随机搜索的迭代次数，默认为20
        
    返回:
        dict: 优化后的最佳参数（如果optimize=True）
        ndarray: 交叉验证的AUC评分
    """
    # 读取数据集
    df = pd.read_csv("../input/adult.csv")
    
    # 定义数值型特征列表
    numeric_features = [
        "fnlwgt", 
        "age", 
        "capital.gain", 
        "capital.loss", 
        "hours.per.week"
    ]
    
    # 映射目标变量为0和1
    target_mapping = {
        "<=50K": 0,
        ">50K": 1
    }
    df.loc[:, "income"] = df.income.map(target_mapping)
    
    # 确定特征列（除kfold、income外的所有列）
    features = [
        col for col in df.columns if col not in ("kfold", "income")
    ]
    
    # 确定分类特征（非数值特征）
    categorical_features = [col for col in features if col not in numeric_features]
    
    # 准备特征矩阵和目标变量
    X = df[features]
    y = df.income.values
    
    # 创建预处理流水线
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
    
    if optimize:
        # 执行超参数优化
        best_params, best_score, best_model = optimize_hyperparameters(
            X, y, preprocessor, n_folds, random_state, n_iter, verbose
        )
        
        # 评估最佳模型
        scores = evaluate_model(best_model, X, y, n_folds, random_state, verbose)
        return best_params, scores
    else:
        # 使用默认参数创建模型
        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', xgb.XGBClassifier(
                n_jobs=-1,
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                random_state=random_state
            ))
        ])
        
        # 评估默认模型
        scores = evaluate_model(model_pipeline, X, y, n_folds, random_state, verbose)
        return scores


def convert_numpy_types(obj):
    """
    递归地将NumPy类型转换为Python原生类型，使其可以被JSON序列化
    
    参数:
        obj: 要转换的对象（可以是字典、列表、NumPy类型等）
        
    返回:
        转换后的对象
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def save_params(params, filename='best_params.json'):
    """
    将最佳参数保存到JSON文件
    
    参数:
        params: dict, 要保存的参数
        filename: str, 文件名，默认为'best_params.json'
    """
    # 确保models目录存在
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    ensure_dir(models_dir)
    
    # 构建完整的文件路径
    filepath = os.path.join(models_dir, filename)
    
    # 转换NumPy类型为Python原生类型
    params_converted = convert_numpy_types(params)
    
    with open(filepath, 'w') as f:
        json.dump(params_converted, f, indent=4)
    print(f"参数已保存到 {filepath}")


def load_params(filename='best_params.json'):
    """
    从JSON文件加载参数
    
    参数:
        filename: str, 文件名，默认为'best_params.json'
        
    返回:
        dict: 加载的参数
    """
    # 构建完整的文件路径
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    filepath = os.path.join(models_dir, filename)
    
    with open(filepath, 'r') as f:
        params = json.load(f)
    print(f"已从 {filepath} 加载参数")
    return params


def save_model(model, filename='best_model.pkl'):
    """
    将模型保存到文件
    
    参数:
        model: 模型对象
        filename: str, 文件名，默认为'best_model.pkl'
    """
    # 确保models目录存在
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    ensure_dir(models_dir)
    
    # 构建完整的文件路径
    filepath = os.path.join(models_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"模型已保存到 {filepath}")


def load_model(filename='best_model.pkl'):
    """
    从文件加载模型
    
    参数:
        filename: str, 文件名，默认为'best_model.pkl'
        
    返回:
        已加载的模型对象
    """
    # 构建完整的文件路径
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    filepath = os.path.join(models_dir, filename)
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"已从 {filepath} 加载模型")
    return model


def train_final_model(X, y, params, preprocessor, random_state=42):
    """
    使用最佳参数在完整数据上训练最终模型
    
    参数:
        X: 特征矩阵
        y: 目标变量
        params: dict, 模型参数
        preprocessor: 预处理器
        random_state: int, 随机种子
        
    返回:
        训练好的模型
    """
    print("使用最佳参数在完整数据集上训练最终模型...")
    
    # 创建带有最佳参数的模型
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            n_jobs=-1,
            random_state=random_state,
            **params
        ))
    ])
    
    # 在完整数据集上训练
    model_pipeline.fit(X, y)
    
    return model_pipeline


def predict_test_data(model_path=None, params_path=None, test_file="../input/adult_test.csv"):
    """
    使用保存的模型或参数对测试数据进行预测并评估
    
    参数:
        model_path: str, 模型文件路径，如果为None则使用参数重新训练
        params_path: str, 参数文件路径，如果model_path为None则必须提供
        test_file: str, 测试数据文件路径
        
    返回:
        float: 测试集上的AUC
        ndarray: 预测的概率
    """
    # 读取测试数据
    try:
        df_test = pd.read_csv(test_file)
        print(f"已加载测试数据，形状: {df_test.shape}")
    except FileNotFoundError:
        print(f"错误: 找不到测试文件 {test_file}")
        return None, None
    
    # 定义数值型特征列表
    numeric_features = [
        "fnlwgt", 
        "age", 
        "capital.gain", 
        "capital.loss", 
        "hours.per.week"
    ]
    
    # 映射目标变量为0和1
    target_mapping = {
        "<=50K": 0,
        ">50K": 1,
        "<=50K.": 0,  # 处理测试集中可能的不同标签
        ">50K.": 1    # 处理测试集中可能的不同标签
    }
    
    # 确保income列存在
    if "income" not in df_test.columns:
        print("警告: 测试数据中没有'income'列，无法计算AUC")
        has_target = False
    else:
        df_test.loc[:, "income"] = df_test.income.map(target_mapping)
        has_target = True
    
    # 确定特征列
    features = [
        col for col in df_test.columns if col not in ("kfold", "income")
    ]
    
    # 确定分类特征
    categorical_features = [col for col in features if col not in numeric_features]
    
    # 准备测试特征
    X_test = df_test[features]
    
    if has_target:
        y_test = df_test.income.values
    
    # 优先使用保存的模型
    if model_path is not None:
        try:
            model = load_model(model_path)
        except FileNotFoundError:
            print(f"错误: 找不到模型文件 {model_path}")
            return None, None
    
    # 如果没有模型，使用参数重新训练
    elif params_path is not None:
        try:
            # 加载参数
            params = load_params(params_path)
            
            # 读取训练数据重新训练模型
            df_train = pd.read_csv("../input/adult.csv")
            df_train.loc[:, "income"] = df_train.income.map(target_mapping)
            
            # 准备训练特征和目标
            X_train = df_train[features]
            y_train = df_train.income.values
            
            # 创建预处理流水线
            preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
            
            # 训练最终模型
            model = train_final_model(X_train, y_train, params, preprocessor)
            
        except FileNotFoundError:
            print(f"错误: 找不到参数文件 {params_path}")
            return None, None
    else:
        print("错误: 必须提供model_path或params_path")
        return None, None
    
    # 预测测试数据
    print("开始对测试数据进行预测...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 如果有目标变量，计算并打印AUC
    if has_target:
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        print(f"\n测试集AUC: {auc:.4f}")
        
        # 计算分类报告
        y_pred = (y_pred_proba > 0.5).astype(int)
        classification_report = metrics.classification_report(y_test, y_pred)
        print("\n分类报告:")
        print(classification_report)
        
        # 混淆矩阵
        conf_matrix = metrics.confusion_matrix(y_test, y_pred)
        print("\n混淆矩阵:")
        print(conf_matrix)
        
        # 计算其他指标
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)
        
        print("\n其他评估指标:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        return auc, y_pred_proba
    else:
        print("无法计算测试集AUC（缺少目标变量）")
        return None, y_pred_proba


if __name__ == "__main__":
    # 1. 执行超参数优化和模型评估，n_iter设为较小值以加快示例运行
    best_params, scores = run(n_folds=5, n_iter=10, verbose=1)
    
    # 2. 保存最佳参数
    save_params(best_params)
    
    # 3. 读取训练数据
    df_train = pd.read_csv("../input/adult.csv")
    
    # 4. 处理数据准备模型训练
    numeric_features = ["fnlwgt", "age", "capital.gain", "capital.loss", "hours.per.week"]
    target_mapping = {"<=50K": 0, ">50K": 1}
    df_train.loc[:, "income"] = df_train.income.map(target_mapping)
    features = [col for col in df_train.columns if col not in ("kfold", "income")]
    categorical_features = [col for col in features if col not in numeric_features]
    X_train = df_train[features]
    y_train = df_train.income.values
    
    # 5. 创建预处理流水线
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
    
    # 6. 使用最佳参数在全量数据上训练最终模型
    final_model = train_final_model(X_train, y_train, best_params, preprocessor)
    
    # 7. 保存模型
    save_model(final_model)
    
    # 8. 在测试集上评估模型
    # test_auc, test_preds = predict_test_data(model_path="best_model.pkl")
    
    print("\n完整流程执行完毕！")