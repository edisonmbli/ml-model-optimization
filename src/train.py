import argparse
import os
import joblib
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import config
import model_dispatcher

def run(model_name, n_folds=5):
    # 读取训练数据
    df = pd.read_csv(config.TRAINING_FILE)
    
    # 准备特征和目标变量
    X = df.drop("label", axis=1).values
    y = df.label.values
    
    # 获取模型
    model = model_dispatcher.models[model_name]
    
    # 使用cross_val_score进行交叉验证
    scores = cross_val_score(model, X, y, cv=n_folds, scoring='accuracy')
    
    # 输出每个fold的准确率
    for fold, accuracy in enumerate(scores):
        print(f"Fold={fold}, Accuracy={accuracy}")
    
    # 输出平均准确率
    print(f"Average Accuracy: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")
    
    # 使用全部数据训练最终模型
    model.fit(X, y)
    
    # 保存模型
    joblib.dump(
        model,
        os.path.join(config.MODEL_OUTPUT, f"{model_name}_final.bin")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型名称，必须在model_dispatcher中定义"
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="交叉验证的折数，默认为5"
    )
    args = parser.parse_args()
    run(
        model_name=args.model,
        n_folds=args.folds
    )