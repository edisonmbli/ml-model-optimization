"""
创建分层k折交叉验证数据集
"""
import pandas as pd
from sklearn import model_selection


if __name__ == "__main__":
    # 读取训练数据
    df = pd.read_csv("../input/cat_train.csv")
    
    # 创建一个名为kfold的新列并用-1填充
    df["kfold"] = -1
    
    # 随机打乱数据行
    df = df.sample(frac=1).reset_index(drop=True)
    
    # 获取标签
    y = df.target.values
    
    # 初始化model_selection模块中的StratifiedKFold类
    kf = model_selection.StratifiedKFold(n_splits=5)
    
    # 填充新的kfold列
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
    
    # 保存带有kfold列的新CSV文件
    df.to_csv("../input/cat_train_folds.csv", index=False)
    
    print("分层k折交叉验证数据集已创建并保存到 ../input/cat_train_folds.csv")