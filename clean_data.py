import pandas as pd

def clean_data(X):
    Xtest = pd.DataFrame(X)

    # 数据清洗，替换空值为0.0，并记录空值所在的行
    replace_count_X = 0
    for col in Xtest.columns:
        for idx, value in enumerate(Xtest[col]):
            if pd.isna(value):
                Xtest.at[idx, col] = 0.0
                replace_count_X += 1

    X = Xtest.to_numpy()

    return X