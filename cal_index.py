import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


def calculate_nse(observed, simulated):
    # 计算观测数据的平均值
    mean_observed = np.mean(observed)

    # 计算 NSE
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - mean_observed) ** 2)

    nse = 1 - numerator / denominator

    return nse

def cal_index(y_true,y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # 计算平均绝对误差(MAE)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    # 计算均方根误差(RMSE)
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)



    # 计算相关系数(CC)

    # 计算均值
    A = y_true_flat
    B = y_pred_flat
    mean_A = np.mean(y_true_flat)
    mean_B = np.mean(y_pred_flat)

    # 计算协方差
    covariance = np.sum((A - mean_A) * (B - mean_B)) / A.shape[0]

    # 计算标准差
    std_A = np.sqrt(np.sum((A - mean_A) ** 2) / A.shape[0])
    std_B = np.sqrt(np.sum((B - mean_B) ** 2) / B.shape[0])

    # 计算相关系数
    correlation_coefficient = covariance / (std_A * std_B)

    # 计算确定性系数(NSE)
    NSE = calculate_nse(y_true_flat,y_pred_flat)

    print(f'MAE: {mae}')
    print(f'RMSE: {rmse}')
    print(f'CC: {correlation_coefficient}')
    print(f'NSE: {NSE}')








