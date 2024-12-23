from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from clean_data import clean_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from global_var import *
import csv
from cal_index import cal_index

# 加载模型，使用model_filename,或者使用任意一次以往记录
model = load_model('model2_20240406_022837.h5',compile=False)

# 导入预测降雨系列\径流预测数据\误差系列
rainfall_pre = 'C:\\Users\\25083\\Desktop\\神经网络训练总文件夹\\data_wanzhen_2\\rain_pred.csv'
flow_pre = 'Q_result_pred.csv'
e_pre = 'e_import.csv'

data_rainfall = np.genfromtxt(rainfall_pre, delimiter=',', skip_header=True)
data_flow_pre = np.genfromtxt(flow_pre, delimiter=',', skip_header=True)
data_e = np.genfromtxt(e_pre, delimiter=',', skip_header=True)

X1 = clean_data(data_rainfall)
X2 = clean_data(data_flow_pre)
Y = clean_data(data_e)
origin_Y = Y
# 归一化处理
 #创建 MinMaxScaler 对象
scaler_X1 = MinMaxScaler()
scaler_X2 = MinMaxScaler()
scaler_Y = MinMaxScaler()

# 对 X 和 Y 进行归一化
X1 = scaler_X1.fit_transform(X1)
X2 = scaler_X2.fit_transform(X2)
Y = scaler_Y.fit_transform(Y)


# 定义循环预测
for i in range(0,forecast_period_2):
    past_X1_value = X1[18 + i - time_delay_autoregressive_2:18 + 1 + i]
    past_X2_value = X2[18 + i - time_delay_autoregressive_2:18 + 1 + i]
    past_Y_value = Y[18 + i - time_delay_autoregressive_2 : 18 + i]

    # 在Y中最后一行补全
    last_value = past_Y_value[-1]
    new_Y = np.append(past_Y_value,last_value)

    reshaped_array = new_Y.reshape((time_delay_autoregressive_2+1, 1))
    reshaped_array = np.tile(reshaped_array, 8)
    # 转换形状
    input_X1 = past_X1_value[np.newaxis, :, :]

    # 将数组的列复制到新的维度，使其变为（13，8）
    input_X2 = np.tile(past_X2_value, 8)
    input_X2 = input_X2[np.newaxis, :, :]

    input_Y = reshaped_array[np.newaxis, :, :]

    # 使用加载好的模型进行预测
    Y_predict = model.predict([input_X1, input_X2, input_Y])
    #print(Y_predict,i)

    Y[18+i][0] = Y_predict[0][0]

# 输出展示结果
show_Y = Y[18:18+forecast_period_2-1+1]
# 返归一化
result = scaler_Y.inverse_transform(show_Y)

# 绘图
# 生成一维数组从1到预见期
array_values = np.arange(1, forecast_period_2+1)
# 将数组形状调整为（预见期，1）
X_axis = array_values.reshape((forecast_period_2, 1))
origin_Y = origin_Y[18:18+forecast_period_2]
# 绘制图形
plt.plot(X_axis, origin_Y, label='origin')
plt.plot(X_axis, result, label='predict')
print(result.shape)
print(origin_Y)

# 将二维数组转换为DataFrame
df = pd.DataFrame(result, columns=['Column1'])

# 将DataFrame写入CSV文件
df.to_csv('printe.csv', index=False)
# 添加图例
plt.legend()
# 显示图形
plt.show()