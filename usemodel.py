from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from clean_data import clean_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from global_var import *
import csv
from cal_index import cal_index

time_delay = time_delay_autoregressive

# 加载模型，使用model_filename,或者使用任意一次以往记录
model = load_model('model_20240411_221223.h5',compile=False)

# 导入用于预测的组合数据，拼接X和Y
# 加载CSV文件，并进行数据清洗
# pre_rainfall = 'pre_rain.csv'
# pre_flow = 'pre_flow.csv'

pre_rainfall = 'C:\\Users\\25083\\Desktop\\神经网络训练总文件夹\\data_wanzhen\\predict_1_rain.csv'
pre_flow = 'C:\\Users\\25083\\Desktop\\神经网络训练总文件夹\\data_wanzhen\\predict_1_flow.csv'

data_rainfall = np.genfromtxt(pre_rainfall, delimiter=',', skip_header=True)
data_flow = np.genfromtxt(pre_flow, delimiter=',', skip_header=True)

X = clean_data(data_rainfall)
Y = clean_data(data_flow)
origin_Y = Y

# 归一化处理
 #创建 MinMaxScaler 对象
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

# 对 X 和 Y 进行归一化
X = scaler_X.fit_transform(X)
Y = scaler_Y.fit_transform(Y)

# 定义循环预测
for i in range(0,forecast_period):
    past_X_value = X[18+i-time_delay:18+1+i]
    past_Y_value = Y[18+i-time_delay:18+i]

    # 在Y中最后一行补全
    last_value = past_Y_value[-1]
    new_Y = np.append(past_Y_value,last_value)

    reshaped_array = new_Y.reshape((time_delay_autoregressive+1, 1))
    reshaped_array = np.tile(reshaped_array, 8)
    # 转换形状
    input_X = past_X_value[np.newaxis, :, :]

    input_Y = reshaped_array[np.newaxis, :, :]

    # 使用加载好的模型进行预测
    Y_predict = model.predict([input_X,input_Y])
    #print(Y_predict,i)

    Y[18+i][0] = Y_predict[0][0]

# 输出展示结果
show_Y = Y[18:18+forecast_period-1+1]
# 返归一化
result = scaler_Y.inverse_transform(show_Y)

# 绘图
# 生成一维数组从1到预见期
array_values = np.arange(1, forecast_period+1)
# 将数组形状调整为（预见期，1）
X_axis = array_values.reshape((forecast_period, 1))
origin_Y = origin_Y[18:18+forecast_period]

# # 绘制图形
plt.plot(X_axis, origin_Y, label='origin')
plt.plot(X_axis, result, label='predict')
# # 添加图例
plt.legend()
# # 显示图形
plt.show()

# 计算指标
cal_index(origin_Y, result)

# 得到误差e系列
e = origin_Y - result
csv_file_path_1 = '2_e.csv'
np.savetxt(csv_file_path_1, e, delimiter=',', fmt='%.6f', header='e', comments='')

# 存储origin数据
csv_file_path_2 = '2_origin.csv'
np.savetxt(csv_file_path_2, origin_Y, delimiter=',', fmt='%.6f', header='origin', comments='')

# 存储预测result的结果
csv_file_path_3 = '2_result.csv'
np.savetxt(csv_file_path_3, result, delimiter=',', fmt='%.6f', header='result', comments='')

# 获取同等位置的P预测值用作新网络输入（产生新的CSV文件）
input_file = 'C:\\Users\\25083\\Desktop\\神经网络训练总文件夹\\data_wanzhen\\train_1_rain.csv'
output_file = 'output_input_pre_rain.csv'

rows_to_keep = [0] + list(range(19, 19+forecast_period))

with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for i, row in enumerate(reader):
        if i in rows_to_keep:
            writer.writerow(row)

