from clean_data import clean_data
from to_txt import to_txt
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import tensorflow as tf
from keras.layers import Input, LSTM, Dense, Dropout
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
import numpy as np
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from datetime import datetime
from keras.models import load_model
from keras.regularizers import l2
from sklearn.metrics import mean_absolute_error
from fireTS.models import NARX
from cal_index import cal_index
from to_excel import to_excel
from global_var import *
import tensorflow as tf
import keras.backend as K
from keras.losses import Huber

# 加载输入数据
pre_rain_file = 'output_input_pre_rain.csv'
e_file = '2_e.csv'
Q_pre_file = '2_result.csv'

# 加载CSV文件数据并清洗数据
data_X1 = np.genfromtxt(pre_rain_file, delimiter=',', skip_header=True)
data_X2 = np.genfromtxt(Q_pre_file, delimiter=',', skip_header=True)
data_Y  = np.genfromtxt(e_file, delimiter=',', skip_header=True)

X1 = clean_data(data_X1)  # 降雨预测数据
X2 = clean_data(data_X2)  # 径流预测的结果
Y = clean_data(data_Y)    # 误差预测系列

# 归一化处理
scaler_X1 = MinMaxScaler()
scaler_X2 = MinMaxScaler()
scaler_Y = MinMaxScaler()

# 对 X 和 Y 进行归一化
X1 = scaler_X1.fit_transform(X1)
X2 = scaler_X2.fit_transform(X2)
Y = scaler_Y.fit_transform(Y)

# 定义外部输入延迟线
time_de = time_delay_autoregressive_2

# 创建一个空数据集，用来整合Y为输入格式
Y_tem = []
Y_output = []
# 构建训练数据集，考虑延迟线
for i in range(time_de,len(Y)):
    # 过去 Y 的延迟线
    past_Y_values = Y[i-time_de:i]
    # 获取数组的最后一个元素
    last_value = past_Y_values[-1]
    # 将最后一个元素添加到数组的末尾
    extended_array = np.append(past_Y_values, last_value)
    # 将数组的形状变为 (13, 1)
    extended_array = extended_array.reshape((time_de+1, 1))

    # 添加到输入Y中
    Y_tem.append(extended_array)

    # 构建目标
    target_data = Y[i]
    Y_output.append(target_data)

# 转化为numpy数组
Y_input = np.array(Y_tem)
Y_output = np.array(Y_output)

# 将数组的列复制到新的维度，使其变为（13，8）
Y_input = np.tile(Y_input, 8)

# 创建一个空数据集，用来整合X为输入格式（，，）
X1_tem = []
# 构建训练数据集
for i in range(time_de,len(Y)):
    # 过去 X 的延迟线
    past_X1_values = X1[i - time_de:i+1]
    # 添加X到输入X中
    X1_tem.append(past_X1_values)
# 转化为numpy数组
X1_input = np.array(X1_tem)

# 创建一个空数据集，用来整合X为输入格式（，，）
X2_tem = []
# 构建训练数据集
for i in range(time_de,len(Y)):
    # 过去 X 的延迟线
    past_X2_values = X2[i - time_de:i+1]
    # 添加X到输入X中
    X2_tem.append(past_X2_values)
# 转化为numpy数组
X2_input = np.array(X2_tem)
# 将数组的列复制到新的维度，使其变为（13，8）
X2_input = np.tile(X2_input, 8)

# print(Y_input.shape)
# print(Y_output.shape)
# print(X1_input.shape)
# print(X2_input.shape)

# 划分训练集和验证集
# 划分训练集和验证集
split_index = int(split_ratio * len(X1_input))
# 划分用于输入的X和Y
X1_train, X1_test = X1_input[:split_index], X1_input[split_index:]
X2_train, X2_test = X2_input[:split_index], X2_input[split_index:]
Y_train, Y_test = Y_input[:split_index], Y_input[split_index:]
# 划分目标的Y
Y_output_train, Y_output_test = Y_output[:split_index], Y_output[split_index:]

def nash_sutcliffe_efficiency(y_true, y_pred):
    """
    Custom evaluation function to calculate Nash-Sutcliffe Efficiency (NSE).
    """
    epsilon = 1e-8
    numerator = K.sum(K.square(y_true - y_pred))
    denominator = K.sum(K.square(y_true - K.mean(y_true))) + epsilon
    nse = 1 - numerator / denominator
    return nse

# 输入层1，降雨预测输入
input_X1 = Input(shape=(time_de+1, 8), name='input_X1')

# 输入层2，径流预测结果作输入
input_X2 = Input(shape=(time_de+1, 8), name='input_X2')

# 输入层2，误差系列作输入
input_Y = Input(shape=(time_de+1, 8), name='input_Y')

# 独立的LSTM层
lstm_X1_1 = LSTM(units=800, activation=act_1, return_sequences=True)(input_X1)
lstm_X1_1 = Dropout(drop)(lstm_X1_1)   # 添加Dropout正则化，设置合适的比例
lstm_X1_2 = LSTM(units=800, activation=act_1, return_sequences=True)(lstm_X1_1)
lstm_X1_2 = Dropout(drop)(lstm_X1_2)   # 添加Dropout正则化，设置合适的比例
lstm_X1_3 = LSTM(units=800, activation=act_1, return_sequences=True)(lstm_X1_2)
lstm_X1_3 = Dropout(drop)(lstm_X1_3)   # 添加Dropout正则化，设置合适的比例


lstm_X2_1 = LSTM(units=800, activation=act_1, return_sequences=True)(input_X2)
lstm_X2_1 = Dropout(drop)(lstm_X2_1)   # 添加Dropout正则化，设置合适的比例
lstm_X2_2 = LSTM(units=800, activation=act_1, return_sequences=True)(lstm_X2_1)
lstm_X2_2 = Dropout(drop)(lstm_X2_2)   # 添加Dropout正则化，设置合适的比例
lstm_X2_3 = LSTM(units=800, activation=act_1, return_sequences=True)(lstm_X2_2)
lstm_X2_3 = Dropout(drop)(lstm_X2_3)   # 添加Dropout正则化，设置合适的比例

lstm_Y_1 = LSTM(units=800, activation=act_1, return_sequences=True)(input_Y)
lstm_Y_1 = Dropout(drop)(lstm_Y_1)   # 添加Dropout正则化，设置合适的比例
lstm_Y_2 = LSTM(units=800, activation=act_1, return_sequences=True)(lstm_Y_1)
lstm_Y_2 = Dropout(drop)(lstm_Y_2)   # 添加Dropout正则化，设置合适的比例
lstm_Y_3 = LSTM(units=800, activation=act_1, return_sequences=True)(lstm_Y_2)
lstm_Y_3 = Dropout(drop)(lstm_Y_3)   # 添加Dropout正则化，设置合适的比例

# 共享的LSTM层1
shared_lstm1 = LSTM(units=num_1, activation=act_1, return_sequences=True)
dropout_shared_lstm1 = Dropout(drop)  # 添加Dropout层

# 共享的LSTM层2
shared_lstm2 = LSTM(units=num_1, activation=act_1, return_sequences=True)
dropout_shared_lstm2 = Dropout(drop)  # 添加Dropout层

# 共享的LSTM层3
shared_lstm3 = LSTM(units=num_1, activation=act_1, return_sequences=True)
dropout_shared_lstm3 = Dropout(drop)  # 添加Dropout层

# 共享的LSTM层4
shared_lstm4 = LSTM(units=num_1, activation=act_1, return_sequences=True)
dropout_shared_lstm4 = Dropout(drop)  # 添加Dropout层

# 共享的LSTM层5
shared_lstm5 = LSTM(units=num_1, activation=act_1, return_sequences=True)
dropout_shared_lstm5 = Dropout(drop)  # 添加Dropout层

# 共享的LSTM层6
shared_lstm6 = LSTM(units=num_1, activation=act_1, return_sequences=False)
dropout_shared_lstm6 = Dropout(drop)  # 添加Dropout层

# 应用共享的LSTM层到输入1和输入2，并添加Dropout
lstm_X1_comb1 = shared_lstm1(lstm_X1_3)
lstm_X1_comb1 = dropout_shared_lstm1(lstm_X1_comb1)
lstm_X2_comb1 = shared_lstm1(lstm_X2_3)
lstm_X2_comb1 = dropout_shared_lstm1(lstm_X2_comb1)
lstm_Y_comb1 = shared_lstm1(lstm_Y_3)
lstm_Y_comb1 = dropout_shared_lstm1(lstm_Y_comb1)
# 应用共享的2层，并添加Dropout
lstm_X1_comb2 = shared_lstm2(lstm_X1_comb1)
lstm_X1_comb2 = dropout_shared_lstm2(lstm_X1_comb2)
lstm_X2_comb2 = shared_lstm2(lstm_X2_comb1)
lstm_X2_comb2 = dropout_shared_lstm2(lstm_X2_comb2)
lstm_Y_comb2 = shared_lstm2(lstm_Y_comb1)
lstm_Y_comb2 = dropout_shared_lstm2(lstm_Y_comb2)

# 应用共享的3层，并添加Dropout
lstm_X1_comb3 = shared_lstm3(lstm_X1_comb2)
lstm_X1_comb3 = dropout_shared_lstm3(lstm_X1_comb3)
lstm_X2_comb3 = shared_lstm3(lstm_X2_comb2)
lstm_X2_comb3 = dropout_shared_lstm3(lstm_X2_comb3)
lstm_Y_comb3 = shared_lstm3(lstm_Y_comb2)
lstm_Y_comb3 = dropout_shared_lstm3(lstm_Y_comb3)

# # 应用共享的4层，并添加Dropout
lstm_X1_comb4 = shared_lstm4(lstm_X1_comb3)
lstm_X1_comb4 = dropout_shared_lstm4(lstm_X1_comb4)
lstm_X2_comb4 = shared_lstm4(lstm_X2_comb3)
lstm_X2_comb4 = dropout_shared_lstm4(lstm_X2_comb4)
lstm_Y_comb4 = shared_lstm4(lstm_Y_comb3)
lstm_Y_comb4 = dropout_shared_lstm4(lstm_Y_comb4)

# # 应用共享的5层，并添加Dropout
lstm_X1_comb5 = shared_lstm5(lstm_X1_comb4)
lstm_X1_comb5 = dropout_shared_lstm5(lstm_X1_comb5)
lstm_X2_comb5 = shared_lstm5(lstm_X2_comb4)
lstm_X2_comb5 = dropout_shared_lstm5(lstm_X2_comb5)
lstm_Y_comb5 = shared_lstm5(lstm_Y_comb4)
lstm_Y_comb5 = dropout_shared_lstm5(lstm_Y_comb5)

# # 应用共享的6层，并添加Dropout
lstm_X1_comb6 = shared_lstm6(lstm_X1_comb5)
lstm_X1_comb6 = dropout_shared_lstm6(lstm_X1_comb6)
lstm_X2_comb6 = shared_lstm6(lstm_X2_comb5)
lstm_X2_comb6 = dropout_shared_lstm6(lstm_X2_comb6)
lstm_Y_comb6 = shared_lstm6(lstm_Y_comb5)
lstm_Y_comb6 = dropout_shared_lstm6(lstm_Y_comb6)

# 合并LSTM输出
concatenated = tf.keras.layers.Concatenate()([lstm_X1_comb6, lstm_X2_comb6, lstm_Y_comb6])

# 最终输出层
output = Dense(units=1, activation='linear')(concatenated)

# 创建模型
model = Model(inputs=[input_X1, input_X2, input_Y], outputs=output)

# 模型编译
optimizer = Adam()
model.compile(optimizer='adam', loss=loss_1, metrics=[nash_sutcliffe_efficiency])

# 定义回调函数，保存在每个epoch后性能最好的模型
checkpoint_callback = ModelCheckpoint('best_model_2.h5', monitor='val_loss', save_best_only=True)

#定义早停器，定义回调函数
# early_stopping = EarlyStopping(monitor='val_loss', patience = patience_2, restore_best_weights=True)

# 定义ReduceLROnPlateau回调函数，当损失不在降低时，降低学习率
# reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.85, patience=10, min_lr=0.00001)

# 训练模型
history = model.fit(
    [X1_train, X2_train, Y_train],Y_output_train,
    epochs = epoch_2,  # 调整为您需要的训练周期数
    batch_size = batch_size_2,  # 调整为您需要的批处理大小
    validation_data=([X1_test, X2_test, Y_test], Y_output_test),
    callbacks=[checkpoint_callback]
)

# 保存模型
# 获取当前时间并格式化为字符串
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
# 生成保存模型的文件名，包括当前时间
model_filename = f'model2_{current_time}.h5'
# 保存模型
model.save(model_filename)

# 进行模型预测---训练集预测结果
Y_train_pred = model.predict([X1_train, X2_train, Y_train])

# 进行模型预测---验证集预测结果
Y_val_pred = model.predict([X1_test, X2_test, Y_test])

# 反归一化预测结果
Y_train_pred_original = scaler_Y.inverse_transform(Y_train_pred)
Y_val_pred_original = scaler_Y.inverse_transform(Y_val_pred)
Y_train_gyh = scaler_Y.inverse_transform(Y_output_train)
Y_test_gyh = scaler_Y.inverse_transform(Y_output_test)

# 计算指标与结果存库
cal_index(Y_train_gyh,Y_train_pred_original)
to_excel(Y_train_gyh , Y_train_pred_original , '2_trainout.xlsx')
cal_index(Y_test_gyh,Y_val_pred_original)
to_excel(Y_test_gyh , Y_val_pred_original , '2_testout.xlsx')