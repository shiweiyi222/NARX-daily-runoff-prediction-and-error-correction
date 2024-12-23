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
from keras.callbacks import Callback
from keras.losses import Huber
import matplotlib.pyplot as plt
import statsmodels.api as sm
from keras.callbacks import LearningRateScheduler

# 指定文件的路径，用于训练的数据（训练集+验证集）
rainfall_file = 'C:\\Users\\25083\\Desktop\\神经网络训练总文件夹\\data_wanzhen\\train_1_rain.csv'
flow_file = 'C:\\Users\\25083\\Desktop\\神经网络训练总文件夹\\data_wanzhen\\train_1_flow.csv'

# 加载CSV文件数据并清洗数据
data_rainfall = np.genfromtxt(rainfall_file, delimiter=',', skip_header=True)
data_flow = np.genfromtxt(flow_file, delimiter=',', skip_header=True)

X = clean_data(data_rainfall)  # 历史日降雨数据
Y = clean_data(data_flow)  # 日径流数据

# 归一化处理
# 创建 MinMaxScaler 对象
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

# 对 X 和 Y 进行归一化
X = scaler_X.fit_transform(X)
Y = scaler_Y.fit_transform(Y)

# 定义外部输入延迟线（外部输入降雨蒸发因素）
time_delay_outinput = time_delay_autoregressive

# 创建一个空数据集，用来整合Y为输入格式
Y_tem = []
Y_output = []
# 构建训练数据集，考虑延迟线
for i in range(time_delay_autoregressive,len(Y)):
    # 过去 Y 的延迟线
    past_Y_values = Y[i-time_delay_autoregressive:i]
    # # 在最后一行添加一个0元素
    # new_row = np.array([0])
    # array_with_zero = np.append(past_Y_values, new_row.reshape(1, -1), axis=0)
    # 获取数组的最后一个元素
    last_value = past_Y_values[-1]
    # 将最后一个元素添加到数组的末尾
    extended_array = np.append(past_Y_values, last_value)
    # 将数组的形状变为 (13, 1)
    extended_array = extended_array.reshape((time_delay_autoregressive+1, 1))

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
X_tem = []
# 构建训练数据集
for i in range(time_delay_autoregressive,len(Y)):
    # 过去 X 的延迟线
    past_X_values = X[i - time_delay_autoregressive:i+1]
    # 添加X到输入X中
    X_tem.append(past_X_values)
# 转化为numpy数组
X_input = np.array(X_tem)

# 划分训练集和验证集
split_index = int(split_ratio * len(X_input))
# 划分用于输入的X和Y
X_train, X_test = X_input[:split_index], X_input[split_index:]
Y_train, Y_test = Y_input[:split_index], Y_input[split_index:]
# 划分目标的Y
Y_output_train, Y_output_test = Y_output[:split_index], Y_output[split_index:]

def nash(y_true, y_pred):
    epsilon = 1e-7
    numerator = K.sum(K.square(y_true - y_pred))
    denominator = K.sum(K.square(y_true - K.mean(y_true))) + epsilon
    nse = 1 - numerator / denominator
    return nse

# 输入层1，过去的降雨/外部输入
input_rain = Input(shape=(time_delay_autoregressive+1, 8), name='input_rain')

# 输入层2，过去的径流/自回归项目
input_flow = Input(shape=(time_delay_autoregressive+1, 8), name='input_flow')

# 独立的LSTM层
lstm_rain_1 = LSTM(units=512, activation=act, return_sequences=True)(input_rain)
lstm_rain_1 = Dropout(drop)(lstm_rain_1)   # 添加Dropout正则化，设置合适的比例
lstm_rain_2 = LSTM(units=512, activation=act, return_sequences=True)(lstm_rain_1)
lstm_rain_2 = Dropout(drop)(lstm_rain_2)   # 添加Dropout正则化，设置合适的比例
lstm_rain_3 = LSTM(units=512, activation=act, return_sequences=True)(lstm_rain_2)
lstm_rain_3 = Dropout(drop)(lstm_rain_3)   # 添加Dropout正则化，设置合适的比例
# lstm_rain_4 = LSTM(units=800, activation=act, return_sequences=True)(lstm_rain_3)
# lstm_rain_4 = Dropout(drop)(lstm_rain_4)   # 添加Dropout正则化，设置合适的比例

lstm_flow_1 = LSTM(units=512, activation=act, return_sequences=True)(input_flow)
lstm_flow_1 = Dropout(drop)(lstm_flow_1)   # 添加Dropout正则化，设置合适的比例
lstm_flow_2 = LSTM(units=512, activation=act, return_sequences=True)(lstm_flow_1)
lstm_flow_2 = Dropout(drop)(lstm_flow_2)   # 添加Dropout正则化，设置合适的比例
lstm_flow_3 = LSTM(units=512, activation=act, return_sequences=True)(lstm_flow_2)
lstm_flow_3 = Dropout(drop)(lstm_flow_3)   # 添加Dropout正则化，设置合适的比例
# lstm_flow_4 = LSTM(units=800, activation=act, return_sequences=True)(lstm_flow_3)
# lstm_flow_4 = Dropout(drop)(lstm_flow_4)   # 添加Dropout正则化，设置合适的比例

# 共享的LSTM层1
shared_lstm1 = LSTM(units=num_1, activation=act, return_sequences=True)
dropout_shared_lstm1 = Dropout(drop)  # 添加Dropout层

# 共享的LSTM层2
shared_lstm2 = LSTM(units=num_1, activation=act, return_sequences=True)
dropout_shared_lstm2 = Dropout(drop)  # 添加Dropout层

# 共享的LSTM层3
shared_lstm3 = LSTM(units=num_1, activation=act, return_sequences=True)
dropout_shared_lstm3 = Dropout(drop)  # 添加Dropout层

# 共享的LSTM层4
shared_lstm4 = LSTM(units=num_1, activation=act, return_sequences=True)
dropout_shared_lstm4 = Dropout(drop)  # 添加Dropout层

# 共享的LSTM层5
shared_lstm5 = LSTM(units=num_1, activation=act, return_sequences=True)
dropout_shared_lstm5 = Dropout(drop)  # 添加Dropout层

# 共享的LSTM层6
shared_lstm6 = LSTM(units=num_1, activation=act, return_sequences=False)
dropout_shared_lstm6 = Dropout(drop)  # 添加Dropout层

# 应用共享的LSTM层到输入1和输入2，并添加Dropout
lstm_rain_comb1 = shared_lstm1(lstm_rain_3)
lstm_rain_comb1 = dropout_shared_lstm1(lstm_rain_comb1)
lstm_flow_comb1 = shared_lstm1(lstm_flow_3)
lstm_flow_comb1 = dropout_shared_lstm1(lstm_flow_comb1)

# 应用共享的2层，并添加Dropout
lstm_rain_comb2 = shared_lstm2(lstm_rain_comb1)
lstm_rain_comb2 = dropout_shared_lstm2(lstm_rain_comb2)
lstm_flow_comb2 = shared_lstm2(lstm_flow_comb1)
lstm_flow_comb2 = dropout_shared_lstm2(lstm_flow_comb2)

# 应用共享的3层，并添加Dropout
lstm_rain_comb3 = shared_lstm3(lstm_rain_comb2)
lstm_rain_comb3 = dropout_shared_lstm3(lstm_rain_comb3)
lstm_flow_comb3 = shared_lstm3(lstm_flow_comb2)
lstm_flow_comb3 = dropout_shared_lstm3(lstm_flow_comb3)

# # 应用共享的4层，并添加Dropout
lstm_rain_comb4 = shared_lstm4(lstm_rain_comb3)
lstm_rain_comb4 = dropout_shared_lstm4(lstm_rain_comb4)
lstm_flow_comb4 = shared_lstm4(lstm_flow_comb3)
lstm_flow_comb4 = dropout_shared_lstm4(lstm_flow_comb4)

# # 应用共享的5层，并添加Dropout
lstm_rain_comb5 = shared_lstm5(lstm_rain_comb4)
lstm_rain_comb5 = dropout_shared_lstm5(lstm_rain_comb5)
lstm_flow_comb5 = shared_lstm5(lstm_flow_comb4)
lstm_flow_comb5 = dropout_shared_lstm5(lstm_flow_comb5)

# # 应用共享的6层，并添加Dropout
lstm_rain_comb6 = shared_lstm6(lstm_rain_comb5)
lstm_rain_comb6 = dropout_shared_lstm6(lstm_rain_comb6)
lstm_flow_comb6 = shared_lstm6(lstm_flow_comb5)
lstm_flow_comb6 = dropout_shared_lstm6(lstm_flow_comb6)
# 合并LSTM输出
concatenated = tf.keras.layers.Concatenate()([lstm_rain_comb6, lstm_flow_comb6])
#concatenated = tf.keras.layers.Concatenate()([lstm_rain_1, lstm_flow_1])
# concatenated = tf.keras.layers.Concatenate()([input_rain, input_flow])

# lstm_1 = LSTM(units=512, activation=act, return_sequences=True)(concatenated)
# lstm_1=  Dropout(drop)(lstm_1)   # 添加Dropout正则化，设置合适的比例


# 最终输出层
output = Dense(units=1, activation='linear')(concatenated)

# 创建模型
model = Model(inputs=[input_rain, input_flow], outputs=output)

# 模型编译
optimizer = Adam()
model.compile(optimizer='adam', loss=loss, metrics=[nash])

# 定义回调函数，保存在每个epoch后性能最好的模型
checkpoint_callback = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

#定义早停器，定义回调函数
early_stopping = EarlyStopping(monitor='val_loss', patience = patience, restore_best_weights=True)

# 定义ReduceLROnPlateau回调函数，当损失不在降低时，降低学习率
# reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.85, patience=10, min_lr=0.00001)

class CustomCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_nash', save_best_only=True, train_nash_threshold=0.92):
        super(CustomCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.train_nash_threshold = train_nash_threshold
        self.best_nash = -float('inf') if save_best_only else float('inf')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_nash = logs.get('nash')
        val_nash = logs.get(self.monitor)

        if train_nash is None or val_nash is None:
            return

        if train_nash >= self.train_nash_threshold and val_nash > self.best_nash:
            self.best_nash = val_nash
            self.model.save_weights(self.filepath, overwrite=True)
# 创建自定义的Checkpoint回调函数
checkpoint_callback_nash = CustomCheckpoint('best_model_nash.h5', train_nash_threshold=0.9)
# 在模型的fit方法中使用批处理大小调整回调函数
history = model.fit(
    [X_train, Y_train], Y_output_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=([X_test, Y_test], Y_output_test),
    callbacks=[checkpoint_callback,checkpoint_callback_nash]
)

# 保存模型
# 获取当前时间并格式化为字符串
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
# 生成保存模型的文件名，包括当前时间
model_filename = f'model_{current_time}.h5'
# 保存模型
model.save(model_filename)

# 进行模型预测---训练集预测结果
Y_train_pred = model.predict([X_train,Y_train])

# 进行模型预测---验证集预测结果
Y_val_pred = model.predict([X_test, Y_test])

# 反归一化预测结果
Y_train_pred_original = scaler_Y.inverse_transform(Y_train_pred)
Y_val_pred_original = scaler_Y.inverse_transform(Y_val_pred)
Y_train_gyh = scaler_Y.inverse_transform(Y_output_train)
Y_test_gyh = scaler_Y.inverse_transform(Y_output_test)

# 计算指标与结果存库
cal_index(Y_train_gyh,Y_train_pred_original)
to_excel(Y_train_gyh , Y_train_pred_original , 'trainout.xlsx')
cal_index(Y_test_gyh,Y_val_pred_original)
to_excel(Y_test_gyh , Y_val_pred_original , 'testout.xlsx')

# # 计算无归一化指标与结果存库
# cal_index(Y_output_train,Y_train_pred)
# to_excel(Y_output_train , Y_train_pred , 'trainout.xlsx')
# cal_index(Y_output_test,Y_val_pred)
# to_excel(Y_output_test , Y_val_pred , 'testout.xlsx')

# # 计算残差（预测值与实际值之差）
# residuals = Y_test_gyh - Y_val_pred_original.flatten()
#
# # 绘制误差自相关图
# fig, ax = plt.subplots(figsize=(10, 5))
# sm.graphics.tsa.plot_acf(residuals, lags=30, ax=ax)
# plt.title("Residual Autocorrelation Plot")
# plt.show()

# # 计算输入误差交叉相关
# input_residual_cross_corr = np.correlate(X_test.flatten(), residuals, mode='full')
#
# # 绘制输入误差交叉相关图
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.plot(input_residual_cross_corr)
# plt.title("Input-Error Cross-Correlation Plot")
# plt.show()











