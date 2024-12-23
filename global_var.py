# 截至第一层模型的全局变量
time_delay_autoregressive = 4  # 延迟线长度
split_ratio = 0.7               # 训练集、验证集比例
drop = 0.2                # 正则化丢弃网络的比例
l2_regularization = 0.001    #lambda参数

loss = 'mean_squared_error'
# loss='mean_absolute_error'
patience = 40                   # 早停器耐心
epochs = 500                   # 训练迭代次数
batch_size = 32            # 批次大小
forecast_period = 950           # 第一次使用预见期长度
num = 512                         #共享层神经元个数
act = 'elu'                    # 激活函数

# 误差预测模型全集变量
time_delay_autoregressive_2 = 6 # 延迟线长度
patience_2 = 25                  # 早停器耐心
act_1 = 'elu'
loss_1 = 'mean_squared_error'
# loss_1 = 'mean_absolute_error'
num_1 = 512
epoch_2 = 500                    # 训练迭代次数
batch_size_2 = 256               # 批次大小
forecast_period_2 = 70           # 第二次真实预测（目标为e）