import numpy as np
import datetime

def to_txt(path_txt,X):
    # 获取当前时间
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 将数组保存到txt文件
    np.savetxt(path_txt, X)

    # 打开文件，将当前时间写入第一行
    with open(path_txt, 'r+') as file:
        content = file.read()
        file.seek(0, 0)
        file.write(f'Time: {current_time}\n\n{content}')






