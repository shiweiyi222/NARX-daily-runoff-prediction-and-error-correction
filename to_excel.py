import pandas as pd
import numpy as np

def to_excel(A,B,path):
    # 将数组转换为 pandas DataFrame
    df = pd.DataFrame({'Column1': A.flatten(), 'Column2': B.flatten()})
    # 将DataFrame写入Excel文件
    excel_filename = path  # 替换为你想要的Excel文件名
    df.to_excel(excel_filename, index=False)

