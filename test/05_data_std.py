'''
    data_std.py
    ~~~~~~~~~~~~~~~~~~~~~~~
    z-score 标准化 - 适用于属性A的最大值和最小值未知的情况，或有超出取值范围的离群数据的情况。
        基于原始数据的均值（mean）和标准差（standard deviation）进行数据的标准化。
        将A的原始值x使用z-score标准化到x'。
            新数据=（原数据-均值）/标准差
    实现航空数据标准化计算
'''
# 目标：获取到最终的标准化后的数据样本

# 步骤1：导入 pandas 模块
import pandas as pd
import os

# 步骤2：设置降维后的数据文件的地址
# excel_file = os.getcwd() + os.sep + 'out' + os.sep + 'reduction_result.xls'
excel_file = os.getcwd() + os.sep + 'out' + os.sep + 'stratified_result.xls'
# 步骤3：使用pd.ExcelFile()函数打开excel文件
ef = pd.ExcelFile(excel_file)
'''
    其中：
        ef.sheet_names, 是 list，保存所有的 sheet 的名字
        ef.book, 是 list， 保存所有的 sheet 的所有的数据
'''
# 步骤4：读取到有效Sheet页数据并转换成DataFrame对象
# 将 ef 中的第一个 sheet 的名字指定的 sheet 转换为 dataframe
data = ef.parse(ef.sheet_names[0])  # 此处等价于：#data = ef.parse('Sheet1')

# 步骤5：数据数据探查信息
# 输出前5条测试片段数据
print("前五条数据为：\n", data.head())
# 输出数据的统计信息
print("数据统计信息为：\n")
print(data.info())
# 输出数据的规模（形状）信息
print('数据规模为：\n', data.shape)



'''
aix=0, 表示求每一列的平均值，结果为每列平均值组成的一行（横向的一维数组 1*N ）
aix=1，表示求每一行的平均值，结果为每行的平均值组成的一列（纵向的一维数组<?> N*1)
如：
    [M.N,P]
    aixm, aixn, aixp
    那个aix=0，那一个维度就压缩为1
'''
# 步骤6：对数据进行标准化计算
# 标准化
data_std = (data - data.mean(axis=0))/(data.std(axis=0))
print('\n标准化计算后的数据：')
print(data_std)

# 步骤7：设置excel文件的保存存储路径
excel_path = os.getcwd() + '\out' + os.sep + 'std_result.xls'

# 步骤8：重新设置数据列索引名称，在每一个列标题前增加 Z 字符
data_std.columns = ['Z'+i for i in data_std.columns]

# 步骤9：将最终的数据探索结果保存到一个excel的中间文件中 std_result.xls
data_std.to_excel(excel_path, index=False)  # 不输出每一行前的索引值
print('>>数据标准化完毕，并保存结果数据文件.')
