
'''
    data_reduction.py
    ~~~~~~~~~~~~~~~~~~~~~~~
    实现航空数据规约与转换
'''

# 目标：根据LRFMC模型从原始数据中选取指定的属性数据项

# 步骤1：导入pandas模块
import pandas as pd
import os

# 步骤2：设置原始清洗后的数据文件的地址
excelFile = os.getcwd() + os.sep + 'out' + os.sep + 'cleaned_result.xls'
# 步骤3：使用pd.ExcelFile()函数读取数据
ef = pd.ExcelFile(excelFile)
# 步骤4：读取到有效Sheet页数据并转换成DataFrame对象
data = ef.parse('Sheet1')
# 步骤4：输出前5条测试片段数据
print(data.head())
# 步骤5：输出数据的统计信息
print(data.info())

# 步骤6：筛选出核心的6个数据属性
# 加上会员号
data_temp = data[['MEMBER_NO', 'FFP_DATE', 'LOAD_TIME', 'FLIGHT_COUNT', 'avg_discount', 'SEG_KM_SUM', 'LAST_TO_END']].copy()
print(data_temp)
print(data_temp.info())

# 步骤7：获取L属性的数据
data_temp['L'] = (pd.to_datetime(data_temp['LOAD_TIME']) - pd.to_datetime(data_temp['FFP_DATE']))/30
data_temp['R'] = data_temp['LAST_TO_END']/30
data_temp['F'] = data_temp['FLIGHT_COUNT']
data_temp['M'] = data_temp['SEG_KM_SUM']
data_temp['C'] = data_temp['avg_discount']

print(data_temp[['L', 'R', 'F', 'M', 'C']])
data_reduction = data_temp[['MEMBER_NO', 'L', 'R', 'F', 'M', 'C']]
# 步骤8：设置excel文件的保存存储路径
excelPath = os.getcwd() + os.sep + 'out' + os.sep + 'reduction_result.xls'

# 步骤9：将最终的数据探索结果保存到一个excel的中间文件中 reduction_result.xls
data_reduction.to_excel(excelPath, index=True)
print('>>数据降维、变换，并保存结果数据文件.')
