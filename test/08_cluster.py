import pandas as pd
import os

# 判断是哪种客户
def condition(num):
    if num == 0:
        return "一般客户"
    elif num == 1:
        return "重要挽留客户"
    elif num == 2:
        return "重要发展客户"
    elif num == 3:
        return "低价值客户"
    elif num == 4:
        return "重要保持客户"
excelPath = os.getcwd() + '\out' + os.sep + 'KMeans_cluster_details_data_result.xls'
excelFile = pd.ExcelFile(excelPath)
data = excelFile.parse('Sheet1')
print(data.shape)

excelPath2 = os.getcwd() + '\out' + os.sep + 'reduction_result.xls'
excelFile2 = pd.ExcelFile(excelPath2)
data2 = excelFile2.parse('Sheet1')
print(data2.shape)

# 加上会员信息
data['MEMBER_NO'] = data2['MEMBER_NO']
data['CLUSTER'] =data.RANK.apply(lambda x:condition(x))
data.to_excel('./out/cluster_result.xls', index=False)




print(data)
