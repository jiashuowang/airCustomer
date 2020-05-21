'''
    data_clean.py
    ~~~~~~~~~~~~~~~~~~~~~~~
    数据清洗
    实现航空数据清洗，去除无意义数据(初筛、终筛）
        无意义数据定义：
            初筛：两年票价均为 null（不为0）
                票价为空值，票价最小值为0，折扣率最小值为0，总飞行公里数大于0的记录。
                票价为空值的数据可能是客户不存在乘机记录造成，也可能是数据缺失造成的。
                条件：
                    第一年并且第二年票价消费金额为0 ： SUM_YR_1 非空 & SUM_YR_2 非空
                处理方法：
                    丢弃票价为空的记录。
            终筛：
                第1年或者第2年的票价不为0， 并且总里程和评价折扣均不为0（可能是客户乘坐0折机票或者积分兑换造成）。
                条件：
                    第一年或者第二年票价消费金额不为0： SUM_YR_1 != 0 | SUM_YR_2 != 0
                    并且
                    总飞行里程并且平均折扣率均不为0：SEG_KM_SUM != 0 & avg_discount != 0
                处理方法：
                    由于原始数据量大，这类数据所占比例较小，对于问题影响不大，因此对其进行丢弃处理。
'''

# 目标：清洗不满足条件的原始数据，并将清洗后去除无意义数据的最终结果保存成excel文件

# 步骤1：导入pandas模块
import pandas as pd
import os

# 步骤2：设置原始数据文件的地址
csvFile = os.getcwd() + os.sep + 'data' + os.sep + 'air_data.csv'
# 步骤3：使用pd.read_csv()函数读取数据
data = pd.read_csv("data/air_data.csv",encoding="gb18030") #导入航空数据
# 视需要，对 dataframe 按照指定列进行排序
# data = data.sort_values(by="MEMBER_NO", ascending=True)  #by 指定列升序排序
# 步骤4：输出前5条测试片段数据
print(data.head())
# 步骤5：输出数据的统计信息
print(data.info())

# 步骤6：设置初步筛选的条件
condition = data['SUM_YR_1'].notnull() & data['SUM_YR_2'].notnull()
# 步骤7：初步删选原始数据，将第1年飞行累计和第2年飞行累计进行筛选过滤
data_first_screen = data[condition]
# 输出初筛结果数据的统计信息、形状信息
print('初筛结果数据的统计信息为：', data_first_screen.info())
print('初筛结果数据的形状为：',data_first_screen.shape)
print('初筛结果的数据为：', data_first_screen)


# 步骤8：设置数据过滤(终筛）的条件
condition1 = data_first_screen['SUM_YR_1'] != 0
condition2 = data_first_screen['SUM_YR_2'] != 0
condition3 = (data_first_screen['SEG_KM_SUM'] != 0) & (data_first_screen['avg_discount'] != 0)
# 步骤9：根据过滤条件筛选数据
data_end_screen = data_first_screen[(condition1 | condition2) & condition3]
# 输出初筛结果数据的统计信息、形状信息
print('终筛结果数据的统计信息为：', data_end_screen.info())
print('终筛结果数据的形状为：',data_end_screen.shape)
print('\n 筛选之后的数据')
print(data_end_screen)

# 步骤10：设置excel文件的保存存储路径
excelPath = os.getcwd() + os.sep + 'out' + os.sep + 'cleaned_result.xls'
print(excelPath)
# 步骤11：将最终的数据探索结果保存到一个excel的中间文件中cleaned.xls
data_end_screen.to_excel(excelPath)
print('>>数据清洗结束，并保存结果数据文件.')