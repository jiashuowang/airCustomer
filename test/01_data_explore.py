'''
    data_explore.py
    ~~~~~~~~~~~~~~~~~~~~~~~
    完成数据的探查
    ：实现航空数据探索
'''

# 目标：通过对历史数据air_data.csv的分析，获取数据集的信息。
# 探查数据集的规模、探查数据集的异常数据并统计输出。
# 包括：所有44个数据属性的含有空值的个数、最小值以及最大值三个数据指标

# 步骤1：导入 pandas 模块、导入 os 模块
import pandas as pd
import os

# 步骤2：设置原始数据文件的地址
data_file = "air_data.csv"
data_path = os.getcwd() + os.sep + "data" + os.sep + data_file

# 步骤3：使用 pandas 读入数据文件
data = pd.read_csv(data_path, encoding='gb18030') #导入航空数据

# 步骤4：输出前5条测试片段数据
print(data.head())

# 步骤5：输出数据的统计信息
print(data.info())

# 以下为探查数据集的空值、最大值、最小值，并统计输出
# 步骤6：获取数据中的空值项
'''
dataframe.describe 函数说明：三个参数
1. percentiles,设定数值型特征的统计量，默认是[.5],也就是返回50%数据量时的数字，可修改，
    如：data.describe(percentiles=[.75, .8]
2. include，默认是只计算数值型特征的统计量，
    当include=['O']，会计算离散型变量的统计特征，
    当include=['all']，会把数值型和离散型特征的都进行统计并显示。
    统计的特征包括：非空数量count，唯一值种类unique，出现最多的类型top以及top出现次数freq
3. exclude，指定不选哪些，默认不丢弃任何列，相当于无影响。
'''
explore = data.describe(percentiles=[], include='all').T    # .T 结果进行转置显示
print('设置数据中的空值项统计：')
# 根据需要，显示 dataframe 的所有行、所有列，列中的全部字段内容
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置字段值的显示长度为100，默认为50
pd.set_option('max_colwidth',60)
print(explore)

# 步骤7：计算每个属性的空值个数,
# 空值元素个数 = 整体元素个数 - 非空元素个数count， 并放入 explore 中的 'null' key 对应的 value 中。
explore['null'] = len(data) - explore['count']
print('每个属性的空值个数：')
print(explore['null'])

# 步骤8：抽取数据中的空值个数null、最大值max、最小值min三个数据属性（数据选项），放入 explore_result 中
explore_result = explore[['null', 'max', 'min']]
# 步骤9：重置列索引名称
explore_result.columns = ['空值数', '最大值', '最小值']
print('\n数据探索整理之后的数据结果：')
print(explore_result)

# 步骤10：设置excel文件的保存存储路径
excelPath = os.getcwd() + '\out' + os.sep + 'explore_result.xls'
# 步骤11：将最终的数据探索结果保存到一个 excel 的中间文件中 explore_result.xls
# 注意，需要安装 xlwt 模块
explore_result.to_excel(excelPath)
print('>>数据探索结束，并保存结果数据文件.')
