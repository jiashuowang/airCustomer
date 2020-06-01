import pandas as pd
import os
from stratified_sampling import get_sample

excel_file = os.getcwd() + os.sep + 'out' + os.sep + 'reduction_result.xls'
ef = pd.ExcelFile(excel_file)
data = ef.parse(ef.sheet_names[0])


# 解决数据倾斜问题，进行分层抽样
data = data[['L', 'R', 'F', 'M', 'C']]
# 去掉边缘数据
data = data[(data['F']<= 150)&(data['M'] <= 250000)]
# save_path = os.getcwd() + '\out' + os.sep + 'test.xls'
# data.to_excel(save_path, index=False)

# 分层条件
def L_condition(num):
    if num < 22:
        return "1"
    elif num >= 22 and num < 35:
        return "2"
    elif num >= 35 and num < 55:
        return "3"
    elif num >= 55 and num < 80:
        return "4"
    elif num >= 80:
        return "5"

def R_condition(num):
    if num < 0.6:
        return "1"
    elif num >= 0.6 and num < 2.2:
        return "2"
    elif num >= 2.2 and num < 5:
        return "3"
    elif num >= 5 and num < 10:
        return "4"
    elif num >= 10:
        return "5"

def F_condition(num):
    if num < 3:
        return "1"
    elif num >= 3 and num < 5:
        return "2"
    elif num >= 5 and num < 10:
        return "3"
    elif num >= 10 and num < 20:
        return "4"
    elif num >= 20:
        return "5"


def M_condition(num):
    if num < 4500:
        return "1"
    elif num >= 4500 and num < 8000:
        return "2"
    elif num >= 8000 and num < 14000:
        return "3"
    elif num >= 14000 and num < 27000:
        return "4"
    elif num >= 27000:
        return "5"


def C_condition(num):
    if num < 0.58:
        return "1"
    elif num >= 0.58 and num < 0.67:
        return "2"
    elif num >= 0.67 and num < 0.74:
        return "3"
    elif num >= 0.74 and num < 0.83:
        return "4"
    elif num >= 0.83:
        return "5"


#  加上标签
data['L_rank'] =data['L'].apply(lambda x:L_condition(x))
data['R_rank'] =data['R'].apply(lambda x:R_condition(x))
data['F_rank'] =data['F'].apply(lambda x:F_condition(x))
data['M_rank'] =data['M'].apply(lambda x:M_condition(x))
data['C_rank'] =data['C'].apply(lambda x:C_condition(x))
# print(data)
# print(data['C_rank'].value_counts())

# 分维度进行分层抽样
stratified_df_L = get_sample(df=data, sampling='stratified', k=1000, stratified_col=['L_rank'])
print(stratified_df_L['L_rank'].value_counts())
stratified_df_R = get_sample(df=data, sampling='stratified', k=1000, stratified_col=['R_rank'])
print(stratified_df_R['R_rank'].value_counts())
stratified_df_F = get_sample(df=data, sampling='stratified', k=1000, stratified_col=['F_rank'])
print(stratified_df_F['F_rank'].value_counts())
stratified_df_M = get_sample(df=data, sampling='stratified', k=1000, stratified_col=['M_rank'])
print(stratified_df_M['M_rank'].value_counts())
stratified_df_C = get_sample(df=data, sampling='stratified', k=1000, stratified_col=['C_rank'])
print(stratified_df_C['C_rank'].value_counts())

frames = [stratified_df_L, stratified_df_R, stratified_df_F, stratified_df_M, stratified_df_C]
stratified_result = pd.concat(frames)

print(stratified_result)

# 存为Excel文件
save_path = os.getcwd() + '\out' + os.sep + 'stratified_result.xls'
stratified_result.to_excel(save_path, index=False)
