import pandas as pd
import os

# 读入clean后的数据
cleanFile = os.getcwd() + os.sep + 'out' + os.sep + 'cleaned_result.xls'
cf = pd.ExcelFile(cleanFile)
cf = cf.parse('Sheet1')

# 有会员聚类信息
cluster = os.getcwd() + os.sep + 'out' + os.sep + 'cluster_result.xls'
cls = pd.ExcelFile(cluster)
cls = cls.parse('Sheet1')

# 会员密码信息
lr = os.getcwd() + os.sep + 'db' + os.sep + 'login.xls'
lr = pd.ExcelFile(lr)
lr = lr.parse('Sheet1')



# ===========================会员管理表
# admin_table = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
# admin_table['ID'] = cls['MEMBER_NO']
# admin_table['FFP_DATE'] = cf['FFP_DATE']
# admin_table['FFP_TIER'] = cf['FFP_TIER']
# admin_table['AGE'] = cf['AGE']
# admin_table['FLIGHT_COUNT'] = cf['FLIGHT_COUNT']
# admin_table['CLUSTER'] = cls['CLUSTER']
# admin_table['AVG_DISCOUNT'] = cf['avg_discount']
# # admin_table.AGE.astype(int)
# # # 写入Excel文件
# admin_table.to_excel('./db/admin.xls', index=False)

# ===========================注册登录表
# login_table = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
# login_table['ID'] = cls['MEMBER_NO']
# # 初始密码都设置为123456
# login_table['PASSWORD'] = '123456'
# # 加入管理员的信息
# admin_data = []
# admin_data.insert(0, {'ID':0,'PASSWORD':'123456'})
# login_table = pd.concat([pd.DataFrame(admin_data), login_table], ignore_index=True)
# login_table.to_excel('./db/login.xls', index=False)
# print(login_table)


# ===========================个人中心表
# me_table = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
# me_table['ID'] = cls['MEMBER_NO']
# me_table['FFP_DATE'] = cf['FFP_DATE']
# me_table['FFP_TIER'] = cf['FFP_TIER']
# me_table['AGE'] = cf['AGE']
# me_table['WORK_CITY'] = cf['WORK_CITY']
# me_table['WORK_PROVINCE'] = cf['WORK_PROVINCE']
# me_table['WORK_COUNTRY'] = cf['WORK_COUNTRY']
# me_table['FLIGHT_COUNT'] = cf['FLIGHT_COUNT']
# me_table['PASSWORD'] = lr['PASSWORD']
# print(me_table)
# me_table.to_excel('./db/personal_center.xls', index=False)




# ============================商品推荐表


# ============================会员积分明细表
# member_points_table = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
# member_points_table['ID'] = cls['MEMBER_NO']
# member_points_table['SEG_KM_SUM'] = cf['SEG_KM_SUM']
# member_points_table['EXCHANGE_COUNT'] = cf['EXCHANGE_COUNT']
# member_points_table['EP_SUM'] = cf['EP_SUM']
# member_points_table['ADD_POINTS_SUM'] = cf['ADD_Point_SUM']
# member_points_table['POINTS_SUM'] = cf['Points_Sum']
# print(member_points_table)
# member_points_table.to_excel('./db/member_points_table.xls', index=False)

# ============================会员积分兑换表
# exchange_points_table = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
