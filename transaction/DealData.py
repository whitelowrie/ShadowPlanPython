# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.cluster import  KMeans
from sklearn import  metrics
from scipy.spatial.distance import cdist
import warnings
import glob
import os
import datetime

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_row', 10)

data = pd.read_csv(u'/Users/bigdata/Desktop/画像_80655/交易数据(20170316).csv',index_col='cust_no')

data.replace('None', 0, inplace=True)

#earlist_deal_date      最早交易时间
#cur_overdue_status     当前是否逾期
#cur_overdue_max_days   最大逾期天数
#his_overdued_max_days  历史最大逾期天数
#his_prepaid_max_days   历史最大提前还款天数
#due_prepaid_period     历史到期并且提前还款数
#total_amt              交易总金额
#overdue_amt            逾期总金额
#total_deals            交易总笔数
#total_deals_p3         三期总金额
#total_deals_p6         六期总金额
#total_deals_p12        十二期总金额
#std_total_t_facility   交易金额
#average_amt            平均交易金额
#month_after_facility   最近交易距授信月份
#average_trans_usage_rate 笔均授信额度使用率
#facility_date_format   授信日期

data = data[['earlist_deal_date',
             'cur_overdue_status',
             'cur_overdue_max_days',
             'his_overdued_max_days',
             'his_prepaid_max_days',
             'due_prepaid_period',
             'total_amt',
             'overdue_amt',
             'total_deals',
             'total_deals_p3',
             'total_deals_p6',
             'total_deals_p12',
             'std_total_t_facility',
             'average_amt',
             'month_after_facility',
             'average_trans_usage_rate',
             'facility_date_format']]

def string_toDateTime(string):
    return datetime.datetime.strptime(string, "%Y-%m-%d")

data['firstdeal_afterfacility'] = map(lambda x: x.days, data['facility_date_format'].apply(string_toDateTime) - data['earlist_deal_date'].apply(string_toDateTime))

del data['earlist_deal_date']
del data['facility_date_format']
del data['month_after_facility']

data = data.astype('float', inplace=True)

numsamples = len(data)
# zhfont_kai = matplotlib.font_manager.FontProperties(fname="")
mark = ['Dr', 'Db', 'Dg', 'Dk', 'Dc', 'Dm', 'Dy', 'sr', 'sm', 'sk', 'sy', '^k', '+b', 'sr', 'db', '<g', 'pc']

columns_x = data.columns[3:]

# print data


# for i, col_x in zip(range(len(columns_x)), columns_x):
#     print i,col_x
#     sns.violinplot(data[col_x], width=0.5)
#     plt.savefig(u'/Users/bigdata/Desktop/画像_80655/ShadowPlan/' + str(i) + '_' + col_x + ".png")
#     plt.close()

# print locals()
# #
# for i, col_x in zip(range(len(columns_x)), columns_x):
#     locals()['df' + str(i)] = data[[col_x]]
#     a = locals()['df' + str(i)]
#     K = range(1, 10)
#     meandistortions = []
#     for k in K:
#         k_means = KMeans(n_clusters=k)
#         k_means.fit(a)
#         meandistortions.append(
#             sum(np.min(cdist(a, k_means.cluster_centers_, "euclidean"), axis=1)) /
#             a.shape[0])
#     plt.plot(K, meandistortions, 'bx-')
#     plt.xlabel('k')
#     plt.ylabel(u'平均畸变程度')
#     plt.title(u'用肘部法则来确定最佳的K值')
#     plt.savefig(u'/Users/bigdata/Desktop/画像_80655/zhoubuK/' + str(i) + '_' + col_x + '.png')
#     plt.close()


print locals()