# -*-coding: utf-8-*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import warnings
import glob
import os
import datetime
reload(sys)
sys.setdefaultencoding('utf-8')

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_row', 10)

# 读取数据
data = pd.read_csv(u'E:\影子计划\交易数据\交易数据.csv', index_col='cust_no')

data.replace('None', 0, inplace=True)

data = data[['earlist_deal_date', 'cur_overdue_status', 'cur_overdue_max_days', 'his_overdued_max_days',
             'his_prepaid_max_days', 'due_prepaid_period', 'due_overduepaid_period', 'total_amt', 'overdue_amt',
             'total_deals', 'total_deals_p3', 'total_deals_p6', 'total_deals_p12', 'std_total_t_facility',
             'average_amt', 'month_after_facility', 'average_trans_usage_rate', 'facility_date_format']]


def string_toDatetime(string):
    """把字符串转成datetime"""
    return datetime.datetime.strptime(string, "%Y-%m-%d")


def days_between(d2, d1):
    d2 = data['facility_date_format']
    d1 = data['earlist_deal_date']
    return (d2 - d1).days


data['earlist_deal_date'] = data['earlist_deal_date'].apply(string_toDatetime)
data['facility_date_format'] = data['facility_date_format'].apply(string_toDatetime)

data['firstdeal_afterfacility'] = [int(i.days) for i in (data['facility_date_format'] - data['earlist_deal_date'])]

data['deals_per_month'] = data['total_deals'] / np.where(data['month_after_facility'] == 0, 1,
                                                         data['month_after_facility'])

del data['earlist_deal_date']
del data['facility_date_format']
del data['month_after_facility']

data = data.astype('float', inplace=True)

numsamples = len(data)
zhfont_kai = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simkai.ttf')
mark = ['Dr', 'Db', 'Dg', 'Dk', 'Dc', 'Dm', 'Dy', 'sr', 'sm', 'sk', 'sy', '^k', '+b', 'sr', 'db', '<g', 'pc']

column_x = data.columns

# 画单维度变量小提琴图
for i, col_x in zip(range(len(column_x)), column_x):
    print i, col_x
    sns.violinplot(data[col_x], width=0.5)
    plt.savefig(u'E:/影子计划/交易数据/小提琴图/' + str(i) + '_' + col_x + '.png')
    plt.close()
print 'over'

# 肘部法则选取最佳k值


for i, col_x in zip(range(len(column_x)), column_x):
    print i, col_x
    locals()['df' + str(i)] = data[[col_x]]
    a = locals()['df' + str(i)]
    K = range(1, 10)
    meandistortions = []
    for k in K:
        k_means = KMeans(n_clusters=k)
        k_means.fit(a)
        meandistortions.append(
            sum(np.min(cdist(a, k_means.cluster_centers_, "euclidean"), axis=1)) /
            a.shape[0])
    plt.plot(K, meandistortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel(u'平均畸变程度', fontproperties=zhfont_kai)
    plt.title(u'用肘部法则来确定最佳的K值', fontproperties=zhfont_kai)
    plt.show()
    plt.savefig(u'E:/影子计划/交易数据/肘部法则K值图/' + str(i) + '_' + col_x + '.png')
    plt.close()
print 'over'

# kmeans 聚类，通过每一簇均值作为质心排序标签


for i, col_x in zip(range(len(column_x)), column_x):
    print i, col_x
    a = data[[col_x]]
    k = [2, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3]
    k_means = KMeans(n_clusters=k[i])
    k_means.fit(a)
    # 获取聚类的中心点，根据结果查看，中心点数组中的下标就是对应的类别
    centers = k_means.cluster_centers_
    centroids = k_means.labels_
    # 按大小排序获得对应的下表
    index_order = np.argsort(centers.flatten()).tolist()
    for j in range(numsamples):
        centroids[j] = index_order.index(centroids[j])

    result = np.array(centroids).tolist()
    a['tag_' + col_x] = result
    order = [index_order.index(n) for n in range(len(index_order))]
    name = order
    values = centers.tolist()
    nvs = zip(name, values)
    nvDict = dict((name, values) for name, values in nvs)
    a['dic_' + col_x] = str(nvDict)
    if i < 10:
        a.to_csv(u'E:/影子计划/交易数据/聚类后结果/' + '0' + str(i) + '_' + col_x + '.csv')
    else:
        a.to_csv(u'E:/影子计划/交易数据/聚类后结果/' + str(i) + '_' + col_x + '.csv')
print 'over'

# 画各类别均值分布图，看分类是否按高低排序的

filenames = glob.glob(u'E:/影子计划/交易数据/聚类后结果/*.csv')
names = os.listdir(u'E:/影子计划/交易数据/聚类后结果/')

i = 0
for name in filenames:
    print i, filenames[i]

    df = pd.read_csv(filenames[i], index_col='cust_no')
    b = df.groupby(df.columns[-2]).mean()
    b = b.sort_values(b.columns[0], ascending=False)
    b.plot.bar(rot=0)
    n = os.path.splitext(names[i])[0]
    plt.title(n)
    plt.savefig(u'E:/影子计划/交易数据/均值分布图/' + n + '.png')
    i = i + 1
print 'over'

# 有两个字段是不根据聚类分的，需要根据经验重新分类
data1 = pd.read_csv(u'E:/影子计划/交易数据/聚类后结果/01_cur_overdue_max_days.csv', index_col='cust_no')
data2 = pd.read_csv(u'E:/影子计划/交易数据/聚类后结果/02_his_overdued_max_days.csv', index_col='cust_no')

data1['tag_cur_overdue_max_days'] = np.where(data1.cur_overdue_max_days == 0, 0, 7)
data1['tag_cur_overdue_max_days'] = np.where((data1.cur_overdue_max_days >= 1) & (data1.cur_overdue_max_days <= 3), 1,
                                             data1['tag_cur_overdue_max_days'])
data1['tag_cur_overdue_max_days'] = np.where((data1.cur_overdue_max_days >= 4) & (data1.cur_overdue_max_days <= 10), 2,
                                             data1['tag_cur_overdue_max_days'])
data1['tag_cur_overdue_max_days'] = np.where((data1.cur_overdue_max_days >= 11) & (data1.cur_overdue_max_days <= 17), 3,
                                             data1['tag_cur_overdue_max_days'])
data1['tag_cur_overdue_max_days'] = np.where((data1.cur_overdue_max_days >= 18) & (data1.cur_overdue_max_days <= 24), 4,
                                             data1['tag_cur_overdue_max_days'])
data1['tag_cur_overdue_max_days'] = np.where((data1.cur_overdue_max_days >= 25) & (data1.cur_overdue_max_days <= 33), 5,
                                             data1['tag_cur_overdue_max_days'])
data1['tag_cur_overdue_max_days'] = np.where((data1.cur_overdue_max_days >= 34) & (data1.cur_overdue_max_days <= 63), 6,
                                             data1['tag_cur_overdue_max_days'])

data2['tag_his_overdued_max_days'] = np.where(data2.his_overdued_max_days == 0, 0, 7)
data2['tag_his_overdued_max_days'] = np.where((data2.his_overdued_max_days >= 1) & (data2.his_overdued_max_days <= 3), 1,
                                             data2['tag_his_overdued_max_days'])
data2['tag_his_overdued_max_days'] = np.where((data2.his_overdued_max_days >= 4) & (data2.his_overdued_max_days <= 10), 2,
                                             data2['tag_his_overdued_max_days'])
data2['tag_his_overdued_max_days'] = np.where((data2.his_overdued_max_days >= 11) & (data2.his_overdued_max_days <= 17), 3,
                                             data2['tag_his_overdued_max_days'])
data2['tag_his_overdued_max_days'] = np.where((data2.his_overdued_max_days >= 18) & (data2.his_overdued_max_days <= 24), 4,
                                             data2['tag_his_overdued_max_days'])
data2['tag_his_overdued_max_days'] = np.where((data2.his_overdued_max_days >= 25) & (data2.his_overdued_max_days <= 33), 5,
                                             data2['tag_his_overdued_max_days'])
data2['tag_his_overdued_max_days'] = np.where((data2.his_overdued_max_days >= 34) & (data2.his_overdued_max_days <= 63), 6,
                                             data2['tag_his_overdued_max_days'])

match = {0: 'WYQ', 1: 'K', 2: 'Y1', 3: 'Y2', 4: 'Y3', 5: 'Y4', 6: 'M2', 7: 'M2+'}
data1['dic_cur_overdue_max_days'] = str(match)
data2['dic_his_overdued_max_days'] = str(match)

data1.to_csv(u'E:/影子计划/交易数据/聚类后结果/01_cur_overdue_max_days.csv')
data2.to_csv(u'E:/影子计划/交易数据/聚类后结果/02_his_overdued_max_days.csv')


# 将每个维度的簇都合在一起，形成dataframe


filenames = glob.glob(u'E:/影子计划/交易数据/聚类后结果/*.csv')
names = os.listdir(u'E:/影子计划/交易数据/聚类后结果/')

for i in range(len(filenames)):
    locals()['df' + str(i)] = pd.read_csv(filenames[i], index_col='cust_no')

for i in range(len(filenames) - 1):
    locals()['df' + str(i + 1)] = pd.concat([locals()['df' + str(i)], locals()['df' + str(i + 1)]], axis=1)

df = locals()['df' + str(len(filenames) - 1)]

df_columns = df.columns.tolist()
tag_columns = [elem for elem in df_columns if ((elem[:4] == 'tag_'))]
dic_columns = [elem for elem in df_columns if ((elem[:4] == 'dic_'))]

tag = df[tag_columns]
dic = df[dic_columns]
dic = dic.reset_index(drop=True).drop_duplicates().T

tag.to_csv(u'E:/影子计划/交易数据/tag_trade.csv')
dic.to_csv(u'E:/影子计划/交易数据/dic_trade.csv')

'''
# 计算轮廓系数k = 2, 0.6337
coeff = metrics.silhouette_score(feature_repay2repay_page_next_9_18, model.labels_, metric='euclidean')
d = cdist(feature_repay2repay_page_next_9_18, model.cluster_centers_, 'euclidean')

for i in range(numsamples):
    plt.plot(feature_repay2repay_page_next_9_18.ix[i, 0], mark[model.labels_[i]])
plt.show()
'''
