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
data = pd.read_csv(u'E:/影子计划/app_behavior_new.csv', index_col='cust_no')

# data中frequent_network_type为文本字段，需整理成数值形式

data['frequent_network_type_new'] = np.where(data.frequent_network_type == '4g', 2, 0)
data['frequent_network_type_new'] = np.where(data.frequent_network_type == 'wifi', 1, data['frequent_network_type_new'])

del data['frequent_phone_maker']
del data['last_phone_model']
del data['frequent_network_type']
data.replace('None', 0, inplace=True)
data.fillna('0', inplace=True)
data = data.astype('float', inplace=True)

# 数据有多维情况，需要归一化cut 按0-1规整
column_x = data.columns

i = 0
for col_x in column_x:
    print i, col_x
    i = i + 1

    data[col_x] = (data[col_x] - min(data[col_x])) / abs(max(data[col_x]) - min(data[col_x]))

numsamples = len(data)
zhfont_kai = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simkai.ttf')
mark = ['Dr', 'Db', 'Dg', 'Dk', 'Dc', 'Dm', 'Dy', 'sr', 'sm', 'sk', 'sy', '^k', '+b', 'sr', 'db', '<g', 'pc']

df0 = data[['appointment2appointment_page_open', 'basic2app_open_loan']]
df1 = data[['basic2app_open', 'active_event_days', 'use_times', 'stay_time(hour)']]
df2 = data[['basic2app_open_discover', 'home2pay_day_click', 'billcheck_times', 'sum_discover', 'sum_credit2credit',
            'profile2card_click']]
df3 = data[['home2interest_free_click']]
df4 = data[['creditbank_num']]
df5 = data[['basic2app_open_me', 'login2login_page_submit']]
df6 = data[['mgm2app_share_wechatfriend']]
df7 = data[['mgm2app_share_wechatmoments']]
df8 = data[['sum_mgm_click', 'sum_mgm_appshare']]
df9 = data[['sum_billcheckshare_times']]
df10 = data[['cardmanage2autopay_suc', 'profile2repay_click', 'depositbind2auto_refund', 'repay2repay_page_next',
             'days_between_lastlogin_repay']]
df11 = data[['cardmanage2unbind_suc']]
df12 = data[['home2news_click']]
df13 = data[['sum_profile2coupon']]
df14 = data[['profile2help_click']]
df15 = data[['reset_trade_password2submit', 'reset_trade_password2submit_fail', 'reset_trade_password2submit_suc']]
df16 = data[['sum_bnefit', 'home2benefit_banner_click', 'benefit2banner_click', 'benefit2category_click',
             'benefit2detail_open']]
df17 = data[['borrow2borrow_page_next', 'cash2borrow_page_next']]
df18 = data[['avg_borrow_time', 'max_borrow_time', 'min_borrow_time']]
df19 = data[['cash2submit_suc', 'repay2submit_suc', 'borrow2submit_suc']]
df20 = data[['avg_repay_time', 'max_repay_time', 'min_repay_time']]
df21 = data[['total_device_num']]
df22 = data[['use_times_4g']]
df23 = data[['use_times_wifi']]
df24 = data[['frequent_network_type_new']]
df25 = data[['different_lbs_num']]
df26 = data[['app_push2app_push_click_hktx']]
df27 = data[['app_push2app_push_click_jkyh']]
df28 = data[['app_push2app_push_click_teyh']]
df29 = data[['app_push2app_push_click_xykyh']]
df30 = data[['app_push2app_push_click_yqtx']]

list = ['BorrowDemand', 'Activeness', 'FocusOn_Card', 'FocusOn_InterestFree', 'creditbank_num',
        'FocusOn_core_function', 'mgm2app_share_wechatfriend', 'mgm2app_share_wechatmoments',
        'RaiseLimitDemand', 'DesirePerf', 'RepayWilling', 'cardmanage2unbind_suc',
        'FocusOn_CompNews', 'Interest_Sensitive', 'Consult_Complain', 'reset_trade_password',
        'FocusOn_benifit', 'TradeWilling', 'BorrowHesitation', 'Trade_frequency', 'RepayHesitation',
        'device_update frequency', 'use_times_4g', 'use_times_wifi', 'frequent_network_type',
        'lbs_stability', 'repay_warning_prefer', 'borrow_tempt_Prefer', 'RaiseLimit_Prefer',
        'creditbenefit_prefer', 'overdue_warning_prefer']

# 肘部法则选取最佳k值
for i in range(0, 31):
    a = locals()['df' + str(i)]
    K = range(1, 10)
    meandistortions = []
    for k in K:
        k_means = KMeans(n_clusters=k)
        k_means.fit(a)
        meandistortions.append(sum(np.min(
            cdist(a, k_means.cluster_centers_,
                  'euclidean'), axis=1)) / a.shape[0])
    plt.plot(K, meandistortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel(u'平均畸变程度', fontproperties=zhfont_kai)
    plt.title(u'用肘部法则来确定最佳的K值', fontproperties=zhfont_kai)
    plt.show()
    plt.savefig(u'E:/影子计划/app行为/肘部法则K值图/' + str(i) + '_' + list[i] + '.png')
    plt.close()
    print i, list[i]
print 'over'

# kmeans 聚类，通过每一簇均值作为质心排序标签


for i in range(0, 31):
    a = locals()['df' + str(i)]
    k = [4, 4, 4, 4, 3, 4, 4, 4, 4, 3, 4, 3, 6, 4, 6, 4, 5, 4, 5, 4, 5, 4, 3, 3, 3, 5, 3, 3, 3, 3, 4]
    k_means = KMeans(n_clusters=k[i])
    k_means.fit(a)
    # 获取聚类的中心点，根据结果查看，中心点数组中的下标就是对应的类别
    centroids = k_means.labels_
    centers = k_means.cluster_centers_
    centerSquare = centers * centers
    dis = np.sqrt(centerSquare.sum(axis=1))
    # 按大小排序获得对应的下表
    index_order = np.argsort(dis.flatten()).tolist()
    for j in range(numsamples):
        centroids[j] = index_order.index(centroids[j])

    result = np.array(centroids).tolist()
    a['tag_' + list[i]] = result
    order = [index_order.index(n) for n in range(len(index_order))]
    name = order
    values = centers.tolist()
    nvs = zip(name, values)
    nvDict = dict((name, values) for name, values in nvs)
    a['dic_' + list[i]] = str(nvDict)
    if i < 10:
        a.to_csv(u'E:/影子计划/app行为/聚类后结果/' + '0' + str(i) + '_' + list[i] + '.csv')
    else:
        a.to_csv(u'E:/影子计划/app行为/聚类后结果/' + str(i) + '_' + list[i] + '.csv')
    print i, list[i]
print 'over'

# 画各类别均值分布图，看分类是否按高低排序的

filenames = glob.glob(u'E:/影子计划/app行为/聚类后结果/*.csv')
names = os.listdir(u'E:/影子计划/app行为/聚类后结果/')

i = 0
for name in filenames:
    print i, filenames[i]

    df = pd.read_csv(filenames[i], index_col='cust_no')
    b = df.groupby(df.columns[-2]).mean()
    b = b.sort_values(b.columns[0], ascending=False)
    b.plot.bar(rot=0)
    n = os.path.splitext(names[i])[0]
    plt.title(n)
    plt.savefig(u'E:/影子计划/app行为/均值分布图/' + n + '.png')
    i = i + 1
print 'over'

# 将每个维度的簇都合在一起，形成dataframe


filenames = glob.glob(u'E:/影子计划/app行为/聚类后结果/*.csv')
names = os.listdir(u'E:/影子计划/app行为/聚类后结果/')

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

tag.to_csv(u'E:/影子计划/app行为/tag_app_behavior.csv')
dic.to_csv(u'E:/影子计划/app行为/dic_app_behavior.csv')

'''
# 计算轮廓系数k = 2, 0.6337
coeff = metrics.silhouette_score(feature_repay2repay_page_next_9_18, model.labels_, metric='euclidean')
d = cdist(feature_repay2repay_page_next_9_18, model.cluster_centers_, 'euclidean')

for i in range(numsamples):
    plt.plot(feature_repay2repay_page_next_9_18.ix[i, 0], mark[model.labels_[i]])
plt.show()
'''
