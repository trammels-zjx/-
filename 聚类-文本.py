from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

from sklearn.cluster import KMeans
import pandas as pd
# import numpy as np
import jieba
import re
# from sklearn import metrics


df = pd.read_excel('附件3.xlsx')
# df = df[['cat', 'review']]
print("数据总量: %d ." % len(df))
df.drop(['留言编号', '留言主题', '留言用户'], axis=1, inplace=True)


# 定义删除除字母,数字，汉字以外的所有符号的函数
def remove_punctuation(line):
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', line)
    return line


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


# 加载停用词
stopwords = stopwordslist("chineseStopWords.txt")

# 删除除字母,数字，汉字以外的所有符号
df['clean_review'] = df['留言详情'].apply(remove_punctuation)



print(df['clean_review'].values)


vectorizer = CountVectorizer()
data = df['clean_review'].values

X = vectorizer.fit_transform([" ".join([b for b in jieba.cut(a)]) for a in data])

tfid = TfidfTransformer()

X = tfid.fit_transform(X.toarray())


# def easy_get_parameter_k_means(data):
#
#     test_score = []
#
#     n_clusters_end = 10
#     n_clusters_start = 5
#
#     while n_clusters_start <= n_clusters_end:
#
#         km = KMeans(n_clusters=n_clusters_start)
#         km.fit(data)
#
#         clusters = km.labels_.tolist()
#
#         score = metrics.silhouette_score(X=X,labels=clusters)
#
#         num = sorted([(np.sum([1 for a in clusters if a==i]),i) for i in set(clusters)])[-1]
#
#         test_score.append([n_clusters_start,score,num[0],num[1]])
#
#         print(n_clusters_start)
#         n_clusters_start += 1
#         # print(clusters)
#
#     return pd.DataFrame(test_score,columns=['共分了几类','分数','最大类包含的个数','聚类的名称']).sort_values(by ='分数',ascending =False)
#
# # 得到最佳参数=6
# a = easy_get_parameter_k_means(X.toarray())
# print(a)


# 得到最佳参数=6
km = KMeans(n_clusters=6)
km.fit(X.toarray())
clusters = km.labels_.tolist()
print(clusters)
df['class'] = clusters
print(df)

num = []
for i in range(6):
    a = df.groupby(['class']).size()[i]
    num.append(a)

df["热度指数"] = 0

for i, item in enumerate(df["class"]):
    df.iloc[i, -1] = 0.92 * num[item] + 0.08*(df["点赞数"][i] - df["反对数"][i])
print(df)

a = df.sort_values(by='热度指数',ascending = False)

a.to_csv("聚类.csv")
