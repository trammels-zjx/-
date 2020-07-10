# -*- coding:utf-8 -*-
import pandas as pd
# import matplotlib
# import numpy as np
# import matplotlib.pyplot as plt
import jieba as jb
import re
# from collections import Counter
# from wordcloud import WordCloud

df = pd.read_excel('附件2.xlsx')
# df = df[['cat', 'review']]
print("数据总量: %d ." % len(df))
df.drop(['留言编号', '留言时间', '留言主题', '留言用户'], axis=1, inplace=True)

# 清洗空值
# print("在 cat 列中总共有 %d 个空值." % df['留言详情'].isnull().sum())
# print("在 review 列中总共有 %d 个空值." % df['一级标签'].isnull().sum())
# df = df[df.isnull().values==True]
# df = df[pd.notnull(df['review'])]

# 统计一下各个类别的数据量
d = {'cat': df['一级标签'].value_counts().index, 'count': df['一级标签'].value_counts()}
df_cat = pd.DataFrame(data=d).reset_index(drop=True)
# print(df_cat)

# 各个类别的分布图
# df_cat.plot(x='cat', y='count', kind='bar', legend=False,  figsize=(8, 5))
# plt.title("类目数量分布")
# plt.ylabel('数量', fontsize=18)
# plt.xlabel('类目', fontsize=18)
# plt.show()

# 添加id
df['cat_id'] = df['一级标签'].factorize()[0]

cat_id_df = df[['一级标签', 'cat_id']].drop_duplicates().sort_values('cat_id').reset_index(drop=True)
cat_to_id = dict(cat_id_df.values)
id_to_cat = dict(cat_id_df[['cat_id', '一级标签']].values)


# print(cat_id_df)
# print(cat_to_id)
# print(id_to_cat)


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
# 分词，并过滤停用词
df['cut_review'] = df['clean_review'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))

# # 词云
# def generate_wordcloud(tup):
#     wordcloud = WordCloud(background_color='white',
#                           font_path='simhei.ttf',
#                           max_words=50, max_font_size=40,
#                           random_state=42
#                           ).generate(str(tup))
#     # plt.imshow(wordcloud)
#     # plt.axis("off")
#     # plt.show()
#     return wordcloud
#
#
# cat_desc = dict()
# for cat in cat_id_df.一级标签.values:
#     text = df.loc[df['一级标签'] == cat, 'cut_review']
#     text = (' '.join(map(str, text))).split(' ')
#     cat_desc[cat] = text
#
# fig, axes = plt.subplots(7, 2, figsize=(30, 38))
# k = 0
# for i in range(7):
#     for j in range(2):
#         cat = id_to_cat[k]
#         most100 = Counter(cat_desc[cat]).most_common(100)
#         ax = axes[i, j]
#         ax.imshow(generate_wordcloud(most100), interpolation="bilinear")
#         ax.axis('off')
#         ax.set_title("{} Top 100".format(cat), fontsize=30)
#         k += 1


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(df['cut_review'], df['cat_id'], random_state=0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)

# 检测模型预测数据概率
X_test1 = count_vect.transform(X_test)
a = clf.score(X_test1,y_test)
print(X_test)

def myPredict(sec):
    format_sec = " ".join([w for w in list(jb.cut(remove_punctuation(sec))) if w not in stopwords])
    pred_cat_id = clf.predict(count_vect.transform([format_sec]))
    print(id_to_cat[pred_cat_id[0]])
    return id_to_cat[pred_cat_id[0]]


# 分类
df1 = pd.read_excel('附件2（测试数据）.xlsx')
df1.drop(['留言时间', '留言主题', '留言用户'], axis=1, inplace=True)

for i, item in enumerate(df1["留言详情"]):
    a = df1['留言详情'][i]
    df1.iloc[i, -1] = myPredict(a)

df1.drop(['留言详情'], axis=1, inplace=True)
df1.to_excel("数据三（已分类数据））.xlsx", index=False)
