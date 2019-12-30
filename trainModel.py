# -*- coding: utf-8 -*-
import pandas as pd
import json
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib 

# 加载原始数据，进行分割
def load_message(path):
    
    lines = []
    datas = []

    with open(path, 'r') as fr:
                
        for i in range(10973):
            line = fr.readline()
            lines.append(line)
        num = len(lines)
        for i in range(num):
            message = lines[i].split('\t')
            datas.append([message[0], message[1]])
    return datas

# 分词
def split_word(dataSource):
  data = pd.DataFrame(dataSource, columns = ['label', 'content'])
  data["分词短信"] = data["content"].apply(lambda x:' '.join(jieba.cut(x)))
  return data

# get train data and test data
def get_train_and_test_data(data): 
  print(data.head())
  X = data["分词短信"].values
  y = data["label"]

  train_X, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)

  return train_X, test_x, train_y, test_y

def handle_feature(data):  
  vectorizer = CountVectorizer()
  X_train_termcounts = vectorizer.fit_transform(data)

  tfidf_transformer = TfidfTransformer()
  X_train_tfidf = tfidf_transformer.fit_transform(X_train_termcounts)
  return X_train_tfidf, vectorizer, tfidf_transformer  

def get_score(test_data):  
  X_input_termcounts = vectorizer.transform(test_data)
  X_input_tfidf = tfidf_transformer.transform(X_input_termcounts)

  predicted_categories = classifier.predict(X_input_tfidf)

  score = accuracy_score(test_y, predicted_categories)
  return score

dataSource = load_message("dataset/train.txt")

data = split_word(dataSource)

train_X, test_x, train_y, test_y = get_train_and_test_data(data)

X_train_tfidf, vectorizer, tfidf_transformer = handle_feature(train_X)

classifier = MultinomialNB().fit(X_train_tfidf, train_y)

score = get_score(test_x)

print(score)

# category_map = {'0': "正常", '1': "垃圾"}

# for sentence, category, real in zip(test_x, predicted_categories, test_y):
#     print("\n短信内容：", sentence, "\nPredicted 分类：", category, "真实值：", real)

# test

# test_data = pd.DataFrame([
#     "您好！截至到2014年08月18日14时06分，您已使用数据流量0.0MB，套餐内剩余流量0.0MB，感谢您使用流量查询服务。中国移动",
#     "乐享无线城市，五月好运好礼领航！新用户成功注册就抽奖，老朋友幸运号码赢好运。辽宁无线城市诚邀新友、旧交一起来把精彩好礼抱回家！详情点击：http://shenyang.wap.wxcs.cn",
#     "星期天来我家吃饭",
#     "孟先生這是大柏科技有限公司鄭先生（大柏）手機號13901053814他們有VOD及POS機系統你可以直接跟他約",
#     "【百度】动态验证码为409160,请在3分钟内填写",
#     "您尾号7544卡5日14:13ATM支出(ATM取款)1,000元，手续费2元。【工商银行】",
#     "电话很多很多基督教"
# ])[0].apply(lambda x: ' '.join(jieba.cut(x))) 

# X_input_termcounts = vectorizer.transform(test_data)
# X_input_tfidf = tfidf_transformer.transform(X_input_termcounts)

# print(classifier.predict(X_input_tfidf))

filename = 'model/filter_text1.pkl'

# Save the model as a pickle in a file 
joblib.dump(classifier, filename) 

joblib.dump(vectorizer, open("model/vectorizer.pickle", "wb"))
joblib.dump(tfidf_transformer, open("model/tfidf_transformer.pickle", "wb"))

# load the model from disk
loaded_model = joblib.load(filename)
test_data = pd.DataFrame([
    "您好！截至到2014年08月18日14时06分，您已使用数据流量0.0MB，套餐内剩余流量0.0MB，感谢您使用流量查询服务。中国移动",
    "乐享无线城市，五月好运好礼领航！新用户成功注册就抽奖，老朋友幸运号码赢好运。辽宁无线城市诚邀新友、旧交一起来把精彩好礼抱回家！详情点击：http://shenyang.wap.wxcs.cn",
    "星期天来我家吃饭",
    "孟先生這是大柏科技有限公司鄭先生（大柏）手機號13901053814他們有VOD及POS機系統你可以直接跟他約",
    "【百度】动态验证码为409160,请在3分钟内填写",
    "您尾号7544卡5日14:13ATM支出(ATM取款)1,000元，手续费2元。【工商银行】",
    "电话很多很多基督教"
])[0].apply(lambda x: ' '.join(jieba.cut(x))) 

X_input_termcounts = vectorizer.transform(test_data)
X_input_tfidf = tfidf_transformer.transform(X_input_termcounts)

print(loaded_model.predict(X_input_tfidf))