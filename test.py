import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.externals import joblib 
import jieba

# load the model from disk
loaded_model = joblib.load("model/filter_text1.pkl")
vectorizer = joblib.load("model/vectorizer.pickle")
tfidf_transformer = joblib.load("model/tfidf_transformer.pickle")


test_data = pd.DataFrame([
    "您好！截至到2014年08月18日14时06分，您已使用数据流量0.0MB，套餐内剩余流量0.0MB，感谢您使用流量查询服务。中国移动",
    "乐享无线城市，五月好运好礼领航！新用户成功注册就抽奖，老朋友幸运号码赢好运。辽宁无线城市诚邀新友、旧交一起来把精彩好礼抱回家！详情点击：http://shenyang.wap.wxcs.cn",
    "星期天来我家吃饭",
    "孟先生這是大柏科技有限公司鄭先生（大柏）手機號13901053814他們有VOD及POS機系統你可以直接跟他約",
    "【百度】动态验证码为409160,请在3分钟内填写",
    "您尾号7544卡5日14:13ATM支出(ATM取款)1,000元，手续费2元。【工商银行】",
    "国五条细则已经落地多日，我挑选了几套不受新政影响的房子，如你还想在世纪城购房置业，请电话 焦宝生13641384093【中原地产】"
])[0].apply(lambda x: ' '.join(jieba.cut(x))) 

X_input_termcounts = vectorizer.transform(test_data)
X_input_tfidf = tfidf_transformer.transform(X_input_termcounts)

# print(X_input_tfidf)

print(loaded_model.predict(X_input_tfidf))